"""Phase 2: translate English questions to Bengali with Gemma 4.

Reads a questions CSV (needs qid and question_en; default data/interim/questions_en.csv),
translates the selected rows one at a time and appends each result to
data/interim/translations_bn.csv as soon as it arrives. An interrupted run resumes where
it stopped: qids already in the output are skipped. Failed calls write no row, so a rerun
retries them.

Everything is logged under logs/translation/:
  runs.jsonl    a "start" and an "end" line per run: arguments, model, generation config,
                rate limits, prompt template and its hash, input file hash, the qids to do,
                and the totals
  calls.jsonl   every API attempt: full prompt, raw response, finish reason, token usage,
                latency, errors and retries

Each output row carries the automatic checks from plan 2D in `flags`:
  numbers  a number in the English is missing from the Bengali (Bengali digits allowed)
  math     a LaTeX block ($...$) did not come through unchanged
  length   Bengali/English length ratio outside 0.8-1.6 (math and URLs excluded)
  script   under half the letters are Bengali
  finish   the model stopped for a reason other than STOP (e.g. hit the token limit)

Usage:
  python scripts/translate.py --dry-run --sample 4             # show a prompt, call nothing
  python scripts/translate.py --sample 300 --seed 42 --split main   # pilot

Calls run in parallel (--workers, default 16) because each takes long; the shared rate
limiter keeps requests and tokens per minute under the limits. Rows are written in the
order calls finish.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv

import gemma

ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "logs"
LOG_DIR = LOG_ROOT / "translation"
MAX_CONSECUTIVE_FAILURES = 3  # stop instead of burning quota when the API keeps failing

PROMPT_VERSION = "v1"
PROMPT_TEMPLATE = """Translate this science question from English to Bengali.

Rules:
- Natural standard Bangladeshi Bengali (পানি, not জল). Translate meaning, not word for word.
- Scientific terms as in Bangladeshi textbooks and the Bangla Academy dictionary. On first use, add the English term in brackets: আপেক্ষিক তাপ (specific heat).
- Keep numbers, units, formulas, symbols, LaTeX ($...$), URLs, line breaks and list markers unchanged.
- Names of people and places in Bengali script: নিউটন, আইনস্টাইন.
- Do NOT answer. Do NOT add, remove or explain anything.
- Output only the translation.

Question:
{question_en}"""
PROMPT_ID = f"{PROMPT_VERSION}-{hashlib.sha256(PROMPT_TEMPLATE.encode('utf-8')).hexdigest()[:8]}"

LEN_RATIO = (0.8, 1.6)
MIN_BN_SHARE = 0.5
BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)  # "2nd" is usually translated as a word
MATH = re.compile(r"\$\$.+?\$\$|\$[^$]+?\$", re.S)
URL = re.compile(r"https?://\S+")
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
LABEL = re.compile(r"^\s*\**\s*(bengali translation|translation|বাংলা অনুবাদ|অনুবাদ)\s*\**\s*[:：]\s*", re.I)

OUTPUT_FIELDS = ["qid", "subject", "split", "question_en", "question_bn", "flags", "len_ratio",
                 "bn_share", "missing_numbers", "missing_math", "finish_reason", "model", "prompt_id",
                 "temperature", "top_p", "top_k", "seed", "max_output_tokens", "thinking", "input_tokens",
                 "output_tokens", "thinking_tokens", "latency_s", "attempts", "run_id", "translated_at"]
SETTINGS = ["temperature", "top_p", "top_k", "seed", "max_output_tokens"]  # blank = model default


def build_prompt(question_en):
    return PROMPT_TEMPLATE.format(question_en=question_en)


def clean_output(text):
    text = re.sub(r"^```\w*\n|\n?```$", "", text.strip()).strip()  # a model may wrap output in a fence
    return LABEL.sub("", text, count=1).strip()


def checks(en, bn, finish_reason):
    prose_en, prose_bn = URL.sub(" ", MATH.sub(" ", en)), URL.sub(" ", MATH.sub(" ", bn))
    bn_numbers = prose_bn.translate(BN_DIGITS).replace(",", "")
    missing_numbers = sorted({n for n in NUMBER.findall(ORDINAL.sub(" ", prose_en))
                              if n.replace(",", "") not in bn_numbers})
    missing_math = [m for m in MATH.findall(en) if m not in bn]
    len_ratio = len(prose_bn.strip()) / max(1, len(prose_en.strip()))
    n_bn, n_latin = len(BENGALI.findall(prose_bn)), len(LATIN.findall(prose_bn))
    bn_share = n_bn / max(1, n_bn + n_latin)
    flags = [name for name, bad in (
        ("numbers", missing_numbers),
        ("math", missing_math),
        ("length", not LEN_RATIO[0] <= len_ratio <= LEN_RATIO[1]),
        ("script", bn_share < MIN_BN_SHARE),
        ("finish", finish_reason != "STOP"),
    ) if bad]
    return {"flags": ";".join(flags), "len_ratio": round(len_ratio, 2), "bn_share": round(bn_share, 2),
            "missing_numbers": ";".join(missing_numbers), "missing_math": len(missing_math)}


def read_csv(path):
    csv.field_size_limit(2**31 - 1)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sample_by_subject(rows, n, seed):
    """Seeded sample of n rows, split as evenly as possible across subjects.

    Rows come back interleaved by subject, so --limit on a sample gives a mix of subjects
    and a later run without --limit continues the same sample.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["subject"]].append(r)
    picked = []
    for i, subject in enumerate(sorted(groups)):
        k = n // len(groups) + (i < n % len(groups))
        picked.append(rng.sample(groups[subject], min(k, len(groups[subject]))))
    return [r for batch in zip_longest(*picked) for r in batch if r is not None]


def select(rows, args):
    if args.subject:
        rows = [r for r in rows if r["subject"] in args.subject]
    if args.split:
        rows = [r for r in rows if r["split"] in args.split]
    if args.qids:
        wanted = set(args.qids.read_text(encoding="utf-8").split())
        rows = [r for r in rows if r["qid"] in wanted]
    if args.sample:
        rows = sample_by_subject(rows, args.sample, args.seed)
    return rows[:args.limit] if args.limit else rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_run(entry):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def open_output(path):
    new = not path.exists() or path.stat().st_size == 0  # empty: a run was killed before writing
    if not new:
        with open(path, encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if header != OUTPUT_FIELDS:
            sys.exit(f"{path} has different columns than this script writes; use a new --output")
    # utf-8-sig only when creating the file: appending with it would write a BOM mid-file.
    f = open(path, "w" if new else "a", encoding="utf-8-sig" if new else "utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
    if new:
        writer.writeheader()
        f.flush()
    return f, writer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "interim" / "questions_en.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "interim" / "translations_bn.csv")
    ap.add_argument("--subject", nargs="+", help="only these subjects")
    ap.add_argument("--split", nargs="+", choices=["main", "gap", "control"], help="only these splits")
    ap.add_argument("--qids", type=Path, help="file of qids to translate, whitespace-separated")
    ap.add_argument("--sample", type=int, help="seeded sample of N rows, even across subjects")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="translate at most N rows")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", gemma.DEFAULT_MODEL))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, help="nucleus sampling (default: model default)")
    ap.add_argument("--top-k", type=int, help="top-k sampling (default: model default)")
    ap.add_argument("--gen-seed", type=int, help="generation seed sent to the API (default: none)")
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    # Gemma 4 thinks by default and its thoughts count against max output tokens; a test
    # translation spent 2,046 of 2,048 tokens thinking and returned nothing. "minimal" turns it off.
    ap.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], default="minimal")
    ap.add_argument("--rpm", type=float, default=float(os.environ.get("GEMMA_RPM", 30)), help="requests/minute limit")
    ap.add_argument("--tpm", type=float, default=float(os.environ.get("GEMMA_TPM", 16000)), help="tokens/minute limit")
    ap.add_argument("--rpd", type=int, default=int(os.environ.get("GEMMA_RPD", 14400)), help="requests/day limit")
    ap.add_argument("--margin", type=float, default=float(os.environ.get("GEMMA_MARGIN", 0.2)),
                    help="stay this fraction below every limit (0.2 = use 80%%)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("GEMMA_WORKERS", 16)),
                    help="requests in flight at once; the limiter still caps requests and tokens per minute")
    return ap.parse_args()


def main():
    load_dotenv(ROOT / ".env")
    args = parse_args()
    rows = select(read_csv(args.input), args)
    done = {r["qid"] for r in read_csv(args.output)} if args.output.exists() else set()
    todo = [r for r in rows if r["qid"] not in done]
    print(f"selected {len(rows)}, already translated {len(rows) - len(todo)}, to do {len(todo)}")
    if args.dry_run:
        if todo:
            print(f"\n--- prompt {PROMPT_ID} for {todo[0]['qid']} ---\n{build_prompt(todo[0]['question_en'])}")
        return
    if not todo:
        return
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set; add it to .env")

    limiter = gemma.RateLimiter(args.rpm, args.tpm, args.rpd, args.margin, gemma.calls_today(LOG_ROOT, args.model))
    model = gemma.Gemma(LOG_DIR / "calls.jsonl", limiter, model=args.model, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k, seed=args.gen_seed,
                        max_output_tokens=args.max_output_tokens, thinking=args.thinking)
    run_id = f"{gemma.utc_now():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    log_run({
        "event": "start", "run_id": run_id, "time": gemma.utc_now().isoformat(timespec="seconds"),
        "args": vars(args), "model": model.model, "config": model.config, "model_info": model.model_info(),
        "limits_with_margin": {"min_gap_s": round(limiter.min_gap, 2), "tpm": limiter.tpm, "rpd": limiter.rpd},
        "requests_already_today": limiter.used_today,
        "prompt_id": PROMPT_ID, "prompt_template": PROMPT_TEMPLATE,
        "input_sha256": sha256(args.input), "n_selected": len(rows), "n_already_done": len(rows) - len(todo),
        "qids_todo": [r["qid"] for r in todo],
        "python": platform.python_version(), "google_genai": version("google-genai"),
    })

    totals, status, failures, failed = Counter(), "finished", 0, []
    started = time.monotonic()
    f, writer = open_output(args.output)
    recovered = recovered_calls(todo, model)
    for r in todo:
        if r["qid"] in recovered:
            writer.writerow(output_row(r, recovered[r["qid"]]))
    f.flush()
    if recovered:
        print(f"recovered {len(recovered)} translations from calls.jsonl that an earlier run never saved")
        log_run({"event": "recovered", "run_id": run_id, "qids": sorted(recovered)})
    totals["recovered"] = len(recovered)
    todo = [r for r in todo if r["qid"] not in recovered]

    # Calls run in parallel threads; the shared limiter keeps their starts under the limits.
    # Only this thread writes the CSV, in the order calls finish.
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(model.generate, build_prompt(r["question_en"]), run_id=run_id, task="translate",
                           qid=r["qid"], prompt_id=PROMPT_ID): r for r in todo}
    try:
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.cancelled():
                continue
            r = futures[fut]
            text, rec = fut.result()
            if rec["status"] == "cancelled":  # stopped before its next attempt; rerun picks it up
                continue
            usage = rec.get("usage") or {}
            totals["input_tokens"] += usage.get("prompt_token_count", 0)
            totals["output_tokens"] += usage.get("candidates_token_count", 0)
            totals["thinking_tokens"] += usage.get("thoughts_token_count", 0)
            totals["latency_s"] = round(totals["latency_s"] + rec.get("latency_s", 0), 2)
            if not text:
                totals["failed"] += 1
                failed.append(r["qid"])
                failures += 1
                print(f"[{i}/{len(todo)}] {r['qid']} FAILED ({rec.get('status')}, finish={rec.get('finish_reason')}"
                      f"{', ' + rec['error'][:120] if rec.get('error') else ''})")
                if failures >= MAX_CONSECUTIVE_FAILURES and status == "finished":
                    status = f"aborted after {failures} failures in a row"
                    cancel(futures, model)  # requests already sent finish and are still written
                continue
            failures = 0
            row = output_row(r, rec)
            writer.writerow(row)
            f.flush()
            totals["translated"] += 1
            totals["flagged"] += bool(row["flags"])
            print(f"[{i}/{len(todo)}] {r['qid']} {row['flags'] or 'ok'} ({rec['latency_s']}s)")
    except gemma.DailyLimitReached as e:
        status = f"stopped at daily cap: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
        running = sum(fut.running() for fut in futures)
        print(f"\nstopping: {running} requests already sent will finish first; their results are "
              "logged and recovered on the next run")
    finally:
        cancel(futures, model)
        pool.shutdown(wait=False, cancel_futures=True)
        f.close()
        totals["wall_time_s"] = round(time.monotonic() - started, 1)
        log_run({"event": "end", "run_id": run_id, "time": gemma.utc_now().isoformat(timespec="seconds"),
                 "status": status, "workers": args.workers, **totals, "failed_qids": failed})
        if failed:
            (LOG_DIR / "failed_qids.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
    print(f"\n{status}: {dict(totals)}")
    if failed:
        print(f"failed after all retries: {' '.join(failed)} (saved to {LOG_DIR / 'failed_qids.txt'}; "
              "a rerun tries them again)")


def recovered_calls(todo, model):
    """Successful calls in calls.jsonl whose rows never reached the CSV (the run stopped first).

    Only calls made with this model, prompt and generation config are reused.
    """
    path = LOG_DIR / "calls.jsonl"
    wanted = {r["qid"] for r in todo}
    found = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if (c.get("task") == "translate" and c.get("qid") in wanted and c["status"] == "ok"
                        and c["model"] == model.model and c.get("prompt_id") == PROMPT_ID
                        and c["config"] == model.config):
                    found[c["qid"]] = c
    return found


def output_row(r, rec):
    """One CSV row from a question and the logged call that translated it."""
    bn = clean_output(rec["response"])
    usage = rec.get("usage") or {}
    config = rec["config"]
    return {"qid": r["qid"], "subject": r["subject"], "split": r["split"],
            "question_en": r["question_en"], "question_bn": bn,
            **checks(r["question_en"], bn, rec["finish_reason"]),
            "finish_reason": rec["finish_reason"], "model": rec["model"], "prompt_id": rec["prompt_id"],
            **{k: config.get(k, "") for k in SETTINGS},
            "thinking": config.get("thinking_config", {}).get("thinking_level", ""),
            "input_tokens": usage.get("prompt_token_count", 0),
            "output_tokens": usage.get("candidates_token_count", 0),
            "thinking_tokens": usage.get("thoughts_token_count", 0),
            "latency_s": rec["latency_s"], "attempts": rec["attempt"],
            "run_id": rec["run_id"], "translated_at": rec["time"]}


def cancel(futures, model):
    model.stop()  # running calls make no further retries
    for fut in futures:
        fut.cancel()  # no effect on calls already running


if __name__ == "__main__":
    main()
