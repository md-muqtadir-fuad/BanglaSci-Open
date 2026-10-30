"""Clean the Bengali Quora Space scrapes into question and answer tables.

Reads data/raw/bigyane-bangali.csv and data/raw/mystery-of-chemistry.csv (browser-scraper
exports whose columns are Quora CSS class names) and writes to data/interim/:
  quora_questions_bn.csv   one row per unique question, keyed by qid
  quora_answers_bn.csv     one row per answer, joined to its question on qid
  quora_counts.csv         rows kept or dropped at each step

The scrapes hold answer previews only (about 300 characters, cut with "…"), so every
answer row records whether it was truncated.

Usage: python scripts/clean_quora.py
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUESTED = "উত্তরের অনুরোধ"  # "requested an answer": marks question-only rows
MORE = "(আরও)"                # "(more)": the preview was cut
NO_ANSWER = "এখনো কোনও উত্তর নেই"  # "no answer yet"
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
MIN_WORDS = 3  # drops post headings such as "মেটালয়েড্স :" and shifted rows holding a date
BENGALI = re.compile(r"[ঀ-৿]")  # questions with no Bengali letter are English, dropped

QUESTION_FIELDS = ["qid", "source", "subject", "url", "question_bn", "n_words",
                   "quora_answer_count", "has_answer"]
ANSWER_FIELDS = ["aid", "qid", "source", "author", "url", "date_text", "answer_bn",
                 "truncated", "n_words"]


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def norm(text):
    """Key for spotting the same question twice: ignore spacing, case, end punctuation."""
    return re.sub(r"[\s?？:;।.]+$", "", clean(text)).lower()


def make_id(prefix, key):
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}"


def answer_count(label):
    label = clean(label).translate(BN_DIGITS)
    if label == NO_ANSWER:
        return 0
    m = re.match(r"(\d+)টি উত্তর", label)
    return int(m.group(1)) if m else ""


def clean_answer(text):
    text = clean(text)
    text = re.sub(r"Attn\b.*$", "", text).strip()  # @-mention lists run to the end of a preview
    truncated = text.endswith("…")
    return text.rstrip("…").strip(), truncated


def question(source, subject, text, url, count=""):
    text = clean(text)
    return {"source": source, "subject": subject, "url": url, "question_bn": text,
            "n_words": len(text.split()), "quora_answer_count": count}


def parse_bigyane(path, counts):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts.append(["bigyane-bangali", "raw rows", len(rows)])
    questions, answers = [], []
    for r in rows:
        if REQUESTED in r["c1h7helg"]:
            # Question-only row; the scraper also caught the next feed question (no URL) in q-box 4.
            questions.append(question("bigyane-bangali", "unlabeled", r["q-box"], r["q-box href 4"],
                                      answer_count(r["q-box 2"])))
            if clean(r["q-box 4"]) not in ("", "·"):
                questions.append(question("bigyane-bangali", "unlabeled", r["q-box 4"], ""))
            continue
        q = question("bigyane-bangali", "unlabeled", r["q-box 2"], r["q-box href 3"])
        questions.append(q)
        text, truncated = clean_answer(r["q-box 3"])
        answers.append({"question": q, "source": "bigyane-bangali", "author": clean(r["q-inlineFlex"]),
                        "url": r["q-box href 3"], "date_text": clean(r["q-box"]), "answer_bn": text,
                        "truncated": truncated or MORE in (r["q-text 4"], r["q-text 5"])})
    return questions, answers


def parse_mystery(path, counts):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts.append(["mystery-of-chemistry", "raw rows", len(rows)])
    return [question("mystery-of-chemistry", "chemistry", r["q-box"], r["q-box href 3"],
                     answer_count(r["q-box 2"])) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "interim")
    args = ap.parse_args()
    csv.field_size_limit(2**31 - 1)
    args.out.mkdir(parents=True, exist_ok=True)

    counts = []
    questions, answers = parse_bigyane(args.raw / "bigyane-bangali.csv", counts)
    questions += parse_mystery(args.raw / "mystery-of-chemistry.csv", counts)
    counts.append(["all", "questions parsed", len(questions)])

    questions = [q for q in questions if q["n_words"] >= MIN_WORDS]
    counts.append(["all", f">= {MIN_WORDS} words", len(questions)])
    questions = [q for q in questions if BENGALI.search(q["question_bn"])]
    counts.append(["all", "has Bengali text", len(questions)])
    unique = {}
    for q in questions:  # first occurrence wins; it is the one with an answer or a URL
        key = norm(q["question_bn"])
        if key not in unique:
            unique[key] = q | {"qid": make_id("qbn", key)}
        elif not unique[key]["url"] and q["url"]:
            unique[key]["url"] = q["url"]
    questions = list(unique.values())
    counts.append(["all", "unique questions", len(questions)])

    counts.append(["bigyane-bangali", "answer previews", len(answers)])
    answers = [a for a in answers if a["answer_bn"]]
    counts.append(["bigyane-bangali", "non-empty (not just Attn)", len(answers)])
    answers = [a for a in answers if norm(a["question"]["question_bn"]) in unique]
    counts.append(["bigyane-bangali", "question kept", len(answers)])
    seen = set()
    rows = []
    for a in answers:
        if a["url"] in seen:
            continue
        seen.add(a["url"])
        q = unique[norm(a["question"]["question_bn"])]
        q["has_answer"] = True
        rows.append(a | {"aid": make_id("abn", a["url"]), "qid": q["qid"],
                         "n_words": len(a["answer_bn"].split())})
    answers = rows
    counts.append(["bigyane-bangali", "unique answers", len(answers)])
    for q in questions:
        q.setdefault("has_answer", False)

    # utf-8-sig so Excel shows the Bengali text correctly.
    for name, fields, data in (("quora_questions_bn.csv", QUESTION_FIELDS, questions),
                               ("quora_answers_bn.csv", ANSWER_FIELDS, answers)):
        with open(args.out / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    with open(args.out / "quora_counts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "step", "rows"])
        w.writerows(counts)

    for source, step, n in counts:
        print(f"{source:<21} {step:<26} {n:>6}")
    print(f"\nanswers truncated: {sum(a['truncated'] for a in answers)} of {len(answers)}")
    print(f"questions by source: " + ", ".join(
        f"{s} {sum(q['source'] == s for q in questions)}" for s in ("bigyane-bangali", "mystery-of-chemistry")))


if __name__ == "__main__":
    main()
