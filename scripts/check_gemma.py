"""Check which Gemma models and settings work right now, one call per case, no retries.

Default: lists the Gemma models your key can see, then tries "hello" and the real
translation prompt (first pilot question) with the settings translate.py uses.

--bisect: when the translation prompt fails but "hello" works, sends variants that each
change one thing (prompt length, rules, question text, settings, system instruction,
model) to find what triggers the error.

Each call is logged to logs/diagnostics/calls.jsonl and counts toward the daily quota tracking.

Usage: python scripts/check_gemma.py [--bisect]
"""
import argparse
import os
import sys

from dotenv import load_dotenv

import gemma
import translate

HELLO = "Reply with one word: hello"
M31, M26 = "gemma-4-31b-it", "gemma-4-26b-a4b-it"
# Same settings translate.py uses. Gemma 4 thinks by default, and with thinking on a test
# translation used 2,046 of 2,048 output tokens on thoughts and returned nothing.
TRANSLATE_SETTINGS = {"temperature": 0.0, "max_output_tokens": 2048, "thinking": "minimal"}
ONLY_NO_THINKING = {"temperature": None, "max_output_tokens": None, "thinking": "minimal"}
THINKING_ON = {"temperature": 0.0, "max_output_tokens": 8192, "thinking": None}


def cases(question, bisect):
    real = translate.build_prompt(question["question_en"])
    qid = question["qid"]
    if not bisect:
        return [
            ("31b hello", M31, TRANSLATE_SETTINGS, HELLO, None),
            (f"31b translate {qid}", M31, TRANSLATE_SETTINGS, real, None),
            (f"26b-a4b translate {qid}", M26, TRANSLATE_SETTINGS, real, None),
        ]
    title = question["question_en"].split("\n")[0]
    rules = translate.PROMPT_TEMPLATE.split("Question:")[0].strip()
    return [
        ("1 short: translate title only", M31, TRANSLATE_SETTINGS, f"Translate to Bengali: {title}", None),
        ("2 our rules + short question", M31, TRANSLATE_SETTINGS,
         translate.build_prompt("Why does ice float on water?"), None),
        ("3 no rules + full question", M31, TRANSLATE_SETTINGS,
         f"Translate to Bengali:\n\n{question['question_en']}", None),
        ("4 full prompt, default temp/max", M31, ONLY_NO_THINKING, real, None),
        ("5 rules as system instruction", M31, TRANSLATE_SETTINGS, question["question_en"], rules),
        ("6 26b full prompt, thinking on", M26, THINKING_ON, real, None),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bisect", action="store_true", help="find which part of the prompt triggers errors")
    args = ap.parse_args()
    load_dotenv(translate.ROOT / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set; add it to .env")
    limiter = gemma.RateLimiter(float(os.environ.get("GEMMA_RPM", 30)), float(os.environ.get("GEMMA_TPM", 16000)),
                                int(os.environ.get("GEMMA_RPD", 14400)), float(os.environ.get("GEMMA_MARGIN", 0.2)),
                                gemma.calls_today(translate.LOG_ROOT))
    log = translate.LOG_ROOT / "diagnostics" / "calls.jsonl"

    client = gemma.Gemma(log, limiter).client
    print("Gemma models for this key:", [m.name for m in client.models.list() if "gemma" in m.name])

    rows = [r for r in translate.read_csv(translate.ROOT / "data" / "interim" / "questions_en.csv")
            if r["split"] == "main"]
    first = translate.sample_by_subject(rows, 300, 42)[0]
    for name, model_id, settings, prompt, system in cases(first, args.bisect):
        model = gemma.Gemma(log, limiter, model=model_id, max_attempts=1, **settings)
        text, rec = model.generate(prompt, system=system, task="diagnostic", case=name)
        usage = rec.get("usage") or {}
        detail = (text or rec.get("error") or "").replace("\n", " ")[:90]
        print(f"{name:<34} {rec['status']:<6} finish={rec.get('finish_reason')} "
              f"{rec.get('latency_s')}s in={usage.get('prompt_token_count')} "
              f"out={usage.get('candidates_token_count')} think={usage.get('thoughts_token_count', 0)} | {detail}")


if __name__ == "__main__":
    main()
