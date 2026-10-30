"""Phase 1B: apply the hard filters to the Stack Exchange exports.

Reads data/raw/<subject>.csv (Data Explorer exports with columns Id, Title, Body,
Score, Tags, CreationDate, AcceptedAnswerId, Url) and writes to data/interim/:
  questions_en.csv    one row per question that passes, keyed by qid
  filter_counts.csv   questions left after each filter, per split (paper table)

Also strips greeting and sign-off lines and drops opinion/resource tags (rule-based part
of 1C). The exports have no ClosedDate, so the plan's "not closed" filter is not applied.

Usage: python scripts/phase1_filter.py
"""
import argparse
import csv
import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = {"physics": "phy", "chemistry": "chem", "biology": "bio", "astronomy": "astro",
            "earthscience": "earth"}  # subject (= Stack Exchange site) -> qid prefix
SPLITS = ["main", "gap", "control"]
MAIN_BEFORE = "2022-01-01"   # main set: written before ChatGPT
CONTROL_FROM = "2025-01-01"  # control set: new questions; 2022-2024 is the unused gap
MIN_SCORE = 5
MIN_WORDS, MAX_WORDS = 15, 300
MAX_MATH_BLOCKS = 2
# Tags whose questions ask for opinions, resources, or identifying something from a photo.
EXCLUDED_TAGS = {"resource-recommendations", "soft-question", "specific-reference", "reference-request",
                 "book-recommendation", "books", "literature", "education", "software",
                 "species-identification", "identification", "textbook-erratum"}
# Whole lines that are only a greeting or a sign-off; removed before counting words.
PLEASANTRY = re.compile(
    r"(hi|hello|hey|greetings|dear)( (all|everyone|everybody|guys|folks|there))?"
    r"|(many )?(thanks|thank you|thx|cheers|regards)( (so much|very much|in advance|a lot))?"
    r"( for (your|any|the) (help|answers?|time))?"
    r"|any (help|suggestions?|ideas?|insights?|thoughts?|input)( (would be|is|are))?"
    r"( (greatly|much|really|very much))? appreciated", re.I)

MATH_SPAN = re.compile(r'<span class="math-container">(.*?)</span>', re.S)
# Older posts have bare TeX with no math-container span around it.
BARE_MATH = re.compile(r"\$\$.+?\$\$|(?<!\\)\$[^$]+?(?<!\\)\$|\\\[.+?\\\]|\\\(.+?\\\)", re.S)
IMAGE_LINK = re.compile(r"i\.stack\.imgur\.com|i\.sstatic\.net")

SUB = dict(zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"))
SUP = dict(zip("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ"))
BLOCK_TAGS = {"p", "div", "blockquote", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "hr"}

QUESTION_FIELDS = ["qid", "subject", "split", "se_id", "url", "created", "year", "score",
                   "accepted_answer_id", "tags", "n_words", "n_math", "title", "question_en"]


class _TextExtractor(HTMLParser):
    """HTML body -> plain text. Keeps TeX verbatim; turns H<sub>2</sub>O into H₂O."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.buffers = [[]]  # <sub>, <sup> and <strike> capture into their own buffer
        self.captures = []

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS or tag == "br":
            self.buffers[-1].append("\n")
        elif tag == "li":
            self.buffers[-1].append("\n- ")
        elif tag in ("td", "th"):
            self.buffers[-1].append(" ")
        elif tag in ("sub", "sup", "s", "strike", "del"):
            self.captures.append(tag)
            self.buffers.append([])

    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS:
            self.buffers[-1].append("\n")
        elif self.captures and tag == self.captures[-1]:
            self.captures.pop()
            text = "".join(self.buffers.pop()).strip()
            if tag in ("sub", "sup"):  # struck-out text is dropped
                table, mark = (SUB, "_") if tag == "sub" else (SUP, "^")
                if text and all(c in table for c in text):
                    text = "".join(table[c] for c in text)
                else:
                    text = mark + (text if len(text) == 1 else "{" + text + "}")
                self.buffers[-1].append(text)

    def handle_data(self, data):
        self.buffers[-1].append(re.sub(r"\s+", " ", data))  # HTML whitespace is a space

    def text(self):
        while len(self.buffers) > 1:  # unclosed capture tag
            self.buffers[-2].extend(self.buffers.pop())
        s = "".join(self.buffers[0])
        s = "\n".join(line.strip() for line in s.split("\n"))
        return re.sub(r"\n{3,}", "\n\n", s).strip()


def html_to_text(body):
    parser = _TextExtractor()
    parser.feed(body)
    parser.close()
    return parser.text()


def tidy(text):
    text = text.replace("​", "").replace("﻿", "")
    lines = [ln for ln in text.split("\n") if not PLEASANTRY.fullmatch(ln.strip(" \t.,!:;)-"))]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def count_math(body):
    spans = MATH_SPAN.findall(body)
    return len(spans) + len(BARE_MATH.findall(MATH_SPAN.sub(" ", body)))


def parse_tags(tags):
    # Dump years differ: "<physics><optics>" vs "|physics|optics|".
    if tags.startswith("<"):
        return re.findall(r"<([^>]+)>", tags)
    return [t for t in tags.split("|") if t]


def split_of(created):
    if created < MAIN_BEFORE:
        return "main"
    return "control" if created >= CONTROL_FROM else "gap"


def to_record(row, subject):
    title = tidy(html.unescape(row["Title"]))
    body_en = tidy(html_to_text(row["Body"]))
    question_en = f"{title}\n\n{body_en}" if body_en else title
    return {
        "qid": f"{SUBJECTS[subject]}_{row['Id']}",
        "subject": subject,
        "split": split_of(row["CreationDate"]),
        "se_id": int(row["Id"]),
        "url": row["Url"],
        "created": row["CreationDate"],
        "year": int(row["CreationDate"][:4]),
        "score": int(row["Score"]),
        "accepted_answer_id": row["AcceptedAnswerId"].strip(),
        "tags": ";".join(parse_tags(row["Tags"])),
        "n_words": len(question_en.split()),
        "n_math": count_math(row["Body"]),
        "title": title,
        "question_en": question_en,
        "body_html": row["Body"],
    }


FILTERS = [  # applied in this order; the order fixes the counts in the paper table
    (f"score >= {MIN_SCORE}", lambda r: r["score"] >= MIN_SCORE),
    ("has accepted answer", lambda r: r["accepted_answer_id"] != ""),
    ("no images", lambda r: "<img" not in r["body_html"] and not IMAGE_LINK.search(r["body_html"])),
    ("no code blocks", lambda r: "<pre" not in r["body_html"]),
    (f"<= {MAX_MATH_BLOCKS} math blocks", lambda r: r["n_math"] <= MAX_MATH_BLOCKS),
    (f"{MIN_WORDS}-{MAX_WORDS} words", lambda r: MIN_WORDS <= r["n_words"] <= MAX_WORDS),
    ("no opinion/resource tags", lambda r: not EXCLUDED_TAGS & set(r["tags"].split(";"))),
]


def by_split(records):
    return [sum(r["split"] == s for r in records) for s in SPLITS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "interim")
    args = ap.parse_args()
    csv.field_size_limit(2**31 - 1)
    args.out.mkdir(parents=True, exist_ok=True)

    questions, counts = [], []
    for subject in SUBJECTS:
        path = args.raw / f"{subject}.csv"
        if not path.exists():
            print(f"WARNING: {path.name} not found, skipping {subject}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8", newline="") as f:
            records = [to_record(row, subject) for row in csv.DictReader(f)]
        counts.append([subject, "raw export", *by_split(records)])
        for name, keep in FILTERS:
            records = [r for r in records if keep(r)]
            counts.append([subject, name, *by_split(records)])
        questions.extend(records)

    questions.sort(key=lambda r: (list(SUBJECTS).index(r["subject"]), r["se_id"]))
    # utf-8-sig so Excel shows Bengali text correctly once translations are added.
    with open(args.out / "questions_en.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUESTION_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(questions)
    with open(args.out / "filter_counts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "step", *SPLITS])
        w.writerows(counts)

    print(f"{'subject':<13} {'step':<25} {'main':>6} {'gap':>6} {'control':>7}")
    for subject, step, *n in counts:
        print(f"{subject:<13} {step:<25} {n[0]:>6} {n[1]:>6} {n[2]:>7}")
    print(f"\nwrote {len(questions)} questions to {args.out / 'questions_en.csv'}")


if __name__ == "__main__":
    main()
