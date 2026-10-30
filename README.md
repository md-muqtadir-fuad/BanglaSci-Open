# BanglaSci-Open

An open-ended Bengali science benchmark for evaluating large language models. Real science
questions from Stack Exchange are translated into Bengali. Models answer in free text, and a
judge scores each answer against reference key points.

**Status: work in progress.** Data collection, cleaning and translation (phases 1 and 2) are
implemented. Labeling, reference key points, evaluation and judge validation come next. No data
has been released yet.

## Pipeline

| Phase | Step | Script | Output in `data/interim/` |
|---|---|---|---|
| 1 | Filter Stack Exchange questions | `scripts/phase1_filter.py` | `questions_en.csv`, `filter_counts.csv` |
| 2 | Translate to Bengali with Gemma 4 | `scripts/translate.py` | `translations_bn.csv`, plus logs in `logs/translation/` |
| – | Check that the Gemma API responds | `scripts/check_gemma.py` | logs in `logs/diagnostics/` |

`scripts/gemma.py` is the shared Gemma client: rate limiting, retries and call logging.

## Setup

Python 3.11.

```
py -3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

Then add your Google AI Studio key to `.env` as `GEMINI_API_KEY`.

## Data

Data is not part of this repository. Put one CSV per Stack Exchange site in `data/raw/`
(`physics.csv`, `chemistry.csv`, `biology.csv`, `earthscience.csv`, and `astronomy.csv` if
used), exported from the [Stack Exchange Data Explorer](https://data.stackexchange.com/).
Columns: `Id, Title, Body, Score, Tags, CreationDate, AcceptedAnswerId, Url`. The exports hold
questions with a score of at least 5 and an accepted answer.

## Running

```
python scripts/phase1_filter.py
python scripts/check_gemma.py
python scripts/translate.py --dry-run --sample 300 --seed 42 --split main     # print a prompt, no API call
python scripts/translate.py --sample 300 --seed 42 --split main --limit 8     # smoke test
python scripts/translate.py --sample 300 --seed 42 --split main               # 300-question pilot
```

`python scripts/translate.py --help` lists every option: selecting by subject, split or qid,
sampling settings, rate limits and the number of parallel workers.

### Phase 1: filtering

A Stack Exchange question is kept if it:
- has a score of at least 5 and an accepted answer
- has no images and no code blocks
- has at most 2 math blocks
- is 15 to 300 words long, counting title and body
- is not tagged as a resource or opinion request (`resource-recommendations`, `soft-question`,
  `reference-request` and similar)

Greeting and sign-off lines are removed. Each question gets a `qid` such as `phy_12345` and a
split based on its creation date: `main` (before 2022), `gap` (2022–2024) or `control` (2025 and
later). `filter_counts.csv` records how many questions each filter removed.

### Phase 2: translation

- **Model:** `gemma-4-31b-it` through the Gemini API (`google-genai`), at temperature 0 with
  thinking off and at most 2,048 output tokens.
- **Prompt:** standard Bangladeshi Bengali with standard scientific terms. A term with no Bengali
  translation is transliterated into Bengali script (quark → কোয়ার্ক). Numbers, units, formulas,
  LaTeX and URLs stay unchanged, and the model must not answer the question or add information.
  Each row records the prompt version and hash (`prompt_id`). An output file never mixes prompt
  versions.
- **Automatic checks** flag a row for human review when:
  - a number is missing
  - LaTeX has changed
  - the Bengali-to-English length ratio is outside 0.8–1.6
  - fewer than half the letters are Bengali
  - the output was cut off
- **Rate limits:** requests run in parallel (`--workers`, default 16) under one shared rate
  limiter. It stays 20% below each per-model limit: requests per minute, tokens per minute and
  requests per day, all set in `.env`.
- **Resuming:** a rerun skips finished questions. Translations that were logged but never saved,
  for example after Ctrl+C, are recovered from the log without a new API call.

### Logs

Every API attempt is logged to `logs/<task>/calls.jsonl` with:
- the full prompt and the raw response
- the generation settings and the finish reason
- token counts and latency
- errors and retries

`logs/translation/runs.jsonl` has a start and an end line for each run. Together they record the
arguments, the model's default settings, the prompt template, the input file hash and the totals.
Logs are not committed.

## License

The code is under the [MIT License](LICENSE).

Stack Exchange content is licensed CC BY-SA by its authors. The released Bengali dataset will be
CC BY-SA 4.0, with a source link for every item.
