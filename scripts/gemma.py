"""Gemma 4 on the Gemini API: rate-limited, retried, and logged call by call.

Every attempt, failed or not, is appended to a JSONL call log with the full prompt, the
raw response, finish reason, token usage and latency, so any output can be traced back
to the exact request that produced it.

Requests are paced a safety margin below the project's limits (requests per minute,
tokens per minute, requests per day). The docs do not publish Gemma limits; AI Studio
showed 30 RPM, 16K TPM and 14.4K RPD for each Gemma 4 model on 2026-10-30. Limits are per
project and per model, and the daily quota resets at midnight Pacific, so the daily count
covers that model's calls in every calls.jsonl under logs/, not just the current run.
"""
import json
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from google import genai
from google.genai import errors, types

DEFAULT_MODEL = "gemma-4-31b-it"
PACIFIC = ZoneInfo("America/Los_Angeles")
RETRY_CODES = {429, 500, 502, 503, 504}


class DailyLimitReached(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def pacific_day(t):
    return t.astimezone(PACIFIC).date()


def calls_today(log_root, model=None):
    """API attempts made since midnight Pacific, counted across every calls.jsonl under log_root.

    Gemma limits are per model, so pass `model` to count only that model's calls.
    """
    today = pacific_day(utc_now())
    n = 0
    for path in Path(log_root).glob("*/calls.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                n += (pacific_day(datetime.fromisoformat(rec["time"])) == today
                      and (model is None or rec["model"] == model))
    return n


def retry_delay(error, attempt):
    """The server's suggested delay when a 429 gives one, else backoff of 10 s, 20 s, ... up to 5 min."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(error))
    return min(float(m.group(1)) + 1, 300) if m else min(10 * 2 ** (attempt - 1), 300)


class RateLimiter:
    """Paces calls so requests and tokens stay `margin` below each limit. Thread-safe.

    Each call reserves its estimated tokens when it starts, so calls still in flight count
    toward the minute's tokens; the estimate is replaced by the real count when it returns.
    """

    def __init__(self, rpm, tpm, rpd, margin, used_today):
        keep = 1 - margin
        self.min_gap = 60 / (rpm * keep)
        self.tpm = tpm * keep
        self.rpd = int(rpd * keep)
        self.day = pacific_day(utc_now())
        self.used_today = used_today
        self.tokens = deque()  # [monotonic start time, tokens] of calls started in the last minute
        self.last = float("-inf")
        self.lock = threading.Lock()

    def wait(self, est_tokens):
        """Block until a call may start; return its reservation for spent()."""
        while True:
            with self.lock:
                if pacific_day(utc_now()) != self.day:
                    self.day, self.used_today = pacific_day(utc_now()), 0
                if self.used_today >= self.rpd:
                    raise DailyLimitReached(f"{self.used_today} requests today; the cap with margin is {self.rpd}")
                now = time.monotonic()
                while self.tokens and self.tokens[0][0] <= now - 60:
                    self.tokens.popleft()
                delay = self.last + self.min_gap - now
                if self.tokens and sum(t for _, t in self.tokens) + est_tokens > self.tpm:
                    delay = max(delay, self.tokens[0][0] + 60 - now)
                if delay <= 0:
                    self.last = now
                    self.used_today += 1
                    reservation = [now, est_tokens]
                    self.tokens.append(reservation)
                    return reservation
            time.sleep(min(delay, 1.0))  # recheck often: a finished call may free reserved tokens

    def spent(self, reservation, tokens):
        with self.lock:
            reservation[1] = tokens


class Gemma:
    def __init__(self, call_log, limiter, model=DEFAULT_MODEL, temperature=0.0, top_p=None, top_k=None,
                 seed=None, max_output_tokens=2048, thinking=None, max_attempts=4):
        self.client = genai.Client()  # reads GEMINI_API_KEY from the environment
        self.call_log = Path(call_log)
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        self.limiter = limiter
        self.model = model
        # Settings left as None are not sent, so the model default applies; model_info() records it.
        self.config = {k: v for k, v in {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                                         "seed": seed, "max_output_tokens": max_output_tokens}.items()
                       if v is not None}
        if thinking:
            self.config["thinking_config"] = {"thinking_level": thinking}
        self.max_attempts = max_attempts
        self.log_lock = threading.Lock()  # generate() may run in several threads at once
        self.stopped = threading.Event()  # set by stop(): running calls make no further attempts

    def model_info(self):
        """Model metadata, including its default temperature, top_p and top_k and token limits."""
        try:
            return self.client.models.get(model=self.model).model_dump(mode="json", exclude_none=True)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def generate(self, prompt, system=None, **log_fields):
        """Return (text, record of the last attempt); text is None if no attempt produced any."""
        config = self.config | ({"system_instruction": system} if system else {})
        est_tokens = (len(prompt) + len(system or "")) // 2  # rough: prompt plus a similar-length output
        for attempt in range(1, self.max_attempts + 1):
            if self.stopped.is_set():
                return None, {**log_fields, "status": "cancelled"}
            reservation = self.limiter.wait(est_tokens)
            record = {**log_fields, "attempt": attempt, "time": utc_now().isoformat(timespec="seconds"),
                      "model": self.model, "config": config, "prompt": prompt}
            start = time.monotonic()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=types.GenerateContentConfig(**config))
            except Exception as e:  # API errors and network failures alike
                code = getattr(e, "code", None)
                record |= {"status": "error", "error_code": code, "error": f"{type(e).__name__}: {e}",
                           "latency_s": round(time.monotonic() - start, 2)}
                self._log(record)
                if (isinstance(e, errors.APIError) and code not in RETRY_CODES) or attempt == self.max_attempts:
                    return None, record
                self.stopped.wait(retry_delay(e, attempt))  # sleeps, but wakes at once on stop()
                continue

            cand = resp.candidates[0] if resp.candidates else None
            usage = resp.usage_metadata
            text = resp.text
            record |= {
                "status": "ok" if text else "empty",
                "finish_reason": getattr(cand.finish_reason, "name", None) if cand else None,
                "response": text,
                "usage": usage.model_dump(mode="json", exclude_none=True) if usage else None,
                "prompt_feedback": (resp.prompt_feedback.model_dump(mode="json", exclude_none=True)
                                    if resp.prompt_feedback else None),
                "model_version": resp.model_version,
                "response_id": resp.response_id,
                "latency_s": round(time.monotonic() - start, 2),
            }
            self._log(record)
            self.limiter.spent(reservation, usage.total_token_count if usage and usage.total_token_count
                               else est_tokens)
            return text, record

    def stop(self):
        self.stopped.set()

    def _log(self, record):
        with self.log_lock, open(self.call_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
