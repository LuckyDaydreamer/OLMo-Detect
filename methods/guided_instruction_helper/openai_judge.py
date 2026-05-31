"""
GPT-4 in-context-learning judge for Guided Instruction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

ICL_PROMPT_TEMPLATE = """INSTRUCTION:
You are provided with a reference text and a candidate text.
The candidate text is a generated replica of the reference text.
Your task is to determine if the candidate text is an exact or near-exact match of the reference text.
Near-exact match candidates must PRECISELY match the reference candidates in terms of sentence structure, overlap, and contextual similarity.
Respond only with 'Yes' or 'No'.
---
Example 1:

REFERENCE TEXT:
The cat waited at the top.

CANDIDATE TEXT:
The cat waited at the top.

ANSWER: Yes (exact match)
---
Example 2:

REFERENCE TEXT:
icy surface of Jupiter's largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede's icy shell for billions of years.

CANDIDATE TEXT:
icy surface of Jupiter's largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede's icy shell for billions of years. This discovery supports the theory that Ganymede has a subsurface ocean. Scientists used gravity data from NASA's Galileo spacecraft to create a geophysical model of the interior of Ganymede.

ANSWER: Yes (near-exact match)
---
Example 3:

REFERENCE TEXT:
50th Anniversary of Normandy Landings lasts a year.

CANDIDATE TEXT:
The 50th anniversary celebration of the first Normandy landing will last a year.

ANSWER: Yes (near-exact match)
---
Example 4:

REFERENCE TEXT:
Microsoft's Hotmail has raised its storage capacity to 250MB.

CANDIDATE TEXT:
Microsoft has increased the storage capacity of its Hotmail e-mail service to 250MB.

ANSWER: Yes (near-exact match)
---
Example 5:

REFERENCE TEXT:
{reference}

CANDIDATE TEXT:
{candidate}

ANSWER:
"""

DEFAULT_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"   # was "gpt-4-0613" (deprecated 2024)
DEFAULT_CACHE_DIR   = ".openai_judge_cache"
DEFAULT_WORKERS     = 8
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_S   = 60


def _looks_like_yes(token_text: str) -> bool:
    """
    Tokens may have leading whitespace and trailing punctuation/space.
    We require EXACT match after normalisation so that semantically unrelated
    tokens that *start with* the same letters (e.g. "yesterday", "yes,
    indeed") don't get conflated with a literal "yes".
    """
    s = token_text.strip().lower().rstrip(",.;:!? \t")
    return s == "yes"


def _looks_like_no(token_text: str) -> bool:
    """Same exact-match policy as _looks_like_yes — guards against "not", "now",
    "none", "nothing" etc. all leaking into P(no)."""
    s = token_text.strip().lower().rstrip(",.;:!? \t")
    return s == "no"


def _hash_key(model: str, reference: str, candidate: str) -> str:
    """sha256 of (model || reference || candidate). Stable across runs."""
    h = hashlib.sha256()
    sep = b"\n---\n"
    h.update(model.encode("utf-8"))
    h.update(sep)
    h.update(reference.encode("utf-8"))
    h.update(sep)
    h.update(candidate.encode("utf-8"))
    return h.hexdigest()


class OpenAIJudge:
    """
    Thread-safe GPT-4 ICL judge with disk cache + exponential backoff.

    Usage:
        judge = OpenAIJudge(model="gpt-4o-mini-2024-07-18", cache_dir=".cache/")
        result = judge.judge_one(reference, candidate)
        # result = {"p_yes": 0.93, "p_no": 0.04, "raw_text": "Yes (exact match)"}
    """

    def __init__(
        self,
        model:       str  = DEFAULT_JUDGE_MODEL,
        cache_dir:   str | None = DEFAULT_CACHE_DIR,
        num_workers: int  = DEFAULT_WORKERS,
        max_retries: int  = DEFAULT_MAX_RETRIES,
        timeout_s:   int  = DEFAULT_TIMEOUT_S,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIJudge requires the `openai` package. "
                "Install with `pip install openai>=1.0`."
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it before running:\n"
                "    export OPENAI_API_KEY=...\n"
            )

        self._client      = OpenAI(timeout=timeout_s)
        self.model        = model
        self.num_workers  = max(1, int(num_workers))
        self.max_retries  = max(1, int(max_retries))
        self.cache_dir    = Path(cache_dir) if cache_dir else None
        self._fs_lock     = threading.Lock()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── disk cache ──────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # 2-level subsharding for filesystem performance
        return self.cache_dir / key[:2] / f"{key}.json"

    def _load_cached(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "p_yes" in obj:
                return obj
            return None
        except Exception:
            return None  # Corrupt cache file → treat as miss

    def _save_cached(self, key: str, value: dict[str, Any]) -> None:
        p = self._cache_path(key)
        if p is None:
            return
        try:
            with self._fs_lock:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False)
                tmp.replace(p)
        except Exception:
            pass  # Cache write failure is non-fatal.

    # ── single API call ─────────────────────────────────────────────────────

    def _api_call_once(self, prompt: str) -> dict[str, Any]:
        """One non-retried call. Raises on any error."""
        resp = self._client.chat.completions.create(
            model       = self.model,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 10,
            logprobs    = True,
            top_logprobs= 20,
        )
        if not resp.choices:
            raise RuntimeError("OpenAI response had no choices.")
        choice  = resp.choices[0]
        content = choice.message.content if choice.message else ""
        content = (content or "").strip()

        p_yes = 0.0
        p_no  = 0.0
        used_logprobs = False

        # Path 1: read first-token logprobs (preferred — gives a continuous score)
        try:
            lp = getattr(choice, "logprobs", None)
            if lp is not None and getattr(lp, "content", None):
                first = lp.content[0]
                top = getattr(first, "top_logprobs", None) or []
                for entry in top:
                    tok = getattr(entry, "token", "") or ""
                    lpv = getattr(entry, "logprob", None)
                    if lpv is None:
                        continue
                    if _looks_like_yes(tok):
                        p_yes += math.exp(float(lpv))
                        used_logprobs = True
                    elif _looks_like_no(tok):
                        p_no += math.exp(float(lpv))
                        used_logprobs = True
        except Exception:
            used_logprobs = False

        # Path 2: fall back to text parsing if logprobs unavailable / empty
        if not used_logprobs:
            low = content.lower().lstrip(" \t.,;:'\"")
            if low.startswith("yes"):
                p_yes, p_no = 1.0, 0.0
            elif low.startswith("no"):
                p_yes, p_no = 0.0, 1.0
            else:
                p_yes, p_no = 0.5, 0.5

        return {
            "p_yes":         float(p_yes),
            "p_no":          float(p_no),
            "raw_text":      content,
            "used_logprobs": bool(used_logprobs),
        }

    def _api_call_retry(self, prompt: str) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._api_call_once(prompt)
            except Exception as e:  # broad — includes RateLimitError, APIError, etc.
                last_err = e
                # Exponential backoff with cap
                wait = min(60.0, 2.0 ** attempt) + 0.1 * attempt
                time.sleep(wait)
        # Surface the final error to stderr so failures aren't silent.
        # (Without this, judge_pairs catches the RuntimeError below and stores
        # NaN per record without ever revealing the underlying error message,
        # which made all-NaN runs look like inexplicable model-quality issues.)
        import sys
        print(f"[openai_judge] all {self.max_retries} retries failed: "
              f"{type(last_err).__name__}: {last_err}", file=sys.stderr, flush=True)
        raise RuntimeError(
            f"OpenAI API failed after {self.max_retries} retries: {last_err!r}"
        )

    # ── public API ──────────────────────────────────────────────────────────

    def judge_one(self, reference: str, candidate: str) -> dict[str, Any]:
        """Get the judge's P(Yes) for one (reference, candidate) pair. Cached."""
        # Empty candidate / reference → unjudgeable; return a deterministic mid-score.
        if not reference or not candidate:
            return {"p_yes": 0.0, "p_no": 0.0, "raw_text": "",
                    "used_logprobs": False, "skipped": True}

        key = _hash_key(self.model, reference, candidate)
        cached = self._load_cached(key)
        if cached is not None:
            cached.setdefault("from_cache", True)
            return cached

        prompt = ICL_PROMPT_TEMPLATE.format(reference=reference, candidate=candidate)
        result = self._api_call_retry(prompt)
        self._save_cached(key, result)
        result["from_cache"] = False
        return result

    def judge_pairs(
        self,
        pairs: Iterable[tuple[Any, str, str]],
        on_progress: Callable[[], None] | None = None,
    ) -> dict[Any, dict[str, Any]]:
        """
        Judge many pairs in parallel.

        pairs: iterable of (id, reference, candidate). The id is opaque to the
               judge — used as the dict key in the returned mapping.
        on_progress: optional callable invoked once per completed pair.
        """
        pairs_list = list(pairs)
        results: dict[Any, dict[str, Any]] = {}

        if self.num_workers <= 1 or len(pairs_list) <= 1:
            for item_id, ref, cand in pairs_list:
                results[item_id] = self.judge_one(ref, cand)
                if on_progress is not None:
                    on_progress()
            return results

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            future_to_id = {
                ex.submit(self.judge_one, ref, cand): item_id
                for item_id, ref, cand in pairs_list
            }
            for fut in as_completed(future_to_id):
                item_id = future_to_id[fut]
                try:
                    results[item_id] = fut.result()
                except Exception as e:
                    # Record the failure but don't abort the whole batch.
                    # Print the first failure to stderr so silent N/A AUROCs
                    # (e.g. from auth errors, model deprecation, rate-limit
                    # exhaustion) don't go unnoticed. Subsequent identical
                    # failures are suppressed to avoid spamming hundreds of
                    # identical lines per dataset.
                    import sys
                    if not getattr(self, "_logged_judge_error", False):
                        print(f"[openai_judge] judge_one raised: "
                              f"{type(e).__name__}: {e}",
                              file=sys.stderr, flush=True)
                        self._logged_judge_error = True
                    results[item_id] = {
                        "p_yes": float("nan"), "p_no": float("nan"),
                        "raw_text": "", "used_logprobs": False,
                        "error": repr(e),
                    }
                if on_progress is not None:
                    on_progress()
        return results