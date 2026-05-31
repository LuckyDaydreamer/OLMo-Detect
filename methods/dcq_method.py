"""
Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models (DCQ)
Paper: https://aclanthology.org/2025.tacl-1.37.pdf
Github Implementation:  https://github.com/shahriargolchin/DCQ

Phase 1 (perturbation generation, GPT-4-driven):
For each instance x, ask GPT-4 to produce 4 word-level perturbations
(synonym swaps, same meaning + sentence structure). One-time cost
per dataset; cached on disk and shared across all OLMo sizes.

Phase 2 — Bias Detector Quiz (BDQ):
Show the 4 perturbations as A/B/C/D and "None of these" as E.
The original is ABSENT. Ask the MODEL UNDER TEST (OLMo here) to
pick a letter. We read its first-token logprobs over {A,B,C,D,E}
instead of parsing a single-letter output, giving a continuous
P(letter|BDQ) per instance.

Phase 3 — Bias Compensator Quiz (BCQ):
For each placement L ∈ {A,B,C,D}, replace the slot-L perturbation
with the original instance and re-prompt. Read OLMo's first-token
logprobs again. P(L|BCQ_L) is the model's recognition score for
that placement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle
from progress_utils import progress


# ─── tuned hyper-parameters (post-tuning; update via dcq_tune.py) ────────────

MATCHED_CONFIG: dict[str, dict[str, Any]] = {
    "OLMo2-1B-Instruct":  {"score_type": "min_pcorrect"},
    "OLMo2-7B-Instruct":  {"score_type": "mean_pcorrect"},
    "OLMo2-13B-Instruct": {"score_type": "min_pcorrect"},
    "OLMo2-32B-Instruct": {"score_type": "mean_pcorrect"},
}

SHIFTED_CONFIG: dict[str, dict[str, Any]] = {
    "OLMo2-1B-Instruct":  {"score_type": "min_pcorrect"},
    "OLMo2-7B-Instruct":  {"score_type": "mean_pcorrect"},
    "OLMo2-13B-Instruct": {"score_type": "min_pcorrect"},
    "OLMo2-32B-Instruct": {"score_type": "mean_pcorrect"},
}

_FALLBACK_CONFIG = {"score_type": "mean_pcorrect"}


def get_optimal_params(model_path: str, dataset_path: str) -> dict[str, Any]:
    model_name = os.path.basename(model_path.rstrip("/"))
    file_name  = os.path.basename(dataset_path).lower()
    config_dict = SHIFTED_CONFIG if "shifted" in file_name else MATCHED_CONFIG
    config = config_dict.get(model_name, _FALLBACK_CONFIG)
    print(f"[dcq] score_type={config['score_type']}")
    return config


# ─── source-hint extraction ───────
_SOURCE_NAME_MAP: dict[str, str] = {
    "dclm":          "DCLM",
    "openwebmath":   "OpenWebMath",
    "pes2o":         "peS2o",
    "starcoder":     "StarCoder",
    "math-mix":      "OLMo2 math midtraining mix",
    "stackexchange": "Stack Exchange",
    "dpo":           "OLMo2 DPO training data",
    "rlvr":          "OLMo2 RLVR training data",
    "sft":           "OLMo2 SFT training data",
}
_DEFAULT_SOURCE_HINT = "the OLMo2 training corpus"


def extract_source_hint(dataset_path: str) -> str:
    parts = re.split(r"[/\\]", dataset_path.lower())
    for part in parts:
        if part in _SOURCE_NAME_MAP:
            return _SOURCE_NAME_MAP[part]
    return _DEFAULT_SOURCE_HINT


# ─── prompts ──────────────────────────

OPTION_GENERATION_PROMPT = """Your task is to create a four-choice quiz by replacing the words in the provided INPUT TEXT with their contextually relevant synonyms.
The meaning and sentence structure of the four options MUST EXACTLY match every detail in the INPUT TEXT.
You MUST NOT include the provided INPUT TEXT as an option.

You MUST make sure that:
(1) You generate DISTINCT options based on the provided INPUT TEXT;
(2) The ONLY difference between options is WORD-LEVEL PERTURBATIONS.
(3) Options are ORDERED;
(4) There is NOT any extra explanation;
(5) You follow the following FORMAT to generate options;
(6) You comply with every specific symbol and letter detail in the given INPUT TEXT; and
(7) All options retain the EXACT LABEL from the INPUT TEXT, if there is one.
---
INPUT TEXT:
{original_instance}
---
FORMAT:

A)
First word-level perturbation goes here

B)
Second word-level perturbation goes here

C)
Third word-level perturbation goes here

D)
Fourth word-level perturbation goes here
---
"""

QUIZ_QUESTION_PROMPT = """You are provided with a five-choice quiz. Your task is to CORRECTLY select the option that EXACTLY corresponds to an instance from the {split_name} split of the {dataset_name} dataset.

When selecting the option, you MUST ensure that you follow the following rules:
(1) You MUST ensure that you ONLY generate a single option letter as your answer.
(2) If you do not know the dataset or the correct answer, you MUST select option "E) None of the provided options."

Hint: While all the following options seem similar in terms of meaning, there is ONLY one option that reflects the EXACT WORDING with respect to the original instance.
---
A)
{option_a}

B)
{option_b}

C)
{option_c}

D)
{option_d}

E)
None of the provided options.
---
ANSWER:
"""


# ─── option-generation parser ─

_OPTION_PARSE_RE = re.compile(r"(?<!\w)([A-D])\)\s*\n")


def parse_generated_options(text: str) -> dict[str, str]:
    chunks = _OPTION_PARSE_RE.split(text)
    out: dict[str, str] = {"a": "", "b": "", "c": "", "d": ""}
    for i in range(1, len(chunks), 2):
        letter = chunks[i].lower()
        content = chunks[i + 1].strip()
        if letter in out:
            out[letter] = content
    return out


# ─── disk-cached OpenAI client for Phase 1 (perturbation generation only) ────

def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    sep = b"\n---\n"
    first = True
    for p in parts:
        if not first:
            h.update(sep)
        first = False
        h.update(p.encode("utf-8"))
    return h.hexdigest()


class _DCQOpenAIClient:
    """
    Minimal OpenAI client used ONLY for Phase 1 (perturbation generation).
    """

    def __init__(
        self,
        model: str       = "gpt-4o-mini-2024-07-18",
        cache_dir: str   = ".dcq_cache",
        max_retries: int = 5,
        timeout_s: int   = 60,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Phase 1 of DCQ requires the `openai` package. "
                "Install with: pip install openai>=1.0"
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Required for DCQ Phase 1 (perturbation "
                "generation). Set it in the environment before running."
            )

        self._client     = OpenAI(timeout=timeout_s)
        self.model       = model
        self.max_retries = max_retries
        self.cache_dir   = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fs_lock    = threading.Lock()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _load_cached(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cached(self, key: str, value: dict[str, Any]) -> None:
        p = self._cache_path(key)
        try:
            with self._fs_lock:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False)
                tmp.replace(p)
        except Exception:
            pass

    def generate_text(
        self,
        prompt: str,
        max_tokens: int   = 4000,
        temperature: float= 1.0,
    ) -> str:
        key = _hash_key("generate_text", self.model, str(max_tokens), str(temperature), prompt)
        cached = self._load_cached(key)
        if cached is not None and "text" in cached:
            return str(cached["text"])

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not resp.choices:
                    raise RuntimeError("Empty choices in OpenAI response.")
                content = resp.choices[0].message.content if resp.choices[0].message else ""
                content = content or ""
                self._save_cached(key, {"text": content})
                return content
            except Exception as e:
                last_err = e
                wait = min(60.0, 2.0 ** attempt) + 0.1 * attempt
                time.sleep(wait)
        raise RuntimeError(f"OpenAI API failed after {self.max_retries} retries: {last_err!r}")


# ─── Phase 1: perturbation generation (GPT-4) ────────────────────────────────

def generate_options_for_dataset(
    dataset: DatasetBundle,
    client:  _DCQOpenAIClient,
    num_workers: int = 8,
) -> dict[int, dict[str, str]]:
    """
    For every record with non-empty text, generate 4 word-level perturbations
    via GPT-4. Returns {record_index: {"a","b","c","d"}}. Dataset-keyed —
    shared across OLMo sizes.
    """
    prepared: list[tuple[int, str]] = []
    for idx, record in enumerate(dataset.records):
        text = extract_text(record)
        if text:
            prepared.append((idx, text))
    if not prepared:
        return {}

    out: dict[int, dict[str, str]] = {}

    def _one(idx_text: tuple[int, str]) -> tuple[int, dict[str, str]]:
        idx, text = idx_text
        prompt = OPTION_GENERATION_PROMPT.format(original_instance=text)
        raw = client.generate_text(prompt, max_tokens=4000, temperature=1.0)
        return idx, parse_generated_options(raw)

    if num_workers <= 1:
        for it in progress(prepared, desc=f"dcq genopts [{dataset.name}]",
                           unit="opt", dynamic_ncols=True, leave=False):
            idx, parsed = _one(it)
            out[idx] = parsed
        return out

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_one, it): it for it in prepared}
        for fut in progress(as_completed(futures), desc=f"dcq genopts [{dataset.name}]",
                            unit="opt", total=len(futures),
                            dynamic_ncols=True, leave=False):
            try:
                idx, parsed = fut.result()
                out[idx] = parsed
            except Exception as e:
                bad_idx = futures[fut][0]
                out[bad_idx] = {"a": "", "b": "", "c": "", "d": "", "_error": repr(e)}
    return out


# ─── Phase 2/3: OLMo first-token letter logprobs ─────────────────────────────

def _build_letter_token_id_map(
    tokenizer,
    letters: tuple[str, ...] = ("A", "B", "C", "D", "E"),
) -> dict[str, list[int]]:
    """
    Map each letter to the set of token IDs that the model could plausibly
    emit FIRST after `ANSWER:\\n` in the quiz prompt.

    Subtlety: depending on the tokenizer, the bare letter "A" might encode
    differently from " A" or "\\nA". When the prompt ends with `ANSWER:\\n`,
    the next token's surface form depends on whether the BPE merges the
    leading newline with the letter, or whether the prompt's trailing
    whitespace makes the next token a bare letter.

    We enumerate several candidate string forms and collect every single-
    token ID encoding any of them. P(letter) is then the SUM of probabilities
    over these candidate token IDs. This is a tight upper bound on the true
    P(letter as next emitted token).
    """
    out: dict[str, list[int]] = {}
    for L in letters:
        candidates = {L, " " + L, "\n" + L, "\t" + L}
        ids: set[int] = set()
        for c in candidates:
            try:
                tok_ids = tokenizer.encode(c, add_special_tokens=False)
            except Exception:
                continue
            if len(tok_ids) == 1:
                ids.add(int(tok_ids[0]))
        out[L] = sorted(ids)
    return out


def _olmo_letter_probs_batch(
    model,
    tokenizer,
    prompts: list[str],
    letters: tuple[str, ...],
    letter_ids: dict[str, list[int]],
    max_tokens: int = 4096,
    batch_size: int = 1,
    progress_desc: str = "dcq quiz",
) -> list[dict[str, float]]:
    """
    For each prompt, return {letter: P(letter)} from the next-token softmax
    distribution. Uses a single forward pass per batch — no generation loop.

    Probabilities are softmax(last-position logits) summed over each letter's
    candidate token IDs. They do NOT renormalize across the 5 letters; that's
    fine for AUC.
    """
    if not prompts:
        return []
    device = next(model.parameters()).device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # left-pad so the prompt's last token is the rightmost

    out: list[dict[str, float]] = []
    try:
        for start in progress(
            range(0, len(prompts), batch_size),
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_tokens),
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attn      = enc["attention_mask"].to(device)

            with torch.no_grad():
                logits = model(
                    input_ids      = input_ids,
                    attention_mask = attn,
                ).logits  # [B, T, V]

            # With LEFT padding the last position of every row is the actual
            # final token of the prompt — its logits predict the next token.
            last_logits = logits[:, -1, :]                               # [B, V]
            probs       = torch.softmax(last_logits.float(), dim=-1)     # [B, V]
            probs_cpu   = probs.detach().cpu()

            for row_idx in range(probs_cpu.shape[0]):
                row = probs_cpu[row_idx]
                d = {L: 0.0 for L in letters}
                for L in letters:
                    ids = letter_ids.get(L, [])
                    if not ids:
                        continue
                    # Sum probability over all token IDs encoding this letter
                    s = float(row[ids].sum().item())
                    d[L] = s
                out.append(d)

            del logits, last_logits, probs, probs_cpu
            torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = original_padding_side
    return out


# ─── BCQ placement permutations ──────────────────────────────────────────────

# When correct goes to slot L, the remaining slots keep their original
# perturbation; the perturbation that was originally in slot L is dropped.
_PLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "A": [("A", "_orig"), ("B", "b"), ("C", "c"), ("D", "d")],
    "B": [("A", "a"), ("B", "_orig"), ("C", "c"), ("D", "d")],
    "C": [("A", "a"), ("B", "b"), ("C", "_orig"), ("D", "d")],
    "D": [("A", "a"), ("B", "b"), ("C", "c"), ("D", "_orig")],
}


def _build_quiz_prompt(
    options: dict[str, str],
    dataset_name: str,
    split_name: str,
) -> str:
    return QUIZ_QUESTION_PROMPT.format(
        split_name  = split_name,
        dataset_name= dataset_name,
        option_a    = options["A"],
        option_b    = options["B"],
        option_c    = options["C"],
        option_d    = options["D"],
    )


@dataclass
class _DCQRecordResult:
    record_index: int
    sample_id: str
    label: Any
    instance: str
    options: dict[str, str]
    bdq_probs: dict[str, float] | None = None
    bcq_probs: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def pcorrect_per_placement(self) -> dict[str, float]:
        return {L: float(self.bcq_probs.get(L, {}).get(L, 0.0)) for L in self.bcq_probs}


def run_dcq_quizzes(
    model_bundle:   ModelBundle,
    dataset:        DatasetBundle,
    options_by_idx: dict[int, dict[str, str]],
    placements:     tuple[str, ...] = ("A", "B", "C", "D"),
    include_bdq:    bool            = True,
    batch_size:     int             = 1,
    max_tokens:     int             = 4096,
    source_hint:    str | None      = None,
) -> list[_DCQRecordResult]:
    """
    Phases 2/3: build all quiz prompts, batch them through OLMo, read
    first-token letter probabilities. Pure-GPU, no API calls.
    """
    if source_hint is None:
        source_hint = extract_source_hint(dataset.path)
    split_name = "training"

    # 1. Filter records that have valid options + non-empty text
    targets: list[tuple[int, dict[str, Any], str, dict[str, str]]] = []
    for idx, record in enumerate(dataset.records):
        if idx not in options_by_idx:
            continue
        opts = options_by_idx[idx]
        if not all(opts.get(k) for k in ("a", "b", "c", "d")):
            continue
        text = extract_text(record)
        if not text:
            continue
        targets.append((idx, record, text, opts))

    if not targets:
        return []

    letters = ("A", "B", "C", "D", "E")
    letter_ids = _build_letter_token_id_map(model_bundle.tokenizer, letters=letters)

    # 2. Build prompts in a deterministic order:
    #    quiz types per record:
    #    - BDQ (if include_bdq)
    #    - BCQ for each placement L in `placements`
    quiz_types: list[str] = []
    if include_bdq:
        quiz_types.append("bdq")
    quiz_types.extend([f"bcq_{L}" for L in placements])

    all_prompts: list[str]   = []
    all_keys:    list[tuple[int, str]] = []  # (record_index, quiz_type)

    for idx, _, text, opts in targets:
        # BDQ: 4 perturbations in original A/B/C/D positions, no original.
        if include_bdq:
            bdq_options = {"A": opts["a"], "B": opts["b"],
                           "C": opts["c"], "D": opts["d"]}
            all_prompts.append(_build_quiz_prompt(bdq_options, source_hint, split_name))
            all_keys.append((idx, "bdq"))

        # BCQ placements
        src_map = {"_orig": text, "a": opts["a"], "b": opts["b"],
                   "c": opts["c"], "d": opts["d"]}
        for L in placements:
            slots = _PLACEMENTS[L]
            bcq_options = {label_: src_map[src_key] for (label_, src_key) in slots}
            all_prompts.append(_build_quiz_prompt(bcq_options, source_hint, split_name))
            all_keys.append((idx, f"bcq_{L}"))

    # 3. Batched forward pass — single OLMo invocation
    probs_list = _olmo_letter_probs_batch(
        model           = model_bundle.model,
        tokenizer       = model_bundle.tokenizer,
        prompts         = all_prompts,
        letters         = letters,
        letter_ids      = letter_ids,
        max_tokens      = max_tokens,
        batch_size      = batch_size,
        progress_desc   = f"dcq quiz [{dataset.name}]",
    )
    assert len(probs_list) == len(all_keys), \
        f"prob/key length mismatch: {len(probs_list)} vs {len(all_keys)}"

    # 4. Re-pack into per-record results
    by_record: dict[int, _DCQRecordResult] = {}
    for (idx, record, text, opts) in targets:
        by_record[idx] = _DCQRecordResult(
            record_index = idx,
            sample_id    = extract_sample_id(record, fallback=idx),
            label        = record.get("label"),
            instance     = text,
            options      = opts,
        )
    for (idx, qtype), probs in zip(all_keys, probs_list):
        rec = by_record[idx]
        if qtype == "bdq":
            rec.bdq_probs = probs
        elif qtype.startswith("bcq_"):
            L = qtype.split("_", 1)[1]
            rec.bcq_probs[L] = probs

    out = [by_record[t[0]] for t in targets]
    return out


# ─── per-record scoring ──────────────────────────────────────────────────────

VALID_SCORE_TYPES = (
    "mean_pcorrect",
    "min_pcorrect",
    "pcorrect_kappa",
    "pcorrect_kappa_np",
)

# Floor for (1 - p_e) in the kappa denominator. Prevents divide-by-zero
# when a letter has overwhelming positional bias (p_e ≈ 1) — clip to keep
# the score finite while preserving ordering.
_KAPPA_DENOM_FLOOR = 1e-3

# Threshold for the "non-preferred" filter. Per the paper (Section 2.2):
# "any option selected fewer than ⌈k/5⌉ times in the BDQ is deemed
# non-preferred." For partition-level mean BDQ probabilities this
# translates to mean P(L | BDQ) < 1/5.
_NON_PREFERRED_THRESHOLD = 1.0 / 5.0


def non_preferred_letters(
    partition_stats: dict[str, float],
    threshold: float = _NON_PREFERRED_THRESHOLD,
) -> list[str]:
    """
    Identify "non-preferred" placement letters per the paper's definition.

    Paper (Section 2.2):
        Non-preferred = chosen less than ⌈k/5⌉ times in BDQ, excluding
        option E (always reserved for "None of the provided options").

    For partition-level BDQ statistics this is equivalent to:
        non-preferred(L) ↔ mean over partition of P(L | BDQ) < 1/5

    Fallback: if no letter qualifies (all four are preferred — rare but
    possible for heavily-biased small models), the paper specifies using
    all of {A, B, C, D} except E. We follow that.
    """
    np_letters = [
        L for L in ("A", "B", "C", "D")
        if float(partition_stats.get(L, 0.0)) < threshold
    ]
    if not np_letters:
        return ["A", "B", "C", "D"]
    return np_letters


def compute_partition_stats(records: list[_DCQRecordResult]) -> dict[str, float]:
    """
    Compute partition-level mean BDQ probability per letter:
        p_e[L] = mean over partition of P(L | BDQ)

    This is the per-sample analog of the paper's `p_e` term in the
    Cohen's-kappa chance correction. Using a partition-level rather than
    per-sample mean makes the kappa score robust to per-sample BDQ noise,
    which was the failure mode of the previous `pcorrect_minus_bdq`
    score type.
    """
    sums:   dict[str, float] = {L: 0.0 for L in ("A", "B", "C", "D")}
    counts: dict[str, int]   = {L: 0   for L in ("A", "B", "C", "D")}
    for rec in records:
        if rec.bdq_probs is None:
            continue
        for L in ("A", "B", "C", "D"):
            p = rec.bdq_probs.get(L)
            if p is None:
                continue
            try:
                pf = float(p)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pf):
                continue
            sums[L]   += pf
            counts[L] += 1
    return {
        L: (sums[L] / counts[L] if counts[L] > 0 else 0.0)
        for L in ("A", "B", "C", "D")
    }


def score_from_record(
    rec: _DCQRecordResult,
    score_type: str,
    partition_stats: dict[str, float] | None = None,
) -> float:
    """
    Convert a record's BCQ/BDQ probabilities into a single per-sample score.
    Higher = more likely contaminated.

    Args:
        rec: The DCQ record (BCQ + optional BDQ probabilities).
        score_type: One of VALID_SCORE_TYPES.
        partition_stats: Required for `pcorrect_kappa`. Output of
            compute_partition_stats() — partition-level mean BDQ per letter.

    Notes on `pcorrect_kappa` vs `pcorrect_kappa_np`:
        Both apply per-sample κ-style chance correction with PARTITION-LEVEL p_e:
            kappa(L)_i = (P(L|BCQ_L)_i − p_e[L]) / (1 − p_e[L])
        and average across placements L. They differ in WHICH placements:
            - `pcorrect_kappa`     : averages over all of {A, B, C, D}.
                                     Smooth weighting via 1/(1−p_e[L]) — biased
                                     placements get small weight automatically.
            - `pcorrect_kappa_np`  : averages over non-preferred placements only
                                     (those with p_e[L] < 1/5), per the paper's
                                     Section 2.2. Hard cutoff version of the
                                     same idea. Most faithful to the paper's
                                     algorithm.

        By contrast, a per-sample subtraction
            s_i = mean_L [P(L|BCQ_L)_i − P(L|BDQ)_i]
        is unstable on small models (per-sample BDQ noise dominates), and a
        partition-level subtraction
            s_i = mean_L [P(L|BCQ_L)_i − p_e[L]]
            = mean_pcorrect_i − constant
        is shift-invariant in AUC space (i.e. equivalent to mean_pcorrect).
        Neither is a useful score type; both are intentionally absent.
    """
    if score_type not in VALID_SCORE_TYPES:
        raise ValueError(f"Unknown score_type={score_type!r}. Choose from {VALID_SCORE_TYPES}.")

    pcorrect = rec.pcorrect_per_placement
    if not pcorrect:
        return float("nan")

    if score_type == "mean_pcorrect":
        return float(sum(pcorrect.values()) / len(pcorrect))

    if score_type == "min_pcorrect":
        return float(min(pcorrect.values()))

    if score_type == "pcorrect_kappa":
        if partition_stats is None:
            raise ValueError(
                "score_type='pcorrect_kappa' requires partition_stats; "
                "call compute_partition_stats(records) and pass the result."
            )
        kappas: list[float] = []
        for L, p in pcorrect.items():
            p_e = float(partition_stats.get(L, 0.0))
            denom = max(_KAPPA_DENOM_FLOOR, 1.0 - p_e)
            kappas.append((float(p) - p_e) / denom)
        return float(sum(kappas) / len(kappas)) if kappas else float("nan")

    if score_type == "pcorrect_kappa_np":
        # (mean P(L | BDQ) < 1/5), then apply kappa-style chance correction.
        if partition_stats is None:
            raise ValueError(
                "score_type='pcorrect_kappa_np' requires partition_stats; "
                "call compute_partition_stats(records) and pass the result."
            )
        np_letters = set(non_preferred_letters(partition_stats))
        kappas_np: list[float] = []
        for L, p in pcorrect.items():
            if L not in np_letters:
                continue
            p_e = float(partition_stats.get(L, 0.0))
            denom = max(_KAPPA_DENOM_FLOOR, 1.0 - p_e)
            kappas_np.append((float(p) - p_e) / denom)
        return float(sum(kappas_np) / len(kappas_np)) if kappas_np else float("nan")

    raise AssertionError("unreachable")


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


# ─── orchestration: cached on model_bundle.runtime_cache ─────────────────────

def _options_runtime_cache_key(dataset: DatasetBundle, judge_model: str) -> str:
    return f"dcq_options::{dataset.path}::judge={judge_model}"


def _quizzes_runtime_cache_key(
    dataset: DatasetBundle,
    placements: tuple[str, ...],
    include_bdq: bool,
    source_hint: str,
    max_tokens: int,
) -> str:
    plc = ",".join(sorted(placements))
    return (f"dcq_quizzes::{dataset.path}::plc={plc}::bdq={include_bdq}"
            f"::src={source_hint}::mt={max_tokens}")


def compute_dcq_records(
    model_bundle: ModelBundle,
    dataset:      DatasetBundle,
    judge_model:  str             = "gpt-4o-mini-2024-07-18",
    cache_dir:    str             = ".dcq_cache",
    num_workers:  int             = 8,
    placements:   tuple[str, ...] = ("A", "B", "C", "D"),
    include_bdq:  bool            = True,
    batch_size:   int             = 1,
    max_tokens:   int             = 4096,
    source_hint:  str | None      = None,
) -> list[_DCQRecordResult]:
    """
    End-to-end: GPT-4 perturbation generation (Phase 1, cached on disk
    across OLMo sizes) + OLMo quiz forward passes (Phases 2/3).
    """
    if source_hint is None:
        source_hint = extract_source_hint(dataset.path)

    quiz_key = _quizzes_runtime_cache_key(
        dataset, placements, include_bdq, source_hint, max_tokens
    )
    cached = model_bundle.runtime_cache.get(quiz_key)
    if cached is not None:
        return cached

    # Phase 1 — GPT-4 perturbation generation
    opt_key = _options_runtime_cache_key(dataset, judge_model)
    options_by_idx = model_bundle.runtime_cache.get(opt_key)
    if options_by_idx is None:
        client = _DCQOpenAIClient(model=judge_model, cache_dir=cache_dir)
        options_by_idx = generate_options_for_dataset(
            dataset, client, num_workers=num_workers,
        )
        model_bundle.runtime_cache[opt_key] = options_by_idx

    # Phases 2/3 — OLMo
    records = run_dcq_quizzes(
        model_bundle    = model_bundle,
        dataset         = dataset,
        options_by_idx  = options_by_idx,
        placements      = placements,
        include_bdq     = include_bdq,
        batch_size      = batch_size,
        max_tokens      = max_tokens,
        source_hint     = source_hint,
    )
    model_bundle.runtime_cache[quiz_key] = records
    return records


# ─── BaseMethod ──────────────────────────────────────────────────────────────

class DCQMethod(BaseMethod):
    """
    DCQ — OLMo as test-taker, GPT-4 as Phase-1 perturbation generator.
    Score convention: higher = more likely contaminated.

    Tuned per (model, matched/shifted) via MATCHED_CONFIG / SHIFTED_CONFIG:
        score_type ∈ {mean_pcorrect, min_pcorrect, pcorrect_kappa, pcorrect_kappa_np}

    BDQ is always run (include_bdq is forced True). It is required for
    `pcorrect_kappa`; for the other two score types it is unused but its
    forward-pass cost is amortised across score-type tuning runs.
    """
    name = "dcq"

    def __init__(
        self,
        max_tokens:   int  = 4096,    # framework wildcard
        batch_size:   int  = 1,       # framework wildcard (used for OLMo forward passes)
        judge_model:  str  = "gpt-4o-mini-2024-07-18",  # Phase 1 only
        cache_dir:    str  = ".dcq_cache",
        num_workers:  int  = 8,       # Phase 1 parallelism
        placements:   str  = "ABCD",
        include_bdq:  bool = True,    # forced True internally; kept for API stability
    ):
        self.max_tokens   = int(max_tokens)
        self.batch_size   = int(batch_size)
        self.judge_model  = str(judge_model)
        self.cache_dir    = str(cache_dir)
        self.num_workers  = int(num_workers)
        self.placements   = tuple(L.upper() for L in placements if L.upper() in "ABCD")
        if not self.placements:
            self.placements = ("A", "B", "C", "D")
        # BDQ is cheap (one forward pass per record) and required for
        # pcorrect_kappa, so we always run it. Ignoring the constructor
        # value here keeps the public API stable for legacy callers.
        self.include_bdq  = True

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        cfg        = get_optimal_params(model_bundle.model_path, dataset.path)
        score_type = str(cfg["score_type"])

        records = compute_dcq_records(
            model_bundle = model_bundle,
            dataset      = dataset,
            judge_model  = self.judge_model,
            cache_dir    = self.cache_dir,
            num_workers  = self.num_workers,
            placements   = self.placements,
            include_bdq  = True,                       # always on
            batch_size   = self.batch_size,
            max_tokens   = self.max_tokens,
        )

        # Partition-level BDQ statistics — required for pcorrect_kappa,
        # logged for diagnostic purposes regardless of score_type.
        partition_stats = compute_partition_stats(records)

        out_samples: list[dict[str, Any]] = []
        score_values: list[float] = []
        for rec in records:
            score = score_from_record(rec, score_type, partition_stats=partition_stats)
            score_values.append(score)
            out_samples.append({
                "record_index":      rec.record_index,
                "sample_id":         rec.sample_id,
                "label":             rec.label,
                "perturbations":     rec.options,
                "bdq_probs":         rec.bdq_probs,
                "bcq_probs":         rec.bcq_probs,
                "pcorrect_per_slot": rec.pcorrect_per_placement,
                "score":             float(score),
            })

        return {
            "method":              self.name,
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "score_type_used":     score_type,
            "judge_model":         self.judge_model,
            "placements_used":     list(self.placements),
            "include_bdq":         True,
            "partition_bdq_pe":    partition_stats,
            "non_preferred":       non_preferred_letters(partition_stats),
            "num_total_records":   len(dataset.records),
            "num_scored_records":  len(out_samples),
            "summary": {"mean_score": _safe_mean(score_values)},
            "samples": out_samples,
        }


def build_method(
    max_tokens:   int  = 4096,
    batch_size:   int  = 1,
    judge_model:  str  = "gpt-4o-mini-2024-07-18",
    cache_dir:    str  = ".dcq_cache",
    num_workers:  int  = 8,
    placements:   str  = "ABCD",
    include_bdq:  bool = True,
    **_,
) -> BaseMethod:
    return DCQMethod(
        max_tokens   = max_tokens,
        batch_size   = batch_size,
        judge_model  = judge_model,
        cache_dir    = cache_dir,
        num_workers  = num_workers,
        placements   = placements,
        include_bdq  = include_bdq,
    )