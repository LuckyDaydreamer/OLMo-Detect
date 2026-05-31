"""
Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models (CDD)
Paper: https://arxiv.org/pdf/2402.15938
Github Implementation:  https://github.com/YihongDong/CDD-TED4LLMs 
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle
from progress_utils import progress


# ─── tuned hyper-parameters (post-tuning; update via tune_cdd.py) ─────────────

MATCHED_CONFIG: dict[str, dict[str, float]] = {
    "OLMo2-1B-Instruct":  {"alpha": 0.3, "length_cap": 50, "prompt_ratio": 0.75},
    "OLMo2-7B-Instruct":  {"alpha": 0.2, "length_cap": 50, "prompt_ratio": 0.25},
    "OLMo2-13B-Instruct": {"alpha": 0.3, "length_cap": 50, "prompt_ratio": 0.5},
    "OLMo2-32B-Instruct": {"alpha": 0.3, "length_cap": 50, "prompt_ratio": 0.5},
}

SHIFTED_CONFIG: dict[str, dict[str, float]] = {
    "OLMo2-1B-Instruct":  {"alpha": 0.3, "length_cap": 50, "prompt_ratio": 0.5},
    "OLMo2-7B-Instruct":  {"alpha": 0.3, "length_cap": 50, "prompt_ratio": 0.5},
    "OLMo2-13B-Instruct": {"alpha": 0.3, "length_cap": 50, "prompt_ratio": 0.5},
    "OLMo2-32B-Instruct": {"alpha": 0.2, "length_cap": 50, "prompt_ratio": 0.5},
}

_FALLBACK_CONFIG = {"alpha": 0.05, "length_cap": 100, "prompt_ratio": 0.5}


def get_optimal_cdd_params(
    model_path: str,
    dataset_path: str,
) -> dict[str, float]:
    model_name = os.path.basename(model_path.rstrip("/"))
    file_name  = os.path.basename(dataset_path).lower()
    config_dict = SHIFTED_CONFIG if "shifted" in file_name else MATCHED_CONFIG
    config = config_dict.get(model_name, _FALLBACK_CONFIG)
    print(f"[cdd] alpha={config['alpha']} "
          f"length_cap={config['length_cap']} "
          f"prompt_ratio={config['prompt_ratio']}")
    return config


# ─── core algorithm helpers ─────────────────────────

def levenshtein_distance(a, b) -> int:
    """
    Token-level edit distance. 
    Memory: O(min(|a|, |b|)) via rolling array.
    """
    if len(a) > len(b):
        a, b = b, a
    distances = list(range(len(a) + 1))
    for j, cb in enumerate(b):
        new_distances = [j + 1]
        for i, ca in enumerate(a):
            if ca == cb:
                new_distances.append(distances[i])
            else:
                new_distances.append(
                    1 + min(distances[i], distances[i + 1], new_distances[-1])
                )
        distances = new_distances
    return distances[-1]


def calculate_ratio(numbers: list[int], alpha_l: float) -> float:
    """
    fraction of distances ≤ alpha · max_length.
    """
    if not numbers:
        return 0.0
    count = sum(1 for d in numbers if d <= alpha_l)
    return count / len(numbers)


def _strip_special_suffix(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    """Cut the sequence at the first EOS / PAD token, exclusive — matches
    the behaviour of HFGenerationAdapter.generate."""
    out: list[int] = []
    for t in token_ids:
        if t in stop_ids:
            break
        out.append(t)
    return out


# ─── generation: one prompt → 1 greedy + n_samples sampled completions ────────

def _generate_for_prompt(
    model,
    tokenizer,
    prompt_ids: list[int],
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    sub_batch_size: int | None,
) -> tuple[list[int], list[list[int]]]:
    """
    Returns (greedy_completion_ids, [sample_completion_ids x n_samples]).
    Uses `num_return_sequences` for the sampled batch so the prompt prefill
    is shared across all n samples (much faster than n independent generate
    calls).
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    stop_ids: set[int] = set()
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    if pad_id is not None:
        stop_ids.add(int(pad_id))

    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    prompt_len = input_ids.shape[1]

    # ── 1. greedy ────────────────────────────────────────────────────────────
    with torch.no_grad():
        out_g = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=False,
            use_cache=True,
        )
    greedy_completion = out_g[0, prompt_len:].tolist()
    greedy_completion = _strip_special_suffix(greedy_completion, stop_ids)

    # ── 2. sampled (chunked if requested) ────────────────────────────────────
    if sub_batch_size is None or sub_batch_size <= 0:
        chunks = [n_samples]
    else:
        chunks = []
        remaining = n_samples
        while remaining > 0:
            chunks.append(min(sub_batch_size, remaining))
            remaining -= chunks[-1]

    sampled: list[list[int]] = []
    sub_seed = seed
    for chunk in chunks:
        torch.manual_seed(int(sub_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sub_seed))
        sub_seed += 1
        with torch.no_grad():
            out_s = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=chunk,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                return_dict_in_generate=False,
                use_cache=True,
            )
        for row in range(chunk):
            comp = out_s[row, prompt_len:].tolist()
            comp = _strip_special_suffix(comp, stop_ids)
            sampled.append(comp)

    del input_ids
    torch.cuda.empty_cache()

    return greedy_completion, sampled


# ─── per-dataset sampling (cached on model_bundle) ────────────────────────────

def _sampling_cache_key(
    dataset: DatasetBundle,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    prompt_ratio: float,
    max_tokens: int,
) -> str:
    return (
        f"cdd_samples::{dataset.path}::{dataset.data_type}"
        f"::n={n_samples}::T={temperature}::mnt={max_new_tokens}"
        f"::pr={prompt_ratio}::mt={max_tokens}"
    )


def compute_cdd_samples(
    model_bundle: ModelBundle,
    dataset: DatasetBundle,
    n_samples: int = 50,
    temperature: float = 0.8,
    max_new_tokens: int = 100,
    prompt_ratio: float = 0.5,
    max_tokens: int = 4096,
    sub_batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Run greedy + n_samples generation on every record. 
    """
    key = _sampling_cache_key(
        dataset, n_samples=n_samples, temperature=temperature,
        max_new_tokens=max_new_tokens, prompt_ratio=prompt_ratio,
        max_tokens=max_tokens,
    )
    cached = model_bundle.runtime_cache.get(key)
    if cached is not None:
        return cached

    model     = model_bundle.model
    tokenizer = model_bundle.tokenizer
    base_seed = model_bundle.seed

    prepared: list[dict[str, Any]] = []
    for idx, record in enumerate(dataset.records):
        text = extract_text(record)
        if not text:
            continue
        prepared.append({
            "record_index": idx,
            "sample_id":    extract_sample_id(record, fallback=idx),
            "label":        record.get("label"),
            "text":         text,
        })

    samples_out: list[dict[str, Any]] = []
    max_prompt_len = max(2, int(max_tokens) - int(max_new_tokens))

    for item_idx, item in enumerate(progress(
        prepared,
        desc=f"cdd sample [{dataset.name}]",
        unit="sample",
        dynamic_ncols=True,
        leave=False,
    )):
        text = item["text"]
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(full_ids) > max_prompt_len:
            full_ids = full_ids[:max_prompt_len]

        # Split: first prompt_ratio fraction of tokens is the prompt.
        split = int(len(full_ids) * prompt_ratio)
        if split < 1:
            split = 1
        if split >= len(full_ids):
            split = max(1, len(full_ids) - 1)
        prompt_ids = full_ids[:split]

        per_sample_seed = int(base_seed) + item_idx * 100  # stride avoids overlap with sub-seeds

        try:
            greedy_ids, sample_ids_list = _generate_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=per_sample_seed,
                sub_batch_size=sub_batch_size,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            greedy_ids, sample_ids_list = _generate_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=per_sample_seed,
                sub_batch_size=1,
            )

        samples_out.append({
            "record_index":          item["record_index"],
            "sample_id":             item["sample_id"],
            "label":                 item["label"],
            "prompt_token_count":    len(prompt_ids),
            "greedy_completion_ids": greedy_ids,
            "sample_completion_ids": sample_ids_list,
        })

    result = {
        "dataset":           dataset.name,
        "dataset_path":      dataset.path,
        "dataset_data_type": dataset.data_type,
        "n_samples":         n_samples,
        "temperature":       temperature,
        "max_new_tokens":    max_new_tokens,
        "prompt_ratio":      prompt_ratio,
        "max_tokens":        max_tokens,
        "samples":           samples_out,
    }
    model_bundle.runtime_cache[key] = result
    return result


# ─── score one record from cached samples  ────────────────

def score_record(
    greedy_completion_ids: list[int],
    sample_completion_ids: list[list[int]],
    alpha: float,
    length_cap: int,
) -> tuple[float, list[int], int]:
    """
    Returns (peak, edit_distances, max_length) — same convention as the repo's
    `get_edit_distance_distribution_star` + `calculate_ratio`.
    """
    g = greedy_completion_ids[:length_cap]
    distances: list[int] = []
    max_length = len(g)
    for s_full in sample_completion_ids:
        s = s_full[:length_cap]
        distances.append(levenshtein_distance(g, s))
        if len(s) > max_length:
            max_length = len(s)
    peak = calculate_ratio(distances, alpha * max_length)
    return peak, distances, max_length


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


# ─── BaseMethod ────────────────

class CDDMethod(BaseMethod):
    """
    Score convention: higher peak ⇒ more likely contaminated. AUROC scripts
    that call `roc_auc_score(y_true=1_for_contaminated, scores)` on the
    `peak` field will work directly.

    Hyper-parameters
    ----------------
    Tuned per (model, matched/shifted) — read from MATCHED_CONFIG / SHIFTED_CONFIG
    via `get_optimal_cdd_params()`:
        alpha, length_cap, prompt_ratio
    Fixed (paper defaults):
        n_samples = 50, temperature = 0.8, xi = 0.01
    """
    name = "cdd"

    def __init__(
        self,
        n_samples:      int   = 50,    # paper default
        temperature:    float = 0.8,   # paper default
        xi:             float = 0.01,  # paper default; binary decision only
        max_tokens:     int   = 4096,  # framework wildcard
        batch_size:     int   = 1,     # framework wildcard (unused in CDD; sampling is per-prompt)
        sub_batch_size: int | None = None,  # OOM-mitigation knob for num_return_sequences
    ):
        self.n_samples      = int(n_samples)
        self.temperature    = float(temperature)
        self.xi             = float(xi)
        self.max_tokens     = int(max_tokens)
        self.batch_size     = int(batch_size)
        self.sub_batch_size = sub_batch_size

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        # Always pull tuned values from CONFIG — no override path.
        cfg          = get_optimal_cdd_params(model_bundle.model_path, dataset.path)
        alpha        = float(cfg["alpha"])
        length_cap   = int(cfg["length_cap"])
        prompt_ratio = float(cfg["prompt_ratio"])

        # 1. Generate (or fetch from cache) greedy + n_samples for every record.
        #    max_new_tokens = length_cap: we never look at tokens beyond length_cap,
        #    so generating more would be wasted compute.
        sample_data = compute_cdd_samples(
            model_bundle    = model_bundle,
            dataset         = dataset,
            n_samples       = self.n_samples,
            temperature     = self.temperature,
            max_new_tokens  = length_cap,
            prompt_ratio    = prompt_ratio,
            max_tokens      = self.max_tokens,
            sub_batch_size  = self.sub_batch_size,
        )

        # 2. Score every record.
        out_samples: list[dict[str, Any]] = []
        peaks: list[float] = []
        for item in sample_data["samples"]:
            peak, distances, ml = score_record(
                greedy_completion_ids = item["greedy_completion_ids"],
                sample_completion_ids = item["sample_completion_ids"],
                alpha                 = alpha,
                length_cap            = length_cap,
            )
            peaks.append(peak)
            out_samples.append({
                "record_index":       item["record_index"],
                "sample_id":          item["sample_id"],
                "label":              item["label"],
                "prompt_token_count": item["prompt_token_count"],
                "greedy_token_count": min(len(item["greedy_completion_ids"]), length_cap),
                "n_samples":          len(item["sample_completion_ids"]),
                "edit_distances":     distances,
                "max_length":         ml,
                "peak":               float(peak),
                "is_contaminated":    bool(peak > self.xi),
            })

        return {
            "method":              self.name,
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "alpha_used":          alpha,
            "length_cap_used":     length_cap,
            "prompt_ratio_used":   prompt_ratio,
            "xi_used":             self.xi,
            "n_samples":           self.n_samples,
            "temperature":         self.temperature,
            "max_tokens":          self.max_tokens,
            "num_total_records":   len(dataset.records),
            "num_scored_records":  len(out_samples),
            "num_skipped_records": len(dataset.records) - len(out_samples),
            "summary": {
                "mean_peak":           _safe_mean(peaks),
                "contamination_ratio": _safe_mean([1.0 if p > self.xi else 0.0 for p in peaks]),
            },
            "samples": out_samples,
        }


def build_method(
    n_samples:      int   = 50,
    temperature:    float = 0.8,
    xi:             float = 0.01,
    max_tokens:     int   = 4096,
    batch_size:     int   = 1,
    sub_batch_size: int | None = None,
    **_,
) -> BaseMethod:
    return CDDMethod(
        n_samples      = n_samples,
        temperature    = temperature,
        xi             = xi,
        max_tokens     = max_tokens,
        batch_size     = batch_size,
        sub_batch_size = sub_batch_size,
    )