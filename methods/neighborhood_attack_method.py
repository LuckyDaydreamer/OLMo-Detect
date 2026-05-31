"""
Membership Inference Attacks against Language Models via Neighbourhood Comparison (Neighborhood Attack)
Paper: https://aclanthology.org/2023.findings-acl.719/
Github Implementation:  https://github.com/mireshghallah/neighborhood-curvature-mia 

Algorithm:
Given a target sample x and a target language model f_θ:
1. Generate n "neighbours" x̃_1, ..., x̃_n that are minimal perturbations
    of x (semantically and grammatically equivalent) but were almost
    certainly not in the model's training data.
2. Compute log-likelihoods LL(x), LL(x̃_1), ..., LL(x̃_n) under f_θ.
3. Score x via the difference between target LL and mean neighbour LL:
    score(x) = LL(x) - mean_i LL(x̃_i)            ("difference" / 'd')
    Optionally z-normalised:
    score(x) = (LL(x) - mean) / std                ("z-score" / 'z')
Higher score → x more likely a training member (i.e., contaminated):
- Member: LL(x) much higher than neighbours' (model overfit on x)
- Non-member: LL(x) ≈ neighbours' (model has no special preference)
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle
from progress_utils import progress


# ─── tuned hyper-parameters (post-tuning; update via neighborhood_attack_tune.py) ─

# `score_type` ∈ {"d", "z"} 

MATCHED_CONFIG: dict[str, dict[str, Any]] = {
    "OLMo2-1B-Instruct":  {"score_type": "d"},
    "OLMo2-7B-Instruct":  {"score_type": "d"},
    "OLMo2-13B-Instruct": {"score_type": "d"},
    "OLMo2-32B-Instruct": {"score_type": "d"},
}

SHIFTED_CONFIG: dict[str, dict[str, Any]] = {
    "OLMo2-1B-Instruct":  {"score_type": "z"},
    "OLMo2-7B-Instruct":  {"score_type": "d"},
    "OLMo2-13B-Instruct": {"score_type": "z"},
    "OLMo2-32B-Instruct": {"score_type": "z"},
}

_FALLBACK_CONFIG = {"score_type": "d"}


def get_optimal_params(model_path: str, dataset_path: str) -> dict[str, Any]:
    model_name = os.path.basename(model_path.rstrip("/"))
    file_name  = os.path.basename(dataset_path).lower()
    config_dict = SHIFTED_CONFIG if "shifted" in file_name else MATCHED_CONFIG
    config = config_dict.get(model_name, _FALLBACK_CONFIG)
    print(f"[neighborhood_attack] score_type={config['score_type']}")
    return config


VALID_SCORE_TYPES = ("d", "z")


# ─── T5 mask-fill ────────────────

_EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")


def tokenize_and_mask(
    text: str,
    span_length: int,
    pct: float,
    buffer_size: int,
    ceil_pct: bool = False,
    rng: np.random.RandomState | None = None,
) -> str:
    """
    Splits text on spaces, randomly masks `n_spans` non-overlapping spans
    of `span_length` tokens (with `buffer_size` tokens of guaranteed
    separation), replaces them with `<extra_id_NUM>` placeholders.

    `rng`: numpy RandomState. We thread one through instead of relying on
    np.random.seed for thread-safety in batched processing.
    """
    if rng is None:
        rng = np.random
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    span_length = min(int(pct * len(tokens)), span_length)
    span_length = max(1, span_length)

    n_spans_f = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = int(np.ceil(n_spans_f))
    else:
        n_spans = int(n_spans_f)

    if n_spans <= 0:
        # Trivially short text — no masks to insert. Caller will detect
        # and fall back to no-perturbation.
        return text

    n_masks = 0
    # Cap retries to avoid infinite loop on pathological inputs.
    max_attempts = 5 * n_spans + 50
    attempts = 0
    while n_masks < n_spans and attempts < max_attempts:
        attempts += 1
        upper = max(1, len(tokens) - span_length)
        start = int(rng.randint(0, upper))
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    if n_masks == 0:
        return text

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    return " ".join(tokens)


def count_masks(texts: list[str]) -> list[int]:
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


def extract_fills(texts: list[str]) -> list[list[str]]:
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    extracted_fills = [_EXTRA_ID_RE.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills


def apply_extracted_fills(masked_texts: list[str], extracted_fills: list[list[str]]) -> list[str]:
    tokens = [x.split(" ") for x in masked_texts]
    n_expected = count_masks(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                # Find and replace the placeholder
                try:
                    pos = text.index(f"<extra_id_{fill_idx}>")
                    text[pos] = fills[fill_idx]
                except ValueError:
                    tokens[idx] = []
                    break
    texts = [" ".join(x) for x in tokens]
    return texts


# ─── T5 mask model wrapper  ────────────

class _T5MaskModel:
    """
    Encapsulates the T5 mask-fill model. Loaded on demand, kept across runs
    in a process to amortise the load. Disk-cache of full perturbed sentence
    sets keyed on (target_text, generation params).
    """
    _cache: dict[str, "_T5MaskModel"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        model_name: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> "_T5MaskModel":
        with cls._lock:
            key = f"{model_name}::{device}::{dtype}"
            if key not in cls._cache:
                cls._cache[key] = cls(model_name, device=device, dtype=dtype)
            return cls._cache[key]

    def __init__(
        self,
        model_name: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        print(f"[neighborhood_attack] Loading mask-fill model: {model_name} on {self.device} ({dtype})")
        # Use slow tokenizer to avoid tokenizers-library issues with extra_id tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, use_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        # Get the integer ID for `<extra_id_99>` to use as a hard upper bound on stop_id
        # without re-encoding every batch.
        self._max_extra_id = 99  # T5 has 100 extra_ids by default

    @torch.no_grad()
    def replace_masks(
        self,
        masked_texts: list[str],
        mask_top_p: float = 1.0,
        max_length: int = 150,
    ) -> list[str]:
        """
        Run T5 generation to fill the masks:
        do_sample=True, top_p=mask_top_p, num_return_sequences=1.
        """
        n_expected = count_masks(masked_texts)
        if not any(n > 0 for n in n_expected):
            # Nothing to fill; return as-is
            return masked_texts
        max_n = max(n_expected)
        # stop_id = first ID returned for `<extra_id_{max_n}>`
        stop_id = self.tokenizer.encode(f"<extra_id_{max_n}>")[0]

        tokens = self.tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            **tokens,
            max_length        = max_length,
            do_sample         = True,
            top_p             = mask_top_p,
            num_return_sequences = 1,
            eos_token_id      = stop_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)


# ─── perturb_texts orchestration ─────────────

def _perturb_one_chunk(
    texts: list[str],
    t5_model: _T5MaskModel,
    span_length: int,
    pct: float,
    buffer_size: int,
    ceil_pct: bool,
    mask_top_p: float,
    rng: np.random.RandomState,
    max_retries: int = 10,
) -> list[str]:
    masked_texts = [tokenize_and_mask(x, span_length, pct, buffer_size, ceil_pct, rng=rng) for x in texts]
    raw_fills = t5_model.replace_masks(masked_texts, mask_top_p=mask_top_p)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    attempts = 0
    while "" in perturbed_texts and attempts < max_retries:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
        masked_retry = [
            tokenize_and_mask(texts[idx], span_length, pct, buffer_size, ceil_pct, rng=rng)
            for idx in idxs
        ]
        raw_retry = t5_model.replace_masks(masked_retry, mask_top_p=mask_top_p)
        extracted_retry = extract_fills(raw_retry)
        new_perturbed = apply_extracted_fills(masked_retry, extracted_retry)
        for idx, new_text in zip(idxs, new_perturbed):
            perturbed_texts[idx] = new_text
        attempts += 1

    # Final fallback: any still-empty text gets the original (would otherwise
    # break LL computation downstream)
    for idx, t in enumerate(perturbed_texts):
        if not t:
            perturbed_texts[idx] = texts[idx]
    return perturbed_texts


def perturb_texts(
    texts: list[str],
    t5_model: _T5MaskModel,
    span_length: int,
    pct: float,
    buffer_size: int,
    ceil_pct: bool,
    mask_top_p: float,
    chunk_size: int,
    seed: int,
    progress_desc: str = "neighbour gen",
) -> list[str]:
    """
    Chunked wrapper around `_perturb_one_chunk`. We use a deterministic RNG
    seeded by (seed, chunk_idx) so that running twice produces the same
    neighbours — important for the disk cache to hit.
    """
    outputs: list[str] = []
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    for chunk_idx, start in enumerate(progress(
        range(0, len(texts), chunk_size),
        desc=progress_desc, unit="chunk", total=n_chunks,
        dynamic_ncols=True, leave=False,
    )):
        chunk = texts[start : start + chunk_size]
        rng = np.random.RandomState(seed + chunk_idx)
        outputs.extend(_perturb_one_chunk(
            chunk, t5_model,
            span_length=span_length, pct=pct, buffer_size=buffer_size,
            ceil_pct=ceil_pct, mask_top_p=mask_top_p, rng=rng,
        ))
    return outputs


# ─── OLMo log-likelihood (one forward pass per text) ─────────────────────────

@torch.no_grad()
def _olmo_ll(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 4096,
) -> float:
    """
    Compute mean per-token log-likelihood under the model.
    Returns NaN for empty / pathological input.
    """
    if not text:
        return float("nan")
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=int(max_tokens), add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return float("nan")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids, labels=input_ids)
    return float(-out.loss.item())


@torch.no_grad()
def _olmo_lls_batch(
    model,
    tokenizer,
    texts: list[str],
    max_tokens: int = 4096,
    batch_size: int = 1,
    progress_desc: str = "olmo LL",
) -> list[float]:
    """
    Compute mean per-token log-likelihoods for a list of texts.
    For batch_size=1 (default for fairness across methods), this is a simple
    loop. For batch_size>1, we right-pad and mask out pad tokens before
    averaging the per-token loss.
    """
    if not texts:
        return []
    if batch_size <= 1:
        out = []
        for t in progress(texts, desc=progress_desc, unit="text",
                          dynamic_ncols=True, leave=False):
            out.append(_olmo_ll(model, tokenizer, t, max_tokens=max_tokens))
        return out

    # Batched path
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    out: list[float] = []
    for start in progress(range(0, len(texts), batch_size),
                          desc=progress_desc, unit="batch",
                          dynamic_ncols=True, leave=False):
        batch = texts[start : start + batch_size]
        # Filter empty texts; ll = NaN
        is_empty = [not t for t in batch]
        non_empty = [t if t else " " for t in batch]
        enc = tokenizer(
            non_empty, return_tensors="pt", padding=True, truncation=True,
            max_length=int(max_tokens), add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn      = enc["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            out.extend([float("nan")] * len(batch))
            continue

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        # Per-token NLL using the mask
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_attn   = attn[:, 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        # gather log P at the next-token positions
        gathered = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # Average over real (non-pad) tokens
        masked = gathered * shift_attn
        n_real = shift_attn.sum(dim=1).clamp_min(1)
        mean_ll = (masked.sum(dim=1) / n_real).cpu().tolist()
        for j, was_empty in enumerate(is_empty):
            out.append(float("nan") if was_empty else float(mean_ll[j]))
    return out


# ─── disk cache for T5 neighbours ────────────────────────────────────────────

def _neighbour_cache_key(
    text: str,
    n_perturbations: int,
    span_length: int,
    pct: float,
    buffer_size: int,
    mask_top_p: float,
    mask_filling_model_name: str,
    seed: int,
) -> str:
    h = hashlib.sha256()
    sig = "::".join([
        f"text={text}",
        f"n={n_perturbations}",
        f"sp={span_length}",
        f"pct={pct}",
        f"buf={buffer_size}",
        f"top_p={mask_top_p}",
        f"mfm={mask_filling_model_name}",
        f"seed={seed}",
    ])
    h.update(sig.encode("utf-8"))
    return h.hexdigest()


class _NeighbourDiskCache:
    """Per-record neighbour cache, JSON files under cache_dir."""
    def __init__(self, cache_dir: str | None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = threading.Lock()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> list[str] | None:
        p = self._path(key)
        if p is None or not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("neighbours"), list):
                return [str(s) for s in obj["neighbours"]]
        except Exception:
            return None
        return None

    def put(self, key: str, neighbours: list[str]) -> None:
        p = self._path(key)
        if p is None:
            return
        try:
            with self._lock:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump({"neighbours": neighbours}, f, ensure_ascii=False)
                tmp.replace(p)
        except Exception:
            pass


# ─── orchestration ───────────────────────────────────────────────────────────

@dataclass
class _NbrRecord:
    record_index: int
    sample_id: str
    label: Any
    text: str
    neighbours: list[str]
    target_ll: float
    neighbour_lls: list[float]


def _runtime_cache_key(
    dataset: DatasetBundle,
    n_perturbations: int,
    span_length: int,
    pct: float,
    buffer_size: int,
    mask_top_p: float,
    mask_filling_model_name: str,
    max_tokens: int,
    seed: int,
) -> str:
    return (f"nbrattack::{dataset.path}"
            f"::n={n_perturbations}::sp={span_length}::pct={pct}"
            f"::buf={buffer_size}::top_p={mask_top_p}"
            f"::mfm={mask_filling_model_name}::mt={max_tokens}::seed={seed}")


def compute_neighbour_records(
    model_bundle: ModelBundle,
    dataset:      DatasetBundle,
    n_perturbations: int = 10,
    pct_words_masked: float = 0.30,
    span_length: int = 2,
    buffer_size: int = 1,
    mask_top_p: float = 1.0,
    mask_filling_model_name: str = "t5-large",
    chunk_size: int = 20,
    ceil_pct: bool = False,
    cache_dir: str | None = ".neighborhood_cache",
    max_tokens: int = 4096,
    batch_size: int = 1,
    seed: int = 0,
    t5_dtype: torch.dtype = torch.float16,
) -> list[_NbrRecord]:
    """
    End-to-end pipeline. Three caches in play:
      1. T5 model object (process-level via _T5MaskModel.get)
      2. Per-record neighbour list (disk; SHARED across OLMo sizes)
      3. Per-record OLMo LLs (model_bundle.runtime_cache; per-OLMo-size)
    """
    # ── runtime cache hit: full per-record results already computed for THIS OLMo
    rt_key = _runtime_cache_key(
        dataset, n_perturbations, span_length, pct_words_masked,
        buffer_size, mask_top_p, mask_filling_model_name, max_tokens, seed,
    )
    cached = model_bundle.runtime_cache.get(rt_key)
    if cached is not None:
        return cached

    # ── 1. Identify usable records
    usable: list[tuple[int, dict, str]] = []
    for idx, record in enumerate(dataset.records):
        text = extract_text(record)
        if text and text.strip():
            usable.append((idx, record, text))
    if not usable:
        model_bundle.runtime_cache[rt_key] = []
        return []

    # ── 2. Per-record neighbour generation (disk-cached; shared across OLMos)
    disk_cache = _NeighbourDiskCache(cache_dir)
    # Find which records still need T5 work
    cached_neighbours: dict[int, list[str]] = {}
    needed: list[tuple[int, str]] = []
    for idx, _record, text in usable:
        ck = _neighbour_cache_key(
            text, n_perturbations, span_length, pct_words_masked,
            buffer_size, mask_top_p, mask_filling_model_name, seed,
        )
        cached = disk_cache.get(ck)
        if cached is not None and len(cached) == n_perturbations:
            cached_neighbours[idx] = cached
        else:
            needed.append((idx, text))

    if needed:
        # Batch over (record × n_perturbations) so chunk_size in T5 is large.
        # We replicate each text n_perturbations times, generate, then split back.
        flat_texts: list[str] = []
        flat_idxs:  list[int] = []
        for idx, text in needed:
            for _ in range(n_perturbations):
                flat_texts.append(text)
                flat_idxs.append(idx)

        t5 = _T5MaskModel.get(mask_filling_model_name, dtype=t5_dtype)
        flat_perturbed = perturb_texts(
            flat_texts, t5,
            span_length=span_length, pct=pct_words_masked,
            buffer_size=buffer_size, ceil_pct=ceil_pct,
            mask_top_p=mask_top_p, chunk_size=chunk_size, seed=seed,
            progress_desc=f"T5 neighbours [{dataset.name}]",
        )
        # Re-pack
        new_neighbours: dict[int, list[str]] = {}
        for idx, perturbed in zip(flat_idxs, flat_perturbed):
            new_neighbours.setdefault(idx, []).append(perturbed)
        # Save to disk and merge
        for idx, text in needed:
            ck = _neighbour_cache_key(
                text, n_perturbations, span_length, pct_words_masked,
                buffer_size, mask_top_p, mask_filling_model_name, seed,
            )
            nbrs = new_neighbours.get(idx, [])
            if len(nbrs) == n_perturbations:
                disk_cache.put(ck, nbrs)
            cached_neighbours[idx] = nbrs

    # ── 3. OLMo LLs
    # We call OLMo on (target + neighbours) for each usable record.
    out: list[_NbrRecord] = []
    for idx, record, text in progress(
        usable, desc=f"OLMo LL [{dataset.name}]", unit="rec",
        dynamic_ncols=True, leave=False,
    ):
        nbrs = cached_neighbours.get(idx, [])
        # Compute target LL
        target_ll = _olmo_ll(
            model_bundle.model, model_bundle.tokenizer, text,
            max_tokens=max_tokens,
        )
        # Compute neighbour LLs (batched if requested)
        if batch_size > 1 and nbrs:
            nbr_lls = _olmo_lls_batch(
                model_bundle.model, model_bundle.tokenizer, nbrs,
                max_tokens=max_tokens, batch_size=batch_size,
                progress_desc=None,  # inner progress would be noisy
            )
        else:
            nbr_lls = [
                _olmo_ll(model_bundle.model, model_bundle.tokenizer, n,
                         max_tokens=max_tokens)
                for n in nbrs
            ]
        out.append(_NbrRecord(
            record_index=idx,
            sample_id=extract_sample_id(record, fallback=idx),
            label=record.get("label"),
            text=text,
            neighbours=nbrs,
            target_ll=target_ll,
            neighbour_lls=nbr_lls,
        ))

    model_bundle.runtime_cache[rt_key] = out
    return out


# ─── per-record scoring ──────────────────────────────────────────────────────

def score_from_record(rec: _NbrRecord, score_type: str) -> float:
    """
    Eq. (3) of the paper, sign-flipped to put higher = more contaminated.

    'd': LL(x) - mean(LL(neighbours))
    'z': (LL(x) - mean(LL(neighbours))) / std(LL(neighbours))
    """
    if score_type not in VALID_SCORE_TYPES:
        raise ValueError(f"Unknown score_type={score_type!r}. Choose from {VALID_SCORE_TYPES}.")
    if not math.isfinite(rec.target_ll):
        return float("nan")
    finite_nbr = [v for v in rec.neighbour_lls if math.isfinite(v)]
    if not finite_nbr:
        return float("nan")
    mean_nbr = float(np.mean(finite_nbr))
    if score_type == "d":
        return float(rec.target_ll - mean_nbr)
    # z-score; std fallback to 1
    if len(finite_nbr) > 1:
        std_nbr = float(np.std(finite_nbr))
    else:
        std_nbr = 1.0
    if std_nbr == 0.0:
        std_nbr = 1.0
    return float((rec.target_ll - mean_nbr) / std_nbr)


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


# ─── BaseMethod ──────────────────────────────────────────────────────────────

class NeighborhoodAttackMethod(BaseMethod):
    """
    Fixed (paper defaults):
        n_perturbations  = 10
        pct_words_masked = 0.30
        span_length      = 2
        buffer_size      = 1
        mask_top_p       = 1.0
        mask_filling_model_name = "t5-large"
        seed = 0
    """
    name = "neighborhood_attack"

    def __init__(
        self,
        max_tokens: int = 4096,
        batch_size: int = 1,
        n_perturbations: int = 10,
        pct_words_masked: float = 0.30,
        span_length: int = 2,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        mask_filling_model_name: str = "t5-large",
        chunk_size: int = 20,
        ceil_pct: bool = False,
        cache_dir: str = ".neighborhood_cache",
        t5_dtype: str = "float16",   # "float16" | "float32" | "bfloat16"
        seed: int = 0,
    ):
        self.max_tokens = int(max_tokens)
        self.batch_size = int(batch_size)
        self.n_perturbations = int(n_perturbations)
        self.pct_words_masked = float(pct_words_masked)
        self.span_length = int(span_length)
        self.buffer_size = int(buffer_size)
        self.mask_top_p = float(mask_top_p)
        self.mask_filling_model_name = str(mask_filling_model_name)
        self.chunk_size = int(chunk_size)
        self.ceil_pct = bool(ceil_pct)
        self.cache_dir = str(cache_dir)
        self.seed = int(seed)
        self._t5_dtype = {
            "float16":  torch.float16,
            "fp16":     torch.float16,
            "float32":  torch.float32,
            "fp32":     torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16":     torch.bfloat16,
        }.get(t5_dtype, torch.float16)

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        cfg = get_optimal_params(model_bundle.model_path, dataset.path)
        score_type = str(cfg["score_type"])

        records = compute_neighbour_records(
            model_bundle    = model_bundle,
            dataset         = dataset,
            n_perturbations = self.n_perturbations,
            pct_words_masked= self.pct_words_masked,
            span_length     = self.span_length,
            buffer_size     = self.buffer_size,
            mask_top_p      = self.mask_top_p,
            mask_filling_model_name = self.mask_filling_model_name,
            chunk_size      = self.chunk_size,
            ceil_pct        = self.ceil_pct,
            cache_dir       = self.cache_dir,
            max_tokens      = self.max_tokens,
            batch_size      = self.batch_size,
            seed            = self.seed,
            t5_dtype        = self._t5_dtype,
        )

        out_samples: list[dict[str, Any]] = []
        score_values: list[float] = []
        for rec in records:
            score = score_from_record(rec, score_type)
            score_values.append(score)
            out_samples.append({
                "record_index": rec.record_index,
                "sample_id":    rec.sample_id,
                "label":        rec.label,
                "target_ll":    rec.target_ll,
                "neighbour_lls": rec.neighbour_lls,
                "neighbour_mean_ll": _safe_mean(rec.neighbour_lls),
                "n_neighbours": len(rec.neighbours),
                "score":        float(score),
            })

        return {
            "method":              self.name,
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "score_type_used":     score_type,
            "n_perturbations":     self.n_perturbations,
            "pct_words_masked":    self.pct_words_masked,
            "span_length":         self.span_length,
            "buffer_size":         self.buffer_size,
            "mask_top_p":          self.mask_top_p,
            "mask_filling_model_name": self.mask_filling_model_name,
            "max_tokens":          self.max_tokens,
            "seed":                self.seed,
            "num_total_records":   len(dataset.records),
            "num_scored_records":  len(out_samples),
            "summary": {"mean_score": _safe_mean(score_values)},
            "samples": out_samples,
        }


def build_method(
    max_tokens: int = 4096,
    batch_size: int = 1,
    n_perturbations: int = 10,
    pct_words_masked: float = 0.30,
    span_length: int = 2,
    buffer_size: int = 1,
    mask_top_p: float = 1.0,
    mask_filling_model_name: str = "t5-large",
    chunk_size: int = 20,
    ceil_pct: bool = False,
    cache_dir: str = ".neighborhood_cache",
    t5_dtype: str = "float16",
    seed: int = 0,
    **_,
) -> BaseMethod:
    return NeighborhoodAttackMethod(
        max_tokens     = max_tokens,
        batch_size     = batch_size,
        n_perturbations= n_perturbations,
        pct_words_masked = pct_words_masked,
        span_length    = span_length,
        buffer_size    = buffer_size,
        mask_top_p     = mask_top_p,
        mask_filling_model_name = mask_filling_model_name,
        chunk_size     = chunk_size,
        ceil_pct       = ceil_pct,
        cache_dir      = cache_dir,
        t5_dtype       = t5_dtype,
        seed           = seed,
    )