"""
RECALL: Membership Inference via Relative Conditional Log-Likelihoods
Paper: https://aclanthology.org/2024.emnlp-main.490
Github Implementation:  https://github.com/ruoyuxie/recall

Core formula:
  RECALL(x) = LL(x | P) / LL(x)

  where:
    LL(x)     = per-token mean log-likelihood of x (unconditional)
    LL(x | P) = per-token mean log-likelihood of x conditioned on prefix P
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle
from progress_utils import progress


# ─── hardcoded n configs ──────────────────────────────────────────────────────
# n = number of non-member prefix shots.
# Update after running recall_tune.py.

MATCHED_CONFIG = {
    "OLMo2-1B-Instruct":  {"n": 10},
    "OLMo2-7B-Instruct":  {"n": 10},
    "OLMo2-13B-Instruct": {"n": 5},
    "OLMo2-32B-Instruct": {"n": 11},
}

SHIFTED_CONFIG = {
    "OLMo2-1B-Instruct":  {"n": 12},
    "OLMo2-7B-Instruct":  {"n": 12},
    "OLMo2-13B-Instruct": {"n": 11},
    "OLMo2-32B-Instruct": {"n": 12},
}


def get_optimal_n(model_path: str, dataset_path: str) -> int:
    model_name = os.path.basename(model_path.rstrip("/"))
    file_name  = os.path.basename(dataset_path).lower()
    config_dict = SHIFTED_CONFIG if "shifted" in file_name else MATCHED_CONFIG
    config = config_dict.get(model_name, {"n": 5})
    n = config["n"]
    print(f"[recall] n={n}")
    return n


# ─── prefix construction ──────────────────────────────────────────────────────

def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def build_prefix(
    prefix_texts: list[str],
    n: int,
    tokenizer,
    avg_target_len: int,
    max_model_len: int,
    max_tokens: int,
) -> tuple[list[int], int]:
    """
    Build a prefix of n shots that fits within (max_model_len - avg_target_len).

    Each shot is truncated to min(max_tokens, budget // n) tokens, so all n
    shots always fit by construction. Required when prefix texts exceed the
    context window (e.g. 10k-token docs with a 4k model).

    Returns (prefix_token_ids, actual_shots_used).
    """
    n_capped = min(n, len(prefix_texts))
    if n_capped == 0:
        return [], 0

    budget = max_model_len - avg_target_len
    if budget <= 0:
        return [], 0

    per_shot = min(max_tokens, budget // n_capped)
    if per_shot <= 0:
        return [], 0

    kept_ids = [
        tokenizer.encode(t, add_special_tokens=True)[:per_shot]
        for t in prefix_texts[:n_capped]
    ]
    prefix_ids = [tok for ids in kept_ids for tok in ids]
    return prefix_ids, len(kept_ids)


# ─── batched conditional LL ───────────────────────────────────────────────────

def _batch_conditional_ll(
    model,
    tokenizer,
    device: torch.device,
    prefix_ids: list[int],
    target_texts: list[str],
    max_tokens: int,
    batch_size: int,
) -> list[float]:
    """
    Compute mean per-token LL of each target text conditioned on the fixed prefix.

    Matches get_conditional_ll() in official code:
      concat_ids = [prefix_ids | target_ids]
      labels[:, :prefix_len] = -100
      ll = -model(concat_ids, labels=labels).loss.item()

    Batched: all N targets processed in ceil(N/batch_size) forward passes.
    """
    prefix_len = len(prefix_ids)
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    results: list[float] = []

    for start in range(0, len(target_texts), batch_size):
        batch_texts = target_texts[start : start + batch_size]

        # Tokenize targets (matching official: tokenizer(text) with add_special_tokens default True)
        target_ids_list = [
            tokenizer.encode(t, add_special_tokens=True)[:max_tokens]
            for t in batch_texts
        ]

        max_target_len = max(len(ids) for ids in target_ids_list)
        max_seq_len    = prefix_len + max_target_len
        B              = len(batch_texts)

        input_ids      = torch.full((B, max_seq_len), pad_id,  dtype=torch.long, device=device)
        labels         = torch.full((B, max_seq_len), -100,    dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_seq_len),          dtype=torch.long, device=device)

        for i, tids in enumerate(target_ids_list):
            tlen = len(tids)
            seq_len = prefix_len + tlen
            # Fill input_ids: [prefix | target | pad]
            input_ids[i, :prefix_len] = prefix_tensor
            input_ids[i, prefix_len:seq_len] = torch.tensor(tids, dtype=torch.long, device=device)
            # Labels: -100 for prefix, target_ids for target, -100 for pad (already -100)
            labels[i, prefix_len:seq_len] = torch.tensor(tids, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

        with torch.no_grad():
            out    = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits   # (B, T, V)

        # Shift: logits[:, t, :] predicts input_ids[:, t+1]
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, T-1)
        del logits, out

        for i in range(B):
            mask = shift_labels[i] != -100
            if mask.sum() == 0:
                results.append(float("nan"))
                continue
            valid_logits = shift_logits[i][mask]           # (n_tgt, V)
            valid_labels = shift_labels[i][mask]            # (n_tgt,)
            log_probs = torch.log_softmax(valid_logits, dim=-1)
            token_lls = log_probs[
                torch.arange(len(valid_labels), device=device), valid_labels
            ]
            results.append(float(token_lls.mean().item()))
        
        del shift_logits, shift_labels, input_ids, labels, attention_mask
        torch.cuda.empty_cache()

    return results


# ─── RECALLMethod ─────────────────────────────────────────────────────────────

class RECALLMethod(BaseMethod):
    """
    RECALL contamination detector.

    Score = LL(x|P) / LL(x).  Higher = more likely contaminated.
    Prefix P = concatenation of n uncontaminated (non-member) texts.
    n is tuned on dev split via recall_tune.py.
    """
    name = "recall"

    def __init__(
        self,
        uncon_dev_path: str,
        max_tokens:     int = 1024,
        batch_size:     int = 3,
        max_model_len:  int = 4096,
        **kwargs,
    ):
        self.uncon_dev_path = uncon_dev_path
        self.max_tokens     = max_tokens
        self.batch_size     = batch_size
        self.max_model_len  = max_model_len

    # ── prefix loading ─────────────────────────────────────────────────────
    def _load_prefix_texts(self) -> list[str]:
        texts = []
        with open(self.uncon_dev_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = extract_text(record)
                if text:
                    texts.append(text)
        return texts

    # ── main run ───────────────────────────────────────────────────────────
    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        tokenizer = model_bundle.tokenizer
        model     = model_bundle.model
        device    = next(model.parameters()).device

        # 1. Resolve n from hardcoded config
        n = get_optimal_n(model_bundle.model_path, dataset.path)

        # 2. Prepare eval records
        prepared: list[dict] = []
        for idx, record in enumerate(dataset.records):
            text = extract_text(record)
            if text:
                prepared.append({
                    "idx":  idx,
                    "id":   extract_sample_id(record, fallback=idx),
                    "text": text,
                })

        if not prepared:
            return self._empty(dataset)

        texts = [item["text"] for item in prepared]

        # 3. Compute avg target length for context window budget
        avg_target_len = int(np.mean([
            min(len(tokenizer.encode(t, add_special_tokens=True)), self.max_tokens)
            for t in texts
        ]))

        # 4. Build prefix
        prefix_texts = self._load_prefix_texts()
        prefix_ids, shots_used = build_prefix(
            prefix_texts, n, tokenizer, avg_target_len,
            self.max_model_len, self.max_tokens,
        )
        print(f"[recall] prefix: {shots_used} shots, {len(prefix_ids)} tokens")

        # 5. Phase 1 — unconditional LLs (batched via existing adapter)
        print(f"[recall] computing unconditional LLs ({len(texts)} samples)...")
        uncon_stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
            texts         = texts,
            batch_size    = self.batch_size,
            max_tokens    = self.max_tokens,
            progress_desc = f"recall uncond [{dataset.name}]",
        )
        # Mean per-token LL = mean of token_log_probs = -loss.item() (matches official get_ll)
        ll_uncond: list[float] = []
        for s in uncon_stats:
            tlp = s.get("token_log_probs", [])
            ll_uncond.append(float(np.mean(tlp)) if tlp else float("nan"))

        # 6. Phase 2 — conditional LLs (batched, prefix prepended)
        if shots_used == 0 or len(prefix_ids) == 0:
            print("[recall] WARNING: prefix is empty (n=0 or context too small). "
                  "RECALL score will be NaN for all samples.")
            ll_cond = [float("nan")] * len(texts)
        else:
            print(f"[recall] computing conditional LLs ({len(texts)} samples)...")
            ll_cond = _batch_conditional_ll(
                model        = model,
                tokenizer    = tokenizer,
                device       = device,
                prefix_ids   = prefix_ids,
                target_texts = texts,
                max_tokens   = self.max_model_len - len(prefix_ids),
                batch_size   = self.batch_size,
            )

        # 7. Compute RECALL scores
        samples: list[dict] = []
        score_values: list[float] = []

        for i, item in enumerate(prepared):
            ll_u = ll_uncond[i]
            ll_c = ll_cond[i]
            # RECALL = LL(x|P) / LL(x)  — both negative, members have higher ratio
            if math.isfinite(ll_u) and math.isfinite(ll_c) and ll_u != 0.0:
                score = ll_c / ll_u
                if not math.isfinite(score):
                    score = float("nan")
            else:
                score = float("nan")
            score_values.append(score)
            samples.append({
                "record_index":        item["idx"],
                "sample_id":           item["id"],
                "text":                item["text"],
                "ll_unconditional":    ll_u,
                "ll_conditional":      ll_c,
                "recall_score":        score,
            })

        return {
            "method":                  self.name,
            "dataset":                 dataset.name,
            "dataset_path":            dataset.path,
            "dataset_data_type":       dataset.data_type,
            "n_shots":                 n,
            "shots_used":              shots_used,
            "prefix_token_len":        len(prefix_ids),
            "max_tokens":              self.max_tokens,
            "uncon_dev_path":          self.uncon_dev_path,
            "num_total_records":       len(dataset.records),
            "num_scored_records":      len(samples),
            "num_skipped_records":     len(dataset.records) - len(samples),
            "summary": {
                "mean_recall_score": _safe_mean(score_values),
            },
            "samples": samples,
        }

    @staticmethod
    def _empty(dataset: DatasetBundle) -> dict[str, Any]:
        return {
            "method":             "recall",
            "dataset":            dataset.name,
            "dataset_path":       dataset.path,
            "dataset_data_type":  dataset.data_type,
            "num_total_records":  len(dataset.records),
            "num_scored_records": 0,
            "num_skipped_records": len(dataset.records),
            "summary":            {},
            "samples":            [],
        }


def build_method(
    uncon_dev_path: str,
    max_tokens:     int = 1024,
    batch_size:     int = 3,
    **kwargs,
) -> BaseMethod:
    return RECALLMethod(
        uncon_dev_path = uncon_dev_path,
        max_tokens     = max_tokens,
        batch_size     = batch_size,
    )