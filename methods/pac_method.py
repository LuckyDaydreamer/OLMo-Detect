"""
Data Contamination Calibration for Black-box LLMs (PAC)
Paper: https://aclanthology.org/2024.findings-acl.644.pdf
Github Implementation:  https://github.com/yyy01/PAC

Algorithm:
1. Augmentation:
     For each sample z, generate N=5 augmented copies via word-level
     random swaps.  Number of swaps per copy = max(1, int(0.3 * n_words)).

2. Log-prob computation (calculate_probs_others):
     Tokenize text, forward pass, log_softmax → per-token log-probs.
     Token at position i gets log P(token_i | token_0..i-1) for i=1..L-1.

3. Polarized Distance (calculate_Polarized_Distance):
     Sort log-probs ascending.
     local_region = bottom 30% (most negative, least likely tokens)
     far_region   = top 5%    (least negative, most likely tokens)
     L_M = mean(far_region) - mean(local_region)     (always ≥ 0)

4. Calibration:
     PAC(z) = L_M(z) - mean(L_M(augmented_i) for i=1..N)

5. Score direction:
     Code uses roc_auc_score(labels, [-p for p in PAC_list]),
     meaning members have LOWER PAC (more negative).
     We output -PAC so higher score → more likely member,
     consistent with our pipeline's convention.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle


# ─────────────────────────── augmentation ────────────────────────

def _swap_word(words: list[str]) -> list[str]:
    """Swap two random words. Exact match with eda.py → swap_word."""
    idx1 = random.randint(0, len(words) - 1)
    idx2 = idx1
    counter = 0
    while idx2 == idx1:
        idx2 = random.randint(0, len(words) - 1)
        counter += 1
        if counter > 3:
            return words
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words


def _random_swap(words: list[str], n: int) -> list[str]:
    """Apply n random swaps. Exact match with eda.py → random_swap."""
    new_words = words.copy()
    for _ in range(n):
        new_words = _swap_word(new_words)
    return new_words


def _eda(sentence: str, alpha: float = 0.3, num_aug: int = 5) -> list[str]:
    """
    Generate augmented sentences via word-level random swaps.
    Exact match with eda.py → eda().
    """
    words = sentence.split(' ')
    num_words = len(words)
    augmented = []
    if alpha > 0:
        n_rs = max(1, int(alpha * num_words))
        for _ in range(num_aug):
            a_words = _random_swap(words, n_rs)
            augmented.append(' '.join(a_words))
    random.shuffle(augmented)
    return augmented[:num_aug] if num_aug >= 1 else augmented


# ─────────────────────── polarized distance ───────────────────

def _polarized_distance(
    log_probs: list[float],
    ratio_local: float = 0.3,
    ratio_far: float = 0.05,
) -> float:
    """
    Compute the polarized distance L_M.
    Exact match with attack.py → calculate_Polarized_Distance.

    local_region = bottom ratio_local fraction (most negative log-probs)
    far_region   = top ratio_far fraction (least negative log-probs)
    L_M = mean(far) - mean(local)
    """
    if len(log_probs) == 0:
        return 0.0
    arr = np.array(log_probs)
    local_len = max(int(len(arr) * ratio_local), 1)
    far_len = max(int(len(arr) * ratio_far), 1)
    sorted_arr = np.sort(arr)
    local_region = sorted_arr[:local_len]           # most negative
    far_region = sorted_arr[::-1][:far_len]          # least negative
    return float(np.mean(far_region) - np.mean(local_region))


# ────────────────────────────── main method ───────────────────────────────

class PACMethod(BaseMethod):
    name = "pac"

    def __init__(
        self,
        max_tokens: int = 512,
        batch_size: int = 1,
        n_aug: int = 5,
        alpha: float = 0.3,
        ratio_local: float = 0.3,
        ratio_far: float = 0.05,
        seed: int = 0,
        **kwargs,
    ):
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.n_aug = n_aug
        self.alpha = alpha
        self.ratio_local = ratio_local
        self.ratio_far = ratio_far
        self.seed = seed

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:

        # ── auto-resolve k1/k2 from config if not explicitly set ─────────
        if self.ratio_far == 0.0 or self.ratio_local == 0.0:
            model_size = _detect_model_size(model_bundle.model_path)
            shifted = _detect_shifted(dataset.path)
            cfg = _lookup_config(model_size, shifted)
            if self.ratio_far == 0.0:
                self.ratio_far = cfg["ratio_far"]
            if self.ratio_local == 0.0:
                self.ratio_local = cfg["ratio_local"]
            print(f"[pac] auto-config: ratio_far={self.ratio_far}, "
                  f"ratio_local={self.ratio_local} "
                  f"(model={model_size}, shifted={shifted})")

        # ── prepare texts ────────────────────────────────────────────────
        prepared: list[dict] = []
        for idx, record in enumerate(dataset.records):
            text = extract_text(record)
            if text:
                prepared.append({
                    "idx": idx,
                    "id": extract_sample_id(record, fallback=idx),
                    "text": text,
                })
        if not prepared:
            return self._empty(dataset)

        texts = [p["text"] for p in prepared]
        n = len(texts)

        # ── augmentation (word-level random swaps) ───────────────────────
        random.seed(self.seed)
        aug_texts: list[str] = []
        for text in texts:
            augs = _eda(text, alpha=self.alpha, num_aug=self.n_aug)
            aug_texts.extend(augs)

        # ── compute log-probs: originals + augmented in one batch ────────
        all_texts = texts + aug_texts
        print(f"[pac] {n} originals + {len(aug_texts)} augmented "
              f"= {len(all_texts)} total forward passes")

        all_stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
            texts=all_texts,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            progress_desc="pac",
        )
        all_lps = [s.get("token_log_probs", []) for s in all_stats]

        orig_lps = all_lps[:n]
        aug_lps = all_lps[n:]

        # ── polarized distance for each ──────────────────────────────────
        orig_pds = [
            _polarized_distance(lps, self.ratio_local, self.ratio_far)
            for lps in orig_lps
        ]
        aug_pds = [
            _polarized_distance(lps, self.ratio_local, self.ratio_far)
            for lps in aug_lps
        ]

        # ── calibration: PAC = PD(z) - mean(PD(augmented)) ──────────────
        # Then negate: score = -PAC so higher = more likely member
        scores: list[float] = []
        for i in range(n):
            aug_start = i * self.n_aug
            aug_end = aug_start + self.n_aug
            mean_aug_pd = float(np.mean(aug_pds[aug_start:aug_end]))
            pac_raw = orig_pds[i] - mean_aug_pd
            score = -pac_raw  # negate: higher = more likely member
            scores.append(score)

        # ── assemble output ──────────────────────────────────────────────
        samples: list[dict] = []
        for i, p in enumerate(prepared):
            samples.append({
                "record_index": p["idx"],
                "sample_id": p["id"],
                "text": p["text"],
                "pac_score": scores[i],
            })

        return {
            "method": self.name,
            "dataset": dataset.name,
            "dataset_path": dataset.path,
            "dataset_data_type": dataset.data_type,
            "max_tokens": self.max_tokens,
            "n_aug": self.n_aug,
            "alpha": self.alpha,
            "ratio_local": self.ratio_local,
            "ratio_far": self.ratio_far,
            "num_total_records": len(dataset.records),
            "num_scored_records": len(samples),
            "num_skipped_records": len(dataset.records) - len(samples),
            "summary": {
                "mean_pac_score": float(np.mean(scores)) if scores else float("nan"),
            },
            "samples": samples,
        }

    @staticmethod
    def _empty(dataset: DatasetBundle) -> dict[str, Any]:
        return {
            "method": "pac",
            "dataset": dataset.name,
            "dataset_path": dataset.path,
            "dataset_data_type": dataset.data_type,
            "num_total_records": len(dataset.records),
            "num_scored_records": 0,
            "num_skipped_records": len(dataset.records),
            "summary": {},
            "samples": [],
        }




# ───────────────────────────── tuned configs ─────────────────────────────
# After running pac_tune.py on dev data, update these per model size.
# Fallback: published defaults from paper §5.2 (k1=0.05, k2=0.30).

PAC_CONFIG: dict[str, dict[str, float]] = {
    # Tuned via pac_tune.py on the matched dev split.
    "1B":  {"ratio_far": 0.05, "ratio_local": 0.10},
    "7B":  {"ratio_far": 0.05, "ratio_local": 0.10},
    "13B": {"ratio_far": 0.05, "ratio_local": 0.10},
    "32B": {"ratio_far": 0.05, "ratio_local": 0.10},
}

PAC_CONFIG_SHIFTED: dict[str, dict[str, float]] = {
    # Tuned via pac_tune.py on the shifted dev split.
    "1B":  {"ratio_far": 0.10, "ratio_local": 0.30},
    "7B":  {"ratio_far": 0.05, "ratio_local": 0.40},
    "13B": {"ratio_far": 0.05, "ratio_local": 0.20},
    "32B": {"ratio_far": 0.05, "ratio_local": 0.40},
}

_DEFAULT_K1 = 0.05
_DEFAULT_K2 = 0.30


def _detect_model_size(model_path: str) -> str:
    import os
    name = os.path.basename((model_path or "").rstrip("/")).lower()
    for size in ("32b", "13b", "7b", "1b"):
        if size in name:
            return size.upper()
    return ""


def _detect_shifted(dataset_path: str) -> bool:
    p = (dataset_path or "").lower()
    return ("_shifted_" in p) or ("/shifted/" in p)


def _lookup_config(model_size: str, shifted: bool) -> dict[str, float]:
    cfg = PAC_CONFIG_SHIFTED if shifted else PAC_CONFIG
    for key in cfg:
        if key.lower() in model_size.lower():
            return cfg[key]
    return {"ratio_far": _DEFAULT_K1, "ratio_local": _DEFAULT_K2}

# ────────────────────────────── factory ───────────────────────────────────

def build_method(
    max_tokens: int = 512,
    batch_size: int = 1,
    n_aug: int = 5,
    alpha: float = 0.3,
    ratio_local: float = 0.0,
    ratio_far: float = 0.0,
    seed: int = 0,
    **kwargs,
) -> BaseMethod:
    """
    If ratio_local / ratio_far are not explicitly set (left at 0.0),
    they are auto-resolved from PAC_CONFIG / PAC_CONFIG_SHIFTED at
    runtime inside PACMethod.run() based on model size and dataset path.
    """
    return PACMethod(
        max_tokens=max_tokens,
        batch_size=batch_size,
        n_aug=n_aug,
        alpha=alpha,
        ratio_local=ratio_local,
        ratio_far=ratio_far,
        seed=seed,
    )