"""
Tune the k1 (ratio_far) and k2 (ratio_local) hyperparameters of PAC on the dev split.
"""

import argparse
import json
import os
import random
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from data_utils import extract_text
from model_utils import load_model_bundle
from pac_method import _eda, _polarized_distance


# ─────────────────────── default grids ────────────────────────────────────

K1_GRID = [0.05, 0.10, 0.20]           # ratio_far  (top-k1% most probable)
K2_GRID = [0.10, 0.20, 0.30, 0.40]     # ratio_local (bottom-k2% least probable)

# Fixed (not tuned — paper ablation shows these are stable)
N_AUG = 5
ALPHA = 0.3
SEED  = 0


# ─────────────────────── helpers ──────────────────────────────────────────

def load_texts(path: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = extract_text(json.loads(line))
            if text:
                texts.append(text)
    return texts


def get_logprobs(
    model_bundle, texts: list[str], aug_texts: list[str],
    batch_size: int, max_tokens: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Run model ONCE on all texts, return (orig_lps, aug_lps)."""
    all_texts = texts + aug_texts
    print(f"    Forward passes: {len(texts)} orig + {len(aug_texts)} aug "
          f"= {len(all_texts)} total")
    stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
        texts=all_texts,
        batch_size=batch_size,
        max_tokens=max_tokens,
        progress_desc="pac_tune",
    )
    all_lps = [s.get("token_log_probs", []) for s in stats]
    return all_lps[:len(texts)], all_lps[len(texts):]


def compute_pac_scores(
    orig_lps: list[list[float]],
    aug_lps: list[list[float]],
    n_aug: int,
    ratio_far: float,
    ratio_local: float,
) -> list[float]:
    """Pure numpy: compute -PAC scores from cached log-probs."""
    orig_pds = [_polarized_distance(lp, ratio_local, ratio_far) for lp in orig_lps]
    aug_pds = [_polarized_distance(lp, ratio_local, ratio_far) for lp in aug_lps]

    scores = []
    for i in range(len(orig_lps)):
        start = i * n_aug
        mean_aug = float(np.mean(aug_pds[start : start + n_aug]))
        pac_raw = orig_pds[i] - mean_aug
        scores.append(-pac_raw)  # negate: higher = more likely member
    return scores


def auroc(m_scores: list[float], nm_scores: list[float]) -> float:
    scores = np.array(m_scores + nm_scores, dtype=np.float64)
    labels = np.array([1] * len(m_scores) + [0] * len(nm_scores))
    valid = np.isfinite(scores)
    if valid.sum() < 2 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels[valid], scores[valid]))


# ─────────────────────── main ─────────────────────────────────────────────

def run_tuning(
    model_path: str,
    contam_path: str,
    uncontam_path: str,
    batch_size: int = 1,
    max_tokens: int = 512,
    k1_grid: list[float] = K1_GRID,
    k2_grid: list[float] = K2_GRID,
) -> dict[str, float]:

    # 1. Load model
    print(f"--> Loading model: {model_path}")
    model_bundle = load_model_bundle(model_path)

    # 2. Load texts
    con_texts = load_texts(contam_path)
    uncon_texts = load_texts(uncontam_path)
    print(f"--> Contam: {len(con_texts)}, Uncontam: {len(uncon_texts)}")

    # 3. Generate augmented texts (deterministic)
    random.seed(SEED)
    con_aug = []
    for t in con_texts:
        con_aug.extend(_eda(t, alpha=ALPHA, num_aug=N_AUG))
    random.seed(SEED + 1)  # different seed for uncontam to avoid correlation
    uncon_aug = []
    for t in uncon_texts:
        uncon_aug.extend(_eda(t, alpha=ALPHA, num_aug=N_AUG))

    # 4. Run model ONCE per split
    print(f"--> Scoring contam:")
    con_orig_lps, con_aug_lps = get_logprobs(
        model_bundle, con_texts, con_aug, batch_size, max_tokens,
    )
    print(f"--> Scoring uncontam:")
    uncon_orig_lps, uncon_aug_lps = get_logprobs(
        model_bundle, uncon_texts, uncon_aug, batch_size, max_tokens,
    )

    # 5. Sweep k1 × k2 — pure numpy
    results = []
    print(f"\n{'k1 (far)':>10}  {'k2 (local)':>10}  {'AUROC':>8}")
    print("-" * 35)

    for k1 in k1_grid:
        for k2 in k2_grid:
            m_scores = compute_pac_scores(con_orig_lps, con_aug_lps, N_AUG, k1, k2)
            nm_scores = compute_pac_scores(uncon_orig_lps, uncon_aug_lps, N_AUG, k1, k2)
            auc = auroc(m_scores, nm_scores)
            results.append({"k1": k1, "k2": k2, "auc": auc})
            finite_aucs = [r["auc"] for r in results if np.isfinite(r["auc"])]
            star = " <-- best" if finite_aucs and np.isfinite(auc) and auc == max(finite_aucs) else ""
            print(f"  {k1:>8.2f}  {k2:>10.2f}  {auc:>8.4f}{star}")

    finite = [r for r in results if np.isfinite(r["auc"])]
    if not finite:
        print("No valid AUROC found.")
        return {"ratio_far": 0.05, "ratio_local": 0.30}

    best = max(finite, key=lambda r: r["auc"])

    print(f"\n{'=' * 50}")
    print(f"MODEL : {os.path.basename(model_path)}")
    print(f"Best k1 (ratio_far) = {best['k1']}  "
          f"k2 (ratio_local) = {best['k2']}  "
          f"(AUROC {best['auc']:.4f})")
    print(f"{'=' * 50}")

    return {"ratio_far": best["k1"], "ratio_local": best["k2"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune PAC k1/k2 hyperparameters on dev data."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--contam_path", type=str, required=True,
                        help="Contaminated (member) dev JSONL")
    parser.add_argument("--uncontam_path", type=str, required=True,
                        help="Uncontaminated (non-member) dev JSONL")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    run_tuning(
        model_path=args.model,
        contam_path=args.contam_path,
        uncontam_path=args.uncontam_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )