"""
Tune the alpha hyperparameter of DCPDD on the dev set.
"""

import argparse
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

from model_utils import load_model_bundle
from dcpdd_method import (
    DCPDDMethod,
    ALPHA_SEARCH_GRID,
    score_dataset_for_tuning,
    compute_dc_pdd_scores_from_cache,
)


def run_tuning(
    model_path:    str,
    contam_path:   str,
    uncontam_path: str,
    c4_ref_json:   str,
    batch_size:    int         = 3,
    max_tokens:    int         = 1024,
    max_cha:       int         = 512,
    lang:          str         = "en",
    alpha_grid:    list[float] = ALPHA_SEARCH_GRID,
) -> float:

    # 1. Load model
    print(f"--> Loading model: {model_path}")
    model_bundle = load_model_bundle(model_path)

    # 2. Load reference frequency + build smoothed array
    print(f"--> Loading C4 reference: {c4_ref_json}")
    with open(c4_ref_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    token_frequency = payload["token_frequency"]
    vocab_size      = int(payload["vocab_size"])
    fre_dis_smo     = DCPDDMethod._build_smoothed_freq_array(token_frequency, vocab_size)

    # 3. Run model ONCE per split
    print(f"--> Scoring contam: {os.path.basename(contam_path)}")
    m_cache = score_dataset_for_tuning(
        model_bundle, contam_path, fre_dis_smo,
        max_cha=max_cha, lang=lang, max_tokens=max_tokens, batch_size=batch_size,
    )

    print(f"--> Scoring uncontam: {os.path.basename(uncontam_path)}")
    nm_cache = score_dataset_for_tuning(
        model_bundle, uncontam_path, fre_dis_smo,
        max_cha=max_cha, lang=lang, max_tokens=max_tokens, batch_size=batch_size,
    )

    # 4. Sweep alpha — pure numpy, zero GPU calls
    def auroc(m_scores, nm_scores):
        scores = np.array(m_scores + nm_scores, dtype=np.float64)
        labels = np.array([1] * len(m_scores) + [0] * len(nm_scores))
        valid  = np.isfinite(scores)
        if valid.sum() < 2 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
            return float("nan")
        return float(roc_auc_score(labels[valid], scores[valid]))

    results = []
    print(f"\n{'alpha':>8}  {'AUROC':>8}")
    print("-" * 20)
    for alpha in alpha_grid:
        m_scores  = compute_dc_pdd_scores_from_cache(m_cache,  fre_dis_smo, alpha)
        nm_scores = compute_dc_pdd_scores_from_cache(nm_cache, fre_dis_smo, alpha)
        auc = auroc(m_scores, nm_scores)
        results.append({"alpha": alpha, "auc": auc})
        finite_aucs = [r["auc"] for r in results if np.isfinite(r["auc"])]
        star = " <-- best" if finite_aucs and np.isfinite(auc) and auc == max(finite_aucs) else ""
        print(f"  {alpha:>8.4f}  {auc:>8.4f}{star}")

    finite = [r for r in results if np.isfinite(r["auc"])]
    if not finite:
        print("No valid AUROC found.")
        return ALPHA_SEARCH_GRID[0]

    best       = max(finite, key=lambda r: r["auc"])
    best_alpha = best["alpha"]

    print(f"\n{'='*50}")
    print(f"MODEL : {os.path.basename(model_path)}")
    print(f"Best alpha = {best_alpha}   (AUROC {best['auc']:.4f})")
    print(f"{'='*50}")
    print(f"\n--> Update MATCHED_CONFIG or SHIFTED_CONFIG in dcpdd_method.py:")
    print(f"    \"{os.path.basename(model_path)}\": {{\"alpha\": {best_alpha}}},")

    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, required=True)
    parser.add_argument("--contam_path",   type=str, required=True,
                        help="Contaminated (member) dev JSONL")
    parser.add_argument("--uncontam_path", type=str, required=True,
                        help="Uncontaminated (non-member) dev JSONL")
    parser.add_argument("--c4_ref_json",   type=str,
                            default=str(Path(__file__).resolve().parent / "dcpdd_data" / "c4_token_occurrence_OLMo2-Instruct.json"))
    parser.add_argument("--batch_size",    type=int, default=3)
    parser.add_argument("--max_tokens",    type=int, default=1024)
    parser.add_argument("--max_cha",       type=int, default=512)
    parser.add_argument("--lang",          type=str, default="en")
    args = parser.parse_args()

    run_tuning(
        model_path    = args.model,
        contam_path   = args.contam_path,
        uncontam_path = args.uncontam_path,
        c4_ref_json   = args.c4_ref_json,
        batch_size    = args.batch_size,
        max_tokens    = args.max_tokens,
        max_cha       = args.max_cha,
        lang          = args.lang,
    )