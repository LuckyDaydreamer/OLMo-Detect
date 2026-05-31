"""
Tune n (number of non-member prefix shots) for RECALL on the dev split.
"""

import argparse
import json
import math
import numpy as np
import os
import torch
from sklearn.metrics import roc_auc_score

from data_utils import load_dataset, extract_text
from model_utils import load_model_bundle
from recall_method import build_prefix, _batch_conditional_ll, _safe_mean

N_SEARCH_GRID = list(range(1, 13))   # 1..12, matching paper


def run_tuning(
    model_path:    str,
    contam_path:   str,
    uncontam_path: str,
    max_shots:     int   = 12,
    max_tokens:    int   = 1024,
    batch_size:    int   = 1,
    max_model_len: int   = 4096,
) -> int:

    # 1. Load model
    print(f"--> Loading model: {model_path}")
    model_bundle  = load_model_bundle(model_path)
    tokenizer     = model_bundle.tokenizer
    model         = model_bundle.model
    device        = next(model.parameters()).device

    # 2. Load texts
    def load_texts(path: str) -> list[str]:
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = extract_text(json.loads(line))
                if t:
                    texts.append(t)
        return texts

    contam_texts   = load_texts(contam_path)
    uncontam_texts = load_texts(uncontam_path)
    print(f"  contam  dev: {len(contam_texts)} texts")
    print(f"  uncontam dev: {len(uncontam_texts)} texts")

    # We use uncontam_texts[:n] as prefix and uncontam_texts[n:] as non-member eval
    # so we need at least 1 non-member left for eval after taking prefix
    max_shots = min(max_shots, len(uncontam_texts) - 1)
    if max_shots < 1:
        print("ERROR: not enough uncontaminated texts for tuning.")
        return 1

    # 3. Compute avg target length (over contam + uncontam for budget estimate)
    all_texts = contam_texts + uncontam_texts
    avg_target_len = int(np.mean([
        len(tokenizer.encode(t, add_special_tokens=True)[:max_tokens])
        for t in all_texts
    ]))

    # 4. Phase 1 — unconditional LLs for all texts (ONCE)
    print(f"--> Computing unconditional LLs ({len(all_texts)} samples)...")
    uncon_stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
        texts         = all_texts,
        batch_size    = batch_size,
        max_tokens    = max_tokens,
        progress_desc = "recall uncond",
    )
    ll_uncond = []
    for s in uncon_stats:
        tlp = s.get("token_log_probs", [])
        ll_uncond.append(float(np.mean(tlp)) if tlp else float("nan"))

    ll_uncond_contam   = ll_uncond[:len(contam_texts)]
    ll_uncond_uncontam = ll_uncond[len(contam_texts):]

    # 5. Sweep n — one batched conditional pass per n
    def auroc(m_scores: list[float], nm_scores: list[float]) -> float:
        scores = np.array(m_scores + nm_scores, dtype=np.float64)
        labels = np.array([1]*len(m_scores) + [0]*len(nm_scores))
        valid  = np.isfinite(scores)
        if valid.sum() < 2 or labels[valid].sum() == 0 or (1-labels[valid]).sum() == 0:
            return float("nan")
        return float(roc_auc_score(labels[valid], scores[valid]))

    def compute_recall_scores(
        ll_u_list: list[float],
        ll_c_list: list[float],
    ) -> list[float]:
        scores = []
        for ll_u, ll_c in zip(ll_u_list, ll_c_list):
            if math.isfinite(ll_u) and math.isfinite(ll_c) and ll_u != 0.0:
                scores.append(ll_c / ll_u)
            else:
                scores.append(float("nan"))
        return scores

    results = []
    print(f"\n{'n':>4}  {'AUROC':>8}  {'prefix_tokens':>14}  {'shots_used':>12}")
    print("-" * 46)

    for n in N_SEARCH_GRID:
        if n > max_shots:
            break

        # Build prefix from uncontam_texts[:n]
        prefix_ids, shots_used = build_prefix(
            prefix_texts   = uncontam_texts,
            n              = n,
            tokenizer      = tokenizer,
            avg_target_len = avg_target_len,
            max_model_len  = max_model_len,
            max_tokens     = max_tokens,
        )

        if shots_used == 0:
            print(f"  {n:>2}   {'N/A':>8}  (prefix too long for context)")
            continue

        # Non-member eval texts: uncontam_texts[n:] (held out from prefix)
        nm_eval_texts = uncontam_texts[n:]
        if not nm_eval_texts:
            print(f"  {n:>2}   {'N/A':>8}  (no held-out non-members)")
            continue

        # Conditional LLs for contam + nm_eval
        eval_texts = contam_texts + nm_eval_texts
        ll_c_all   = _batch_conditional_ll(
            model        = model,
            tokenizer    = tokenizer,
            device       = device,
            prefix_ids   = prefix_ids,
            target_texts = eval_texts,
            max_tokens   = max_tokens,
            batch_size   = batch_size,
        )
        ll_c_contam = ll_c_all[:len(contam_texts)]
        ll_c_nm     = ll_c_all[len(contam_texts):]

        # RECALL scores
        recall_contam = compute_recall_scores(ll_uncond_contam,             ll_c_contam)
        recall_nm     = compute_recall_scores(ll_uncond_uncontam[n:],       ll_c_nm)

        auc = auroc(recall_contam, recall_nm)
        results.append({"n": n, "auc": auc, "shots_used": shots_used, "prefix_len": len(prefix_ids)})

        finite_aucs = [r["auc"] for r in results if np.isfinite(r["auc"])]
        star = " <-- best" if finite_aucs and np.isfinite(auc) and auc == max(finite_aucs) else ""
        print(f"  {n:>2}   {auc:>8.4f}  {len(prefix_ids):>14}  {shots_used:>12}{star}")

    finite = [r for r in results if np.isfinite(r["auc"])]
    if not finite:
        print("No valid AUROC found. Defaulting to n=5.")
        return 5

    best   = max(finite, key=lambda r: r["auc"])
    best_n = best["n"]

    print(f"\n{'='*55}")
    print(f"MODEL : {os.path.basename(model_path)}")
    print(f"Best n = {best_n}   (AUROC {best['auc']:.4f})")
    print(f"{'='*55}")
    print(f"\n--> Update MATCHED_CONFIG or SHIFTED_CONFIG in recall_method.py:")
    print(f'    "{os.path.basename(model_path)}": {{"n": {best_n}}},')

    return best_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, required=True)
    parser.add_argument("--contam_path",   type=str, required=True,
                        help="Contaminated (member) dev JSONL")
    parser.add_argument("--uncontam_path", type=str, required=True,
                        help="Uncontaminated (non-member) dev JSONL — "
                             "first n texts used as prefix, rest as eval")
    parser.add_argument("--max_shots",     type=int, default=12)
    parser.add_argument("--max_tokens",    type=int, default=1024)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    run_tuning(
        model_path    = args.model,
        contam_path   = args.contam_path,
        uncontam_path = args.uncontam_path,
        max_shots     = args.max_shots,
        max_tokens    = args.max_tokens,
        batch_size    = args.batch_size,
        max_model_len = args.max_model_len,
    )