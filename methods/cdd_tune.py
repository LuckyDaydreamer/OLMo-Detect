"""
Tune CDD's hyper-parameters (alpha, length_cap, prompt_ratio) on the dev split. 
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "methods" else _THIS_DIR
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from data_utils import load_dataset
from model_utils import load_model_bundle

from cdd_method import (
    compute_cdd_samples,
    score_record,
)


# ─── default search grids ─────────────────────────────────────────────────────
DEFAULT_ALPHA_GRID    = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
DEFAULT_LENGTH_CAPS   = [50, 100, 200]
DEFAULT_PROMPT_RATIOS = [0.25, 0.5, 0.75]


def _peaks_for(samples_payload: dict, alpha: float, length_cap: int) -> list[float]:
    """Pure-CPU sweep over cached samples — no GPU calls."""
    peaks: list[float] = []
    for item in samples_payload["samples"]:
        peak, _, _ = score_record(
            greedy_completion_ids = item["greedy_completion_ids"],
            sample_completion_ids = item["sample_completion_ids"],
            alpha                 = alpha,
            length_cap            = length_cap,
        )
        peaks.append(peak)
    return peaks


def _auroc(contam_peaks: list[float], uncontam_peaks: list[float]) -> float:
    scores = np.array(contam_peaks + uncontam_peaks, dtype=np.float64)
    labels = np.array([1] * len(contam_peaks) + [0] * len(uncontam_peaks))
    valid = np.isfinite(scores)
    if valid.sum() < 2 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels[valid], scores[valid]))


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CDD on labelled dev data.")
    parser.add_argument("--model",         type=str, required=True,
                        help="Path to model directory (e.g. olmo_models/OLMo2-7B-Instruct).")
    parser.add_argument("--contam_path",   type=str, required=True,
                        help="Contaminated dev JSONL.")
    parser.add_argument("--uncontam_path", type=str, required=True,
                        help="Uncontaminated dev JSONL.")
    parser.add_argument("--max_tokens",    type=int, default=4096,
                        help="Max model context length (prompt + generation). Default 4096 for OLMo2.")
    parser.add_argument("--n_samples",     type=int, default=50,
                        help="Number of sampled completions per prompt (paper: 50).")
    parser.add_argument("--temperature",   type=float, default=0.8,
                        help="Sampling temperature (paper: 0.8).")
    parser.add_argument("--sub_batch_size", type=int, default=None,
                        help="Chunk size for `num_return_sequences`. Set if you OOM (e.g. 10 for 32B).")
    parser.add_argument("--alpha_grid",    type=str,
                        default=",".join(str(x) for x in DEFAULT_ALPHA_GRID))
    parser.add_argument("--length_caps",   type=str,
                        default=",".join(str(x) for x in DEFAULT_LENGTH_CAPS))
    parser.add_argument("--prompt_ratios", type=str,
                        default=",".join(str(x) for x in DEFAULT_PROMPT_RATIOS))
    parser.add_argument("--max_dev_per_side", type=int, default=0,
                        help="If >0, truncate contam/uncontam dev to this many records "
                             "per side. AUROC ranks across (alpha, length_cap, prompt_ratio) "
                             "are stable at ~100; this is only used for tuning, not the "
                             "production run.")
    parser.add_argument("--save_json",     type=str, default=None,
                        help="Optional path to save the full sweep table as JSON.")
    args = parser.parse_args()

    alpha_grid    = _parse_floats(args.alpha_grid)
    length_caps   = _parse_ints(args.length_caps)
    prompt_ratios = _parse_floats(args.prompt_ratios)

    # Generate enough tokens for the largest length_cap; smaller ones just truncate.
    max_new_tokens = max(length_caps)

    # ── 1. Load model + datasets ──────────────────────────────────────────────
    print(f"--> Loading model bundle: {args.model}")
    model_bundle = load_model_bundle(args.model)

    print(f"--> Loading datasets...")
    contam_dataset   = load_dataset(args.contam_path)
    uncontam_dataset = load_dataset(args.uncontam_path)
    if args.max_dev_per_side > 0:
        contam_dataset.records   = contam_dataset.records[:args.max_dev_per_side]
        uncontam_dataset.records = uncontam_dataset.records[:args.max_dev_per_side]
        print(f"  truncated dev to first {args.max_dev_per_side} records per side")
    print(f"  contam   dev: {len(contam_dataset.records)} records  ({contam_dataset.name})")
    print(f"  uncontam dev: {len(uncontam_dataset.records)} records  ({uncontam_dataset.name})")
    print(f"  generation: max_new_tokens={max_new_tokens} (= max of length_caps grid)")

    # ── 2. Sweep ──────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    best = {"auc": -1.0, "alpha": None, "length_cap": None, "prompt_ratio": None}

    for prompt_ratio in prompt_ratios:
        print(f"\n=== Sampling at prompt_ratio={prompt_ratio} "
              f"(this is the expensive step) ===")
        contam_payload = compute_cdd_samples(
            model_bundle    = model_bundle,
            dataset         = contam_dataset,
            n_samples       = args.n_samples,
            temperature     = args.temperature,
            max_new_tokens  = max_new_tokens,
            prompt_ratio    = prompt_ratio,
            max_tokens      = args.max_tokens,
            sub_batch_size  = args.sub_batch_size,
        )
        uncontam_payload = compute_cdd_samples(
            model_bundle    = model_bundle,
            dataset         = uncontam_dataset,
            n_samples       = args.n_samples,
            temperature     = args.temperature,
            max_new_tokens  = max_new_tokens,
            prompt_ratio    = prompt_ratio,
            max_tokens      = args.max_tokens,
            sub_batch_size  = args.sub_batch_size,
        )

        print(f"\n  prompt_ratio={prompt_ratio}")
        print(f"  {'alpha':>8}  {'length_cap':>10}  {'AUROC':>8}")
        print("  " + "-" * 32)

        for length_cap in length_caps:
            for alpha in alpha_grid:
                contam_peaks   = _peaks_for(contam_payload,   alpha, length_cap)
                uncontam_peaks = _peaks_for(uncontam_payload, alpha, length_cap)
                auc = _auroc(contam_peaks, uncontam_peaks)
                row = {
                    "prompt_ratio": prompt_ratio,
                    "length_cap":   length_cap,
                    "alpha":        alpha,
                    "auc":          auc,
                }
                all_results.append(row)
                if not np.isfinite(auc):
                    print(f"  {alpha:>8.4f}  {length_cap:>10d}  {'N/A':>8}")
                    continue
                star = ""
                if auc > best["auc"]:
                    best = {"auc": auc, "alpha": alpha,
                            "length_cap": length_cap, "prompt_ratio": prompt_ratio}
                    star = " <-- best"
                print(f"  {alpha:>8.4f}  {length_cap:>10d}  {auc:>8.4f}{star}")

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL  : {os.path.basename(args.model)}")
    print(f"DATASET: {os.path.basename(args.contam_path)} vs {os.path.basename(args.uncontam_path)}")
    print(f"Best   : alpha={best['alpha']}  length_cap={best['length_cap']}  "
          f"prompt_ratio={best['prompt_ratio']}")
    print(f"AUROC  : {best['auc']:.4f}")
    print(f"{'=' * 60}")

    # Suggest the line to paste into cdd_method.py.
    name     = os.path.basename(args.model.rstrip("/"))
    cfg_dict = "SHIFTED_CONFIG" if "shifted" in args.contam_path.lower() else "MATCHED_CONFIG"
    print(f"\n--> Update {cfg_dict} in cdd_method.py:")
    print(
        f'    "{name}": '
        f'{{"alpha": {best["alpha"]}, '
        f'"length_cap": {best["length_cap"]}, '
        f'"prompt_ratio": {best["prompt_ratio"]}}},'
    )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model":          args.model,
                    "contam_path":    args.contam_path,
                    "uncontam_path":  args.uncontam_path,
                    "n_samples":      args.n_samples,
                    "temperature":    args.temperature,
                    "max_tokens":     args.max_tokens,
                    "max_new_tokens": max_new_tokens,
                    "best":           best,
                    "all":            all_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved sweep table → {args.save_json}")


if __name__ == "__main__":
    main()