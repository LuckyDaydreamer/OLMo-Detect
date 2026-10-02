"""
Tune Neighborhood Attack's score_type by on the dev split. 
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

from neighborhood_attack_method import (
    compute_neighbour_records,
    score_from_record,
    VALID_SCORE_TYPES,
)


def _scores_for(records, score_type: str) -> list[float]:
    return [score_from_record(r, score_type) for r in records]


def _auroc(contam_scores: list[float], uncontam_scores: list[float]) -> float:
    s = np.array(contam_scores + uncontam_scores, dtype=np.float64)
    y = np.array([1] * len(contam_scores) + [0] * len(uncontam_scores))
    valid = np.isfinite(s)
    if valid.sum() < 2 or y[valid].sum() == 0 or (1 - y[valid]).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y[valid], s[valid]))


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Neighborhood Attack on labelled dev data.")
    parser.add_argument("--model",         type=str, required=True,
                        help="Path to OLMo model directory (target/test-taker).")
    parser.add_argument("--contam_path",   type=str, required=True)
    parser.add_argument("--uncontam_path", type=str, required=True)
    parser.add_argument("--max_tokens",    type=int, default=4096)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--seed",          type=int, default=0)

    # Neighbour-generation knobs (paper defaults — we don't tune these per model)
    parser.add_argument("--n_perturbations", type=int, default=10,
                        help="Number of neighbours per record (paper uses 100 for "
                             "BERT version, repo defaults to 10/25 for T5 version).")
    parser.add_argument("--pct_words_masked", type=float, default=0.30)
    parser.add_argument("--span_length",      type=int, default=2)
    parser.add_argument("--buffer_size",      type=int, default=1)
    parser.add_argument("--mask_top_p",       type=float, default=1.0)
    parser.add_argument("--mask_filling_model_name", type=str, default="t5-large",
                        help="HF model ID. Paper uses bert-base; the public repo "
                             "implementation (which we follow) uses T5. "
                             "t5-large = 770M, t5-3b = 3B.")
    parser.add_argument("--chunk_size",       type=int, default=20,
                        help="T5 generation batch size.")
    parser.add_argument("--ceil_pct",         action="store_true")
    parser.add_argument("--t5_dtype",         type=str, default="float16",
                        choices=["float16", "float32", "bfloat16", "fp16", "fp32", "bf16"])

    parser.add_argument("--cache_dir", type=str, default=".neighborhood_cache",
                        help="Disk cache for T5 neighbours. MUST be on persistent "
                             "storage — neighbour generation is expensive and shared "
                             "across OLMo sizes evaluating the same dataset.")

    parser.add_argument("--score_types", type=str, default=",".join(VALID_SCORE_TYPES))
    parser.add_argument("--save_json",   type=str, default=None)
    args = parser.parse_args()

    score_types = _parse_str_list(args.score_types)
    bad = [s for s in score_types if s not in VALID_SCORE_TYPES]
    if bad:
        parser.error(f"Unknown score_types: {bad}. Choose from {VALID_SCORE_TYPES}.")

    import torch
    t5_dtype = {
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }[args.t5_dtype]

    # ── 1. Load model + datasets ─────────────────────────────────────────────
    print(f"--> Loading model bundle: {args.model}")
    model_bundle = load_model_bundle(args.model)

    print(f"--> Loading datasets...")
    contam_dataset   = load_dataset(args.contam_path)
    uncontam_dataset = load_dataset(args.uncontam_path)
    print(f"  contam   dev: {len(contam_dataset.records)} records  ({contam_dataset.name})")
    print(f"  uncontam dev: {len(uncontam_dataset.records)} records  ({uncontam_dataset.name})")
    print(f"  n_perturbations: {args.n_perturbations}")
    print(f"  mask model:      {args.mask_filling_model_name} ({args.t5_dtype})")
    print(f"  cache_dir:       {args.cache_dir}")
    print(f"  score_types:     {score_types}")

    common_kwargs = dict(
        n_perturbations         = args.n_perturbations,
        pct_words_masked        = args.pct_words_masked,
        span_length             = args.span_length,
        buffer_size             = args.buffer_size,
        mask_top_p              = args.mask_top_p,
        mask_filling_model_name = args.mask_filling_model_name,
        chunk_size              = args.chunk_size,
        ceil_pct                = args.ceil_pct,
        cache_dir               = args.cache_dir,
        max_tokens              = args.max_tokens,
        batch_size              = args.batch_size,
        seed                    = args.seed,
        t5_dtype                = t5_dtype,
    )

    # ── 2. Compute records on both sides (cached on runtime_cache) ───────────
    print(f"\n=== Computing neighbours + LLs on contam dev ===")
    contam_records = compute_neighbour_records(
        model_bundle=model_bundle, dataset=contam_dataset, **common_kwargs,
    )
    print(f"  -> scored {len(contam_records)} records")

    print(f"\n=== Computing neighbours + LLs on uncontam dev ===")
    uncontam_records = compute_neighbour_records(
        model_bundle=model_bundle, dataset=uncontam_dataset, **common_kwargs,
    )
    print(f"  -> scored {len(uncontam_records)} records")

    # ── 3. Sweep score_type (pure-CPU on cached LLs) ─────────────────────────
    print(f"\n  {'score_type':>14}  {'AUROC':>8}")
    print("  " + "-" * 26)
    best = {"auc": -1.0, "score_type": None}
    all_results: list[dict] = []
    for score_type in score_types:
        cs = _scores_for(contam_records,   score_type)
        us = _scores_for(uncontam_records, score_type)
        auc = _auroc(cs, us)
        all_results.append({"score_type": score_type, "auc": auc})
        if not np.isfinite(auc):
            print(f"  {score_type:>14}  {'N/A':>8}")
            continue
        star = ""
        if auc > best["auc"]:
            best = {"auc": auc, "score_type": score_type}
            star = " <-- best"
        print(f"  {score_type:>14}  {auc:>8.4f}{star}")

    # ── 4. Report ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL  : {os.path.basename(args.model)}")
    print(f"DATASET: {os.path.basename(args.contam_path)} vs "
          f"{os.path.basename(args.uncontam_path)}")
    print(f"Best   : score_type={best['score_type']}")
    print(f"AUROC  : {best['auc']:.4f}")
    print(f"{'=' * 60}")

    name     = os.path.basename(args.model.rstrip("/"))
    cfg_dict = "SHIFTED_CONFIG" if "shifted" in args.contam_path.lower() else "MATCHED_CONFIG"
    print(f"\n--> Update {cfg_dict} in neighborhood_attack_method.py:")
    print(f'    "{name}": {{"score_type": "{best["score_type"]}"}},')

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model":          args.model,
                    "contam_path":    args.contam_path,
                    "uncontam_path":  args.uncontam_path,
                    "n_perturbations": args.n_perturbations,
                    "mask_filling_model_name": args.mask_filling_model_name,
                    "score_types":    score_types,
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