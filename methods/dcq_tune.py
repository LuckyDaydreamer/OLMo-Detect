"""
Tune DCQ's score_type on the dev split.
After tuning, paste the printed line into MATCHED_CONFIG / SHIFTED_CONFIG in dcq_method.py.
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

from dcq_method import (
    compute_dcq_records,
    compute_partition_stats,
    non_preferred_letters,
    score_from_record,
    VALID_SCORE_TYPES,
)


# Score types that depend on partition-level BDQ statistics (and hence
# require BDQ to have been run).
_BDQ_DEPENDENT_SCORE_TYPES = {"pcorrect_kappa", "pcorrect_kappa_np"}


def _scores_for(
    records,
    score_type: str,
    partition_stats: dict[str, float] | None = None,
) -> list[float]:
    return [
        score_from_record(r, score_type, partition_stats=partition_stats)
        for r in records
    ]


def _auroc(contam_scores: list[float], uncontam_scores: list[float]) -> float:
    s = np.array(contam_scores + uncontam_scores, dtype=np.float64)
    y = np.array([1] * len(contam_scores) + [0] * len(uncontam_scores))
    valid = np.isfinite(s)
    if valid.sum() < 2 or y[valid].sum() == 0 or (1 - y[valid]).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y[valid], s[valid]))


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _maybe_subsample(dataset, max_records: int | None):
    if not max_records or len(dataset.records) <= max_records:
        return dataset
    dataset.records = dataset.records[:max_records]
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune DCQ on labelled dev data.")
    parser.add_argument("--model",         type=str, required=True,
                        help="Path to OLMo model directory (test-taker).")
    parser.add_argument("--contam_path",   type=str, required=True)
    parser.add_argument("--uncontam_path", type=str, required=True)
    parser.add_argument("--max_tokens",    type=int, default=4096,
                        help="Max prompt context length for OLMo forward pass.")
    parser.add_argument("--batch_size",    type=int, default=1,
                        help="OLMo forward-pass batch size. Increase for speed.")
    parser.add_argument("--max_dev_records", type=int, default=None,
                        help="Cap dev records per side. Saves Phase-1 API cost.")

    # GPT-4 Phase-1 settings
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini-2024-07-18",
                        help="Phase 1 perturbation generator. Paper default is "
                             "gpt-4-0613 (deprecated by OpenAI in 2024); we substitute "
                             "gpt-4o-mini-2024-07-18 as a current, supported, "
                             "lower-cost alternative.")
    parser.add_argument("--cache_dir",   type=str, default=".dcq_cache",
                        help="Disk cache for GPT-4 Phase-1 results. MUST be on "
                             "persistent storage so re-runs across OLMo sizes are free.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Phase-1 parallelism (OpenAI API workers).")

    # Quiz structure
    parser.add_argument("--placements", type=str, default="ABCD",
                        help="Which BCQ placements to score. Subset of ABCD.")
    parser.add_argument("--no_bdq", action="store_true",
                        help="Skip the BDQ baseline forward pass. Disables "
                             "the kappa-based score types (pcorrect_kappa, "
                             "pcorrect_kappa_np).")

    parser.add_argument("--score_types", type=str,
                        default=",".join(VALID_SCORE_TYPES))
    parser.add_argument("--source_hint", type=str, default=None)
    parser.add_argument("--save_json",   type=str, default=None)
    args = parser.parse_args()

    score_types = _parse_str_list(args.score_types)
    bad = [s for s in score_types if s not in VALID_SCORE_TYPES]
    if bad:
        parser.error(f"Unknown score_types: {bad}. Choose from {VALID_SCORE_TYPES}.")

    placements = tuple(L.upper() for L in args.placements if L.upper() in "ABCD")
    if not placements:
        parser.error("--placements must be a subset of ABCD.")

    bdq_dependent = [s for s in score_types if s in _BDQ_DEPENDENT_SCORE_TYPES]
    include_bdq = (not args.no_bdq) or bool(bdq_dependent)
    if bdq_dependent and args.no_bdq:
        parser.error(
            f"Score type(s) {bdq_dependent} require BDQ; cannot combine with --no_bdq. "
            f"Drop --no_bdq or remove those score types from --score_types."
        )

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY required for DCQ Phase-1 perturbation generation.")

    # ── 1. Load model + datasets ──────────────────────────────────────────────
    print(f"--> Loading model bundle: {args.model}")
    model_bundle = load_model_bundle(args.model)

    print(f"--> Loading datasets...")
    contam_dataset   = load_dataset(args.contam_path)
    uncontam_dataset = load_dataset(args.uncontam_path)
    print(f"  contam   dev: {len(contam_dataset.records)} records  ({contam_dataset.name})")
    print(f"  uncontam dev: {len(uncontam_dataset.records)} records  ({uncontam_dataset.name})")
    if args.max_dev_records:
        contam_dataset   = _maybe_subsample(contam_dataset,   args.max_dev_records)
        uncontam_dataset = _maybe_subsample(uncontam_dataset, args.max_dev_records)
        print(f"  capped at: {args.max_dev_records} per side "
              f"(contam={len(contam_dataset.records)}, uncontam={len(uncontam_dataset.records)})")

    print(f"  judge (Phase 1): {args.judge_model}")
    print(f"  cache_dir:       {args.cache_dir}")
    print(f"  placements:      {placements}")
    print(f"  include_bdq:     {include_bdq}")
    print(f"  score_types:     {score_types}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  max_tokens:      {args.max_tokens}")

    # ── 2. Run DCQ end-to-end on both sides (cached) ─────────────────────────
    print(f"\n=== DCQ on contam dev ===")
    contam_records = compute_dcq_records(
        model_bundle = model_bundle,
        dataset      = contam_dataset,
        judge_model  = args.judge_model,
        cache_dir    = args.cache_dir,
        num_workers  = args.num_workers,
        placements   = placements,
        include_bdq  = include_bdq,
        batch_size   = args.batch_size,
        max_tokens   = args.max_tokens,
        source_hint  = args.source_hint,
    )
    print(f"  -> scored {len(contam_records)} records")

    print(f"\n=== DCQ on uncontam dev ===")
    uncontam_records = compute_dcq_records(
        model_bundle = model_bundle,
        dataset      = uncontam_dataset,
        judge_model  = args.judge_model,
        cache_dir    = args.cache_dir,
        num_workers  = args.num_workers,
        placements   = placements,
        include_bdq  = include_bdq,
        batch_size   = args.batch_size,
        max_tokens   = args.max_tokens,
        source_hint  = args.source_hint,
    )
    print(f"  -> scored {len(uncontam_records)} records")

    # ── 3. Compute partition-level BDQ statistics ─────────────────────────────
    # Each partition has its own BDQ profile, so each gets its own p_e.
    # Note the asymmetry consequence: pcorrect_kappa / pcorrect_kappa_np
    # use a partition-specific p_e, meaning contam and uncontam samples
    # are scored under slightly different normalisations. This is the
    # natural per-partition adaptation of the paper's algorithm.
    contam_stats   = compute_partition_stats(contam_records)   if include_bdq else None
    uncontam_stats = compute_partition_stats(uncontam_records) if include_bdq else None

    if include_bdq:
        print(f"\n  contam   p_e (mean P(L|BDQ)): "
              f"{ {L: round(contam_stats[L], 3) for L in 'ABCD'} }")
        print(f"  uncontam p_e (mean P(L|BDQ)): "
              f"{ {L: round(uncontam_stats[L], 3) for L in 'ABCD'} }")
        print(f"  contam   non-preferred:       {non_preferred_letters(contam_stats)}")
        print(f"  uncontam non-preferred:       {non_preferred_letters(uncontam_stats)}")

    # ── 4. Sweep score_type (pure-CPU on cached probs) ───────────────────────
    print(f"\n  {'score_type':>22}  {'AUROC':>8}")
    print("  " + "-" * 34)
    best = {"auc": -1.0, "score_type": None}
    all_results: list[dict] = []
    for score_type in score_types:
        cs = _scores_for(contam_records,   score_type, partition_stats=contam_stats)
        us = _scores_for(uncontam_records, score_type, partition_stats=uncontam_stats)
        auc = _auroc(cs, us)
        all_results.append({"score_type": score_type, "auc": auc})
        if not np.isfinite(auc):
            print(f"  {score_type:>22}  {'N/A':>8}")
            continue
        star = ""
        if auc > best["auc"]:
            best = {"auc": auc, "score_type": score_type}
            star = " <-- best"
        print(f"  {score_type:>22}  {auc:>8.4f}{star}")

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL  : {os.path.basename(args.model)}")
    print(f"DATASET: {os.path.basename(args.contam_path)} vs "
          f"{os.path.basename(args.uncontam_path)}")
    print(f"Best   : score_type={best['score_type']}")
    print(f"AUROC  : {best['auc']:.4f}")
    print(f"{'=' * 60}")

    name     = os.path.basename(args.model.rstrip("/"))
    cfg_dict = "SHIFTED_CONFIG" if "shifted" in args.contam_path.lower() else "MATCHED_CONFIG"
    print(f"\n--> Update {cfg_dict} in dcq_method.py:")
    print(f'    "{name}": {{"score_type": "{best["score_type"]}"}},')

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model":          args.model,
                    "contam_path":    args.contam_path,
                    "uncontam_path":  args.uncontam_path,
                    "judge_model":    args.judge_model,
                    "placements":     list(placements),
                    "include_bdq":    include_bdq,
                    "contam_pe":      contam_stats,
                    "uncontam_pe":    uncontam_stats,
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