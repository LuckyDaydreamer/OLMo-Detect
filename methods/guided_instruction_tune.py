"""
Tune Guided-Instruction's hyperparameters by on the dev split. 
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

# Make sibling imports resolvable whether this file lives at project root or in methods/.
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "methods" else _THIS_DIR
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from data_utils import load_dataset
from model_utils import load_model_bundle

from guided_instruction_method import (
    compute_completions,
    compute_icl_scores,
    score_record,
    VALID_SCORE_TYPES,
)


# ─── default search grids ────────────────────────────────────────────────────
DEFAULT_SCORE_TYPES = ["rouge_guided", "rouge_delta"]
DEFAULT_MIN_P_GRID  = [40]
DEFAULT_MAX_P_GRID  = [70]


def _scores_for(
    comp_payload: dict,
    score_type: str,
    icl_scores: dict[int, float] | None,
    use_bleurt: bool,
    bleurt_checkpoint: str | None,
) -> list[float]:
    """Pure-CPU sweep over cached completions + cached ICL judgments."""
    out: list[float] = []
    for item in comp_payload["samples"]:
        icl_val = None
        if icl_scores is not None:
            icl_val = icl_scores.get(item["record_index"])
        scored = score_record(
            item,
            score_type        = score_type,
            icl_score         = icl_val,
            use_bleurt        = use_bleurt,
            bleurt_checkpoint = bleurt_checkpoint,
        )
        out.append(scored["score"])
    return out


def _auroc(contam_scores: list[float], uncontam_scores: list[float]) -> float:
    s = np.array(contam_scores + uncontam_scores, dtype=np.float64)
    y = np.array([1] * len(contam_scores) + [0] * len(uncontam_scores))
    valid = np.isfinite(s)
    if valid.sum() < 2 or y[valid].sum() == 0 or (1 - y[valid]).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y[valid], s[valid]))


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Guided-Instruction on labelled dev data.")
    parser.add_argument("--model",         type=str, required=True)
    parser.add_argument("--contam_path",   type=str, required=True)
    parser.add_argument("--uncontam_path", type=str, required=True)
    parser.add_argument("--max_tokens",    type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--score_types",   type=str,
                        default=",".join(DEFAULT_SCORE_TYPES),
                        help=f"Comma-separated. Allowed: {sorted(VALID_SCORE_TYPES)}.")
    parser.add_argument("--min_p_grid",    type=str,
                        default=",".join(str(x) for x in DEFAULT_MIN_P_GRID))
    parser.add_argument("--max_p_grid",    type=str,
                        default=",".join(str(x) for x in DEFAULT_MAX_P_GRID))

    # BLEURT (optional)
    parser.add_argument("--use_bleurt",    action="store_true")
    parser.add_argument("--bleurt_checkpoint", type=str, default=None)

    # GPT-4 ICL judge
    parser.add_argument("--use_icl_judge", action="store_true",
                        help="Run the GPT-4 ICL judge pass. Auto-enabled if "
                             "`gpt4_icl` is in --score_types.")
    parser.add_argument("--judge_model",   type=str, default="gpt-4o-mini-2024-07-18",
                        help="Paper default is gpt-4-0613, which OpenAI deprecated in "
                             "2024. We substitute gpt-4o-mini-2024-07-18: a current, "
                             "supported model that handles the binary Yes/No "
                             "near-match judgment at much lower cost.")
    parser.add_argument("--judge_cache_dir", type=str, default=".openai_judge_cache",
                        help="Disk cache for judge results — re-runs are free.")
    parser.add_argument("--judge_workers", type=int, default=8,
                        help="Parallel API workers (ThreadPoolExecutor).")

    parser.add_argument("--source_hint",   type=str, default=None,
                        help="Override the source name passed to the guided "
                             "prompt. If unset, extracted from --contam_path.")
    parser.add_argument("--save_json",     type=str, default=None)
    args = parser.parse_args()

    score_types = _parse_str_list(args.score_types)
    bad = [s for s in score_types if s not in VALID_SCORE_TYPES]
    if bad:
        parser.error(f"Unknown score_types: {bad}. Choose from {sorted(VALID_SCORE_TYPES)}.")

    min_p_grid  = _parse_floats(args.min_p_grid)
    max_p_grid  = _parse_floats(args.max_p_grid)

    pp_pairs: list[tuple[float, float]] = []
    if len(min_p_grid) == len(max_p_grid):
        pp_pairs = list(zip(min_p_grid, max_p_grid))
    else:
        for mn in min_p_grid:
            for mx in max_p_grid:
                if mn <= mx:
                    pp_pairs.append((mn, mx))
    if not pp_pairs:
        raise ValueError("No valid (min_p, max_p) pairs in the grid.")

    needs_icl = ("gpt4_icl" in score_types) or args.use_icl_judge
    if needs_icl and "gpt4_icl" not in score_types:
        score_types = score_types + ["gpt4_icl"]
    if needs_icl and not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "score_type 'gpt4_icl' (or --use_icl_judge) requires OPENAI_API_KEY in env."
        )

    # ── 1. Load model + datasets ──────────────────────────────────────────────
    print(f"--> Loading model bundle: {args.model}")
    model_bundle = load_model_bundle(args.model)

    print(f"--> Loading datasets...")
    contam_dataset   = load_dataset(args.contam_path)
    uncontam_dataset = load_dataset(args.uncontam_path)
    print(f"  contam   dev: {len(contam_dataset.records)} records  ({contam_dataset.name})")
    print(f"  uncontam dev: {len(uncontam_dataset.records)} records  ({uncontam_dataset.name})")
    print(f"  generation: max_new_tokens={args.max_new_tokens}")
    print(f"  score_types: {score_types}")
    print(f"  (min_p, max_p) grid: {pp_pairs}")
    if needs_icl:
        print(f"  GPT-4 ICL judge: model={args.judge_model}, "
              f"cache_dir={args.judge_cache_dir}, workers={args.judge_workers}")

    # ── 2. Sweep ──────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    best = {"auc": -1.0, "score_type": None, "min_p": None, "max_p": None}

    for (min_p, max_p) in pp_pairs:
        print(f"\n=== Generating completions at min_p={min_p}, max_p={max_p} "
              f"(this is the expensive step) ===")
        contam_payload = compute_completions(
            model_bundle    = model_bundle,
            dataset         = contam_dataset,
            min_p           = min_p,
            max_p           = max_p,
            max_new_tokens  = args.max_new_tokens,
            max_tokens      = args.max_tokens,
            batch_size      = args.batch_size,
            seed            = args.seed,
            source_hint     = args.source_hint,
        )
        uncontam_payload = compute_completions(
            model_bundle    = model_bundle,
            dataset         = uncontam_dataset,
            min_p           = min_p,
            max_p           = max_p,
            max_new_tokens  = args.max_new_tokens,
            max_tokens      = args.max_tokens,
            batch_size      = args.batch_size,
            seed            = args.seed,
            source_hint     = args.source_hint,
        )

        # ICL judge pass (cached on disk + in-memory; only runs if needed).
        contam_icl: dict[int, float] | None = None
        uncontam_icl: dict[int, float] | None = None
        if needs_icl:
            print(f"\n=== Running GPT-4 ICL judge at min_p={min_p}, max_p={max_p} "
                  f"(cached on disk; re-runs are free) ===")
            contam_icl = compute_icl_scores(
                model_bundle    = model_bundle,
                completion_data = contam_payload,
                judge_model     = args.judge_model,
                judge_cache_dir = args.judge_cache_dir,
                judge_workers   = args.judge_workers,
            )
            uncontam_icl = compute_icl_scores(
                model_bundle    = model_bundle,
                completion_data = uncontam_payload,
                judge_model     = args.judge_model,
                judge_cache_dir = args.judge_cache_dir,
                judge_workers   = args.judge_workers,
            )

        print(f"\n  min_p={min_p}, max_p={max_p}")
        print(f"  {'score_type':>16}  {'AUROC':>8}")
        print("  " + "-" * 28)

        for score_type in score_types:
            contam_scores = _scores_for(
                contam_payload, score_type,
                contam_icl, args.use_bleurt, args.bleurt_checkpoint,
            )
            uncontam_scores = _scores_for(
                uncontam_payload, score_type,
                uncontam_icl, args.use_bleurt, args.bleurt_checkpoint,
            )
            auc = _auroc(contam_scores, uncontam_scores)
            row = {"min_p": min_p, "max_p": max_p,
                   "score_type": score_type, "auc": auc}
            all_results.append(row)
            if not np.isfinite(auc):
                print(f"  {score_type:>16}  {'N/A':>8}")
                continue
            star = ""
            if auc > best["auc"]:
                best = {"auc": auc, "score_type": score_type,
                        "min_p": min_p, "max_p": max_p}
                star = " <-- best"
            print(f"  {score_type:>16}  {auc:>8.4f}{star}")

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"MODEL  : {os.path.basename(args.model)}")
    print(f"DATASET: {os.path.basename(args.contam_path)} vs "
          f"{os.path.basename(args.uncontam_path)}")
    print(f"Best   : score_type={best['score_type']}  "
          f"min_p={best['min_p']}  max_p={best['max_p']}")
    print(f"AUROC  : {best['auc']:.4f}")
    print(f"{'=' * 60}")

    name     = os.path.basename(args.model.rstrip("/"))
    cfg_dict = "SHIFTED_CONFIG" if "shifted" in args.contam_path.lower() else "MATCHED_CONFIG"
    print(f"\n--> Update {cfg_dict} in guided_instruction_method.py:")
    print(
        f'    "{name}": '
        f'{{"score_type": "{best["score_type"]}", '
        f'"min_p": {best["min_p"]}, '
        f'"max_p": {best["max_p"]}}},'
    )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model":          args.model,
                    "contam_path":    args.contam_path,
                    "uncontam_path":  args.uncontam_path,
                    "max_tokens":     args.max_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "seed":           args.seed,
                    "score_types":    score_types,
                    "judge_model":    args.judge_model if needs_icl else None,
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