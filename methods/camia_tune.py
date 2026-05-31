"""
Tune the composition method (Edgington / Fisher / George) for CAMIA on the dev split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from data_utils import extract_text
from model_utils import load_model_bundle
from camia_method import (
    CAMIAMethod,
    _signals_matrix,
    _calibrate_directions,
    _apply_directions,
    _pvalues,
    _combine,
)


def load_texts(path: str, limit: Optional[int] = None) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = extract_text(json.loads(line))
            if text:
                texts.append(text)
                if limit is not None and len(texts) >= limit:
                    break
    return texts


def run_tuning(
    model_path:              str,
    contam_path:             str,
    uncontam_path:           str,
    max_tokens:              int = 512,
    batch_size:              int = 2,
    max_calibration_samples: int = 100,
) -> str:

    # ── 1. model ──────────────────────────────────────────────────────────────
    print(f"--> Loading model bundle: {model_path}")
    model_bundle = load_model_bundle(model_path)
    tokenizer    = model_bundle.tokenizer

    # ── 2. dev texts ──────────────────────────────────────────────────────────
    contam_texts   = load_texts(contam_path)
    uncontam_texts = load_texts(uncontam_path)
    print(f"  contam   dev: {len(contam_texts)} texts")
    print(f"  uncontam dev: {len(uncontam_texts)} texts")

    if len(contam_texts) == 0 or len(uncontam_texts) < 4:
        print("ERROR: not enough dev texts for tuning "
              "(need >=1 contam and >=4 uncontam).", file=sys.stderr)
        sys.exit(1)

    # Calibration / eval split for uncon_dev:
    #   first k samples = null distribution
    #   rest            = held-out non-member eval
    k = min(max_calibration_samples, len(uncontam_texts) - 1)
    if k < 1:
        print("ERROR: max_calibration_samples too large for uncontam dev size.",
              file=sys.stderr)
        sys.exit(1)
    cal_uncon_texts  = uncontam_texts[:k]
    eval_uncon_texts = uncontam_texts[k:]
    if len(eval_uncon_texts) < 1:
        print("ERROR: no held-out non-members left after calibration split.",
              file=sys.stderr)
        sys.exit(1)
    print(f"  -> calibration null: {len(cal_uncon_texts)} non-members")
    print(f"  -> eval set: {len(contam_texts)} members + "
          f"{len(eval_uncon_texts)} held-out non-members")

    # ── 3. helper instance for forward-pass utilities ─────────────────────────
    helper = CAMIAMethod(
        uncon_dev_path          = uncontam_path,
        con_dev_path            = contam_path,
        composition             = "edgington",   # placeholder; we sweep below
        max_tokens              = max_tokens,
        batch_size              = batch_size,
        max_calibration_samples = max_calibration_samples,
    )

    def compute_sigs(texts: list[str], desc: str) -> dict[str, np.ndarray]:
        lps     = helper._lp_batch(model_bundle, texts, max_tokens, desc)
        r1, r2  = helper._repeated_lp(model_bundle, texts, lps, desc)
        tids    = [tokenizer.encode(t, add_special_tokens=False)[:max_tokens]
                   for t in texts]
        return _signals_matrix(lps, r1, r2, tids)

    # ── 4. forward passes (orig + 2 repetitions) ──────────────────────────────
    print(f"--> Computing signals on contaminated dev "
          f"({len(contam_texts)} samples)...")
    con_sigs_raw = compute_sigs(contam_texts, "camia_tune_con")

    print(f"--> Computing signals on uncontaminated dev "
          f"({len(uncontam_texts)} samples)...")
    uncon_sigs_raw = compute_sigs(uncontam_texts, "camia_tune_uncon")

    # ── 5. direction calibration from labelled dev means ──────────────────────
    directions = _calibrate_directions(con_sigs_raw, uncon_sigs_raw)

    con_sigs   = _apply_directions(con_sigs_raw,   directions)
    uncon_sigs = _apply_directions(uncon_sigs_raw, directions)

    # ── 6. build calibration null and eval matrices ───────────────────────────
    common_keys = set(con_sigs).intersection(uncon_sigs)
    if not common_keys:
        print("ERROR: no signals shared between con and uncon dev.",
              file=sys.stderr)
        sys.exit(1)

    calib_sigs = {kk: uncon_sigs[kk][:k] for kk in common_keys}
    eval_sigs  = {
        kk: np.concatenate([con_sigs[kk], uncon_sigs[kk][k:]])
        for kk in common_keys
    }
    eval_labels = np.concatenate([
        np.ones(len(contam_texts),    dtype=int),
        np.zeros(len(eval_uncon_texts), dtype=int),
    ])
    n_eval = len(eval_labels)

    # ── 7. p-values ───────────────────────────────────────────────────────────
    pvals = _pvalues(eval_sigs, calib_sigs)
    if not pvals:
        print("ERROR: no signals could produce p-values "
              "(empty calibration?).", file=sys.stderr)
        sys.exit(1)

    print(f"  signals used: {len(pvals)}")

    # ── 8. sweep compositions ─────────────────────────────────────────────────
    print(f"\n  {'composition':<12}  {'AUROC':>8}")
    print("  " + "-" * 24)

    results: dict[str, float] = {}
    for comp in ("edgington", "fisher", "george"):
        try:
            scores = _combine(pvals, comp, n_eval)
        except Exception as exc:
            print(f"  {comp:<12}  ERROR: {exc}")
            results[comp] = float("nan")
            continue

        if len(scores) != n_eval:
            print(f"  {comp:<12}  shape mismatch "
                  f"({len(scores)} vs {n_eval})")
            results[comp] = float("nan")
            continue

        valid = np.isfinite(scores)
        if (valid.sum() < 2
                or eval_labels[valid].sum() == 0
                or (1 - eval_labels[valid]).sum() == 0):
            results[comp] = float("nan")
            print(f"  {comp:<12}  N/A (degenerate eval set)")
            continue

        auc = float(roc_auc_score(eval_labels[valid], scores[valid]))
        results[comp] = auc
        print(f"  {comp:<12}  {auc:.4f}")

    finite = {kk: vv for kk, vv in results.items() if np.isfinite(vv)}
    if not finite:
        print("\nNo valid AUROC found. Defaulting to edgington.")
        return "edgington"

    best = max(finite, key=finite.get)

    print(f"\n{'=' * 55}")
    print(f"MODEL : {os.path.basename(model_path)}")
    print(f"Best composition: {best}  (AUROC {finite[best]:.4f})")
    print(f"{'=' * 55}")
    print(f"\n--> Update COMPOSITION_CONFIG (or _SHIFTED) in camia_method.py:")
    # Map the model dir name to the size key used in COMPOSITION_CONFIG
    name = os.path.basename(model_path)
    size_key = "1B" if "1B" in name else \
               "7B" if "7B" in name else \
               "13B" if "13B" in name else \
               "32B" if "32B" in name else name
    print(f'    "{size_key}":  "{best}",')

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune CAMIA composition method on dev data."
    )
    parser.add_argument("--model",         type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--contam_path",   type=str, required=True,
                        help="Contaminated dev JSONL")
    parser.add_argument("--uncontam_path", type=str, required=True,
                        help="Uncontaminated dev JSONL")
    parser.add_argument("--max_tokens",    type=int, default=512)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--max_calibration_samples", type=int, default=100,
                        help="Number of uncon dev samples used as the null. "
                             "The rest become held-out non-member eval samples.")
    args = parser.parse_args()

    run_tuning(
        model_path              = args.model,
        contam_path             = args.contam_path,
        uncontam_path           = args.uncontam_path,
        max_tokens              = args.max_tokens,
        batch_size              = args.batch_size,
        max_calibration_samples = args.max_calibration_samples,
    )