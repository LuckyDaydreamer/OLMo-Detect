"""
Compute AUC and TPR@5%FPR for every detection method on ONE domain.

A "domain" is a (contaminated, uncontaminated) pair of *result directories*
produced by run_detection.py, each laid out as:

    <results_dir>/<model_name>/<method>.json

where each <method>.json stores per-record scores under
result.samples[*].<score_column>. This script pairs the contaminated
(label=1) vs uncontaminated (label=0) scores for each (method, model size)
and reports both AUC and TPR at a target FPR (default 5%).

It pairs the two result directories produced by run_detection.py and reports,
for each method and model size, both AUC and TPR@5%FPR — the two metrics used
in the paper.

--------------------------------------------------------------------------
USAGE
  python "AUC_TPR@5%FPR.py" \
      --con   results/contaminated_pretraining_openwebmath_test_matched \
      --uncon results/uncontaminated_pretraining_openwebmath_test \
      [--output results/openwebmath_auc_tpr.json] \
      [--fpr 0.05] \
      [--emmia-dir results/emmia/pretraining_openwebmath_test_matched]

  --con   PATH   Result dir for the contaminated half of the domain.
  --uncon PATH   Result dir for the uncontaminated half of the domain.
  --output PATH  Optional JSON dump of the full results table.
  --fpr   FLOAT  Target FPR for the TPR metric (default 0.05 = 5%).
  --models "..." Space-separated model dir names to evaluate
                 (default: the 4 OLMo-2 sizes that are present).
  --emmia-dir P  Optional EMMIA cell dir (contains emmia_<size>.json) for
                 this domain; EMMIA has a different output layout, so it is
                 only included when this is given.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# (method file stem, scoring column in result.samples[*], display name).
# Same mapping the paper's per-domain AUC scripts use.
METHODS = [
    ("loss_zlib_lowercase", "loss_score",                "Perplexity"),
    ("loss_zlib_lowercase", "zlib_score",                "Zlib"),
    ("loss_zlib_lowercase", "lowercase_score",           "Lowercase"),
    ("minkprob",            "min_k_prob_mean_logprob",   "Min-K%"),
    ("minkprob",            "min_k_pp_score",            "Min-K%++"),
    ("dcpdd",               "dc_pdd_score",              "DC-PDD"),
    ("recall",              "recall_score",              "RECALL"),
    ("camia",               "camia_score",               "CAMIA"),
    ("pac",                 "pac_score",                 "PAC"),
    ("dcq",                 "score",                     "DCQ"),
    ("neighborhood_attack", "score",                     "Neighborhood Attack"),
    ("cdd",                 "peak",                      "CDD"),
    ("guided_instruction",  "score",                     "Guided Instruction"),
    ("selfcrit",            "self_critique_score",       "Self-Critique"),
]

# Default model sizes (dir name under each result dir).
MODEL_SIZES = [
    ("1b",  "OLMo2-1B-Instruct"),
    ("7b",  "OLMo2-7B-Instruct"),
    ("13b", "OLMo2-13B-Instruct"),
    ("32b", "OLMo2-32B-Instruct"),
]

# EMMIA sub-methods; its per-cell file holds these under em_results / final_scores.
EMMIA_SUBS = ["loss", "zlib", "mink", "minkpp"]


def _valid(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if v == "-Infinity":
        return False
    return True


def load_scores(path: Path, col: str):
    """Return the list of valid scores for `col` from a per-method result file."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    samples = d.get("result", {}).get("samples", [])
    out = []
    for s in samples:
        v = s.get(col)
        if _valid(v):
            out.append(float(v))
    return out


def auc_tpr(con_scores, unc_scores, target_fpr: float):
    """AUC and TPR@target_fpr for contaminated(=1) vs uncontaminated(=0) scores.

    Higher score = more likely contaminated (members), matching every method's
    convention in this benchmark.
    """
    if not con_scores or not unc_scores:
        return None, None, len(con_scores or []), len(unc_scores or [])
    y = np.array([1] * len(con_scores) + [0] * len(unc_scores), dtype=np.int8)
    scores = np.array(con_scores + unc_scores, dtype=np.float64)
    auc = float(roc_auc_score(y, scores))
    if np.all(scores == scores[0]):
        # Degenerate: all scores identical -> ROC is the diagonal; TPR≈FPR.
        return auc, float(target_fpr), len(con_scores), len(unc_scores)
    fpr, tpr, _ = roc_curve(y, scores)
    tpr_at = float(np.interp(target_fpr, fpr, tpr))
    return auc, tpr_at, len(con_scores), len(unc_scores)


def standard_method(con_dir: Path, unc_dir: Path, method_file: str, col: str,
                    model_dir: str, target_fpr: float):
    con = load_scores(con_dir / model_dir / f"{method_file}.json", col)
    unc = load_scores(unc_dir / model_dir / f"{method_file}.json", col)
    return auc_tpr(con, unc, target_fpr)


def emmia_cell(emmia_dir: Path, size: str, target_fpr: float):
    """AUC and TPR@fpr for EMMIA.

    EMMIA has a different output layout, so the two metrics are computed the
    same way as the paper's *separate* aggregators, and — importantly — the
    best sub-method is chosen INDEPENDENTLY for each metric:

      * AUC  (aggregate_emmia_auc.py): max em_results.<sub>.final_auc.
      * TPR  (aggregate_emmia_tpr.py): max, over sub-methods, of the TPR@fpr
              recomputed from the per-sample final_scores. Chosen separately,
              not tied to the AUC-best sub. Requires per-sample data, so TPR
              is None for cells recovered-from-logs (no `samples`).
    """
    p = emmia_dir / f"emmia_{size}.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())

    # ── AUC: best final_auc across sub-methods (always available) ──
    em = d.get("em_results") or {}
    aucs = {k: em.get(k, {}).get("final_auc") for k in EMMIA_SUBS}
    aucs = {k: v for k, v in aucs.items() if isinstance(v, (int, float))}
    best_auc = max(aucs.values()) if aucs else None

    # ── TPR: best TPR@fpr across sub-methods, computed independently ──
    best_tpr = None
    samples = d.get("samples") or []
    if samples:
        tprs = []
        for sub in EMMIA_SUBS:
            con_s, unc_s = [], []
            for s in samples:
                v = (s.get("final_scores") or {}).get(sub)
                if not isinstance(v, (int, float)):
                    continue
                src = s.get("source")
                if src == "contaminated":
                    con_s.append(float(v))
                elif src == "uncontaminated":
                    unc_s.append(float(v))
            if con_s and unc_s:
                _, t, _, _ = auc_tpr(con_s, unc_s, target_fpr)
                if t is not None:
                    tprs.append(t)
        if tprs:
            best_tpr = max(tprs)
    return best_auc, best_tpr


def _present_models(con_dir: Path, unc_dir: Path, requested):
    """Keep only sizes whose model dir exists under both con and unc."""
    out = []
    for sz, model_dir in requested:
        if (con_dir / model_dir).is_dir() and (unc_dir / model_dir).is_dir():
            out.append((sz, model_dir))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--con", required=True, help="Contaminated result dir.")
    ap.add_argument("--uncon", required=True, help="Uncontaminated result dir.")
    ap.add_argument("--output", default=None, help="Optional JSON output path.")
    ap.add_argument("--fpr", type=float, default=0.05, help="Target FPR (default 0.05).")
    ap.add_argument("--models", default=None,
                    help="Space-separated model dir names (default: 4 OLMo-2 sizes present).")
    ap.add_argument("--emmia-dir", default=None,
                    help="Optional EMMIA cell dir (emmia_<size>.json) for this domain.")
    args = ap.parse_args()

    con_dir = Path(args.con)
    unc_dir = Path(args.uncon)
    for d in (con_dir, unc_dir):
        if not d.is_dir():
            ap.error(f"result dir not found: {d}")

    if args.models:
        wanted = [(m.split("OLMo2-")[-1].split("-")[0].lower() if "OLMo2-" in m else m, m)
                  for m in args.models.split()]
    else:
        wanted = MODEL_SIZES
    sizes = _present_models(con_dir, unc_dir, wanted)
    if not sizes:
        ap.error("No model dirs found under both --con and --uncon.")

    fpr_pct = f"{args.fpr * 100:g}%"
    emmia_dir = Path(args.emmia_dir) if args.emmia_dir else None

    size_labels = [sz for sz, _ in sizes]
    hdr = f"{'Method':22s} " + " ".join(f"{s.upper():>16s}" for s in size_labels) + f"   {'Avg':>16s}"
    print(f"Domain: {con_dir.name}  vs  {unc_dir.name}")
    print(f"Metric: AUC / TPR@{fpr_pct}   (per model size, then cross-size mean)")
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    methods_to_run = list(METHODS)
    if emmia_dir is not None:
        methods_to_run.append(("__emmia__", None, "EMMIA"))

    for method_file, col, display in methods_to_run:
        per_size = {}
        cells = []
        for sz, model_dir in sizes:
            if method_file == "__emmia__":
                auc, tpr = emmia_cell(emmia_dir, sz, args.fpr)
                n_c = n_u = None
            else:
                auc, tpr, n_c, n_u = standard_method(
                    con_dir, unc_dir, method_file, col, model_dir, args.fpr)
            per_size[sz] = {"auc": auc, "tpr_at_fpr": tpr, "n_con": n_c, "n_uncon": n_u}
            cells.append((auc, tpr))

        aucs = [a for a, _ in cells if a is not None]
        tprs = [t for _, t in cells if t is not None]
        avg_auc = sum(aucs) / len(aucs) if aucs else None
        avg_tpr = sum(tprs) / len(tprs) if tprs else None

        def cell_str(a, t):
            a_s = f"{a:.4f}" if a is not None else "  n/a "
            t_s = f"{t:.4f}" if t is not None else "  n/a "
            return f"{a_s}/{t_s}"

        line = f"{display:22s} " + " ".join(f"{cell_str(a, t):>16s}" for a, t in cells)
        line += f"   {cell_str(avg_auc, avg_tpr):>16s}"
        print(line)

        out_rows.append({
            "method": display,
            "method_file": method_file,
            "score_column": col,
            "per_size": per_size,
            "mean_auc_across_sizes": avg_auc,
            "mean_tpr_across_sizes": avg_tpr,
        })

    if args.output:
        payload = {
            "con_dir": str(con_dir),
            "uncon_dir": str(unc_dir),
            "target_fpr": args.fpr,
            "model_sizes": size_labels,
            "emmia_dir": str(emmia_dir) if emmia_dir else None,
            "rows": out_rows,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
