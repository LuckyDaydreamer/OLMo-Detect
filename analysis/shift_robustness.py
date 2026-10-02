"""Robustness to distribution shift: classify every MIA as robust / inflated / degraded / inconclusive from
Delta = AUC_shifted - AUC_matched over the (subset, size) pairs that have a shifted variant (10 subsets x 4 sizes = 40 pairs per method;
non-members are shared between the two splits, so Delta is a paired comparison).

Statistics per method:
  mean Delta over pairs (macro), 95% CI from a CLUSTER bootstrap over subsets (all 4 sizes of a subset resampled together; B resamples),
  share of pairs with Delta < 0, Wilcoxon signed-rank p (two-sided) over the pairs (secondary, ignores clustering).
Classification with equivalence margin DELTA (default 0.03 AUC):
  degraded     CI entirely below 0
  inflated     CI entirely above 0
  robust       CI inside [-DELTA, +DELTA]   (two one-sided tests / equivalence)
  inconclusive otherwise (CI overlaps 0 but is wider than the margin)
Same AUC definitions as analysis/paper_tables.py (results_latest; EMMIA from emmia_raw; SelfCrit only DPO on the shifted split -> 4 pairs, flagged).

Usage: ../OLMo-Detect/.venv/bin/python analysis/shift_robustness.py [--delta 0.03] [--boot 10000]
Outputs: tables/table:distribution_shifts_detail.tex, analysis/logs/shift_robustness.json, markdown to stdout.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT

ap = argparse.ArgumentParser()
ap.add_argument("--delta", type=float, default=0.03)
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
SIZES = PT.SIZES
SUBSETS = [k for _, _, keys in PT.STRUCT for k in keys if PT.leaves(k, "shifted", "7b") is not None]   # subsets with a shifted variant
rng = np.random.default_rng(a.seed)

def pairs(name, stem, col):
    """-> list of (subset, size, auc_matched, auc_shifted, complete)"""
    out = []
    for k in SUBSETS:
        if name == "SelfCrit" and k != "dpo": continue
        for s in SIZES:
            am, okm, _ = PT.subset_auc(name, stem, col, k, "matched", s)
            ash, oks, note = PT.subset_auc(name, stem, col, k, "shifted", s)
            if am is None or ash is None: continue
            out.append((k, s, am, ash, okm and oks))
    return out

def cluster_ci(d_by_subset, B):
    subs = list(d_by_subset); n = len(subs)
    if n == 0: return (np.nan, np.nan)
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, n, n)
        means[b] = np.mean(np.concatenate([d_by_subset[subs[i]] for i in pick]))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

def classify(lo, hi, delta):
    if np.isnan(lo): return "n/a"
    if hi < 0: return "degraded"
    if lo > 0: return "inflated"
    if lo >= -delta and hi <= delta: return "robust"
    return "inconclusive"

res = {}
for name, stem, col in PT.METHODS:
    P = pairs(name, stem, col)
    if not P: res[name] = None; continue
    d = np.array([sh - m for _, _, m, sh, _ in P]); by = {}
    for k, s, m, sh, ok in P: by.setdefault(k, []).append(sh - m)
    by = {k: np.array(v) for k, v in by.items()}
    lo, hi = cluster_ci(by, a.boot)
    p = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan")
    res[name] = dict(n_pairs=len(P), n_subsets=len(by), matched=float(np.mean([m for _, _, m, _, _ in P])), shifted=float(np.mean([sh for _, _, _, sh, _ in P])),
                     delta_mean=float(d.mean()), ci=(lo, hi), frac_neg=float(np.mean(d < 0)), p_wilcoxon=p, cls=classify(lo, hi, a.delta),
                     incomplete=int(sum(1 for *_, ok in P if not ok)), per_subset={k: float(v.mean()) for k, v in by.items()})

(ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
(ROOT / "analysis/logs/shift_robustness.json").write_text(json.dumps(dict(delta=a.delta, boot=a.boot, subsets=SUBSETS, methods=res), indent=1))

SYM = {"robust": "R", "inflated": "I", "degraded": "D", "inconclusive": "?", "n/a": "---"}
md = ["| Method | n pairs | Matched | Shifted | mean Δ [95% CI] | pairs Δ<0 | Wilcoxon p | Class |", "|---|---|---|---|---|---|---|---|"]
tex = []
for name, _, _ in PT.METHODS:
    r = res[name]
    if r is None: md.append(f"| {name} | --- | | | | | | --- |"); tex.append(f"{name} & --- & --- & --- & --- & --- & --- \\\\"); continue
    flag = "$^{*}$" if r["incomplete"] else ""
    ci = f"{r['delta_mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
    md.append(f"| {name}{'*' if r['incomplete'] else ''} | {r['n_pairs']} | {r['matched']:.3f} | {r['shifted']:.3f} | {ci} | {100*r['frac_neg']:.0f}% | {r['p_wilcoxon']:.3f} | {r['cls']} |")
    tex.append(f"{name}{flag} & {r['n_pairs']} & {r['matched']:.2f} & {r['shifted']:.2f} & ${r['delta_mean']:+.2f}$ [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] & {100*r['frac_neg']:.0f}\\% & {SYM[r['cls']]} \\\\")
cap = (f"Robustness of unsupervised MIAs to distribution shift. $\\Delta=\\mathrm{{AUC}}_{{\\text{{Shifted}}}}-\\mathrm{{AUC}}_{{\\text{{Matched}}}}$ is computed for every (subset, model size) pair "
       f"that has a shifted variant ({len(SUBSETS)} subsets $\\times$ 4 sizes; non-members are shared between the splits), and averaged (Matched/Shifted = mean AUC over the same pairs). "
       f"The 95\\% CI is a cluster bootstrap over subsets ({a.boot:,} resamples; sizes of one subset resampled together). Classification: \\textbf{{D}}egraded if the CI lies below 0, "
       f"\\textbf{{I}}nflated if above 0, \\textbf{{R}}obust if the CI lies within the equivalence margin $[-{a.delta}, +{a.delta}]$, and ? (inconclusive) otherwise. "
       f"SelfCrit is evaluated on DPO only (the only SelfCrit domain with a shifted variant). $^{{*}}$: some pairs computed on partial coverage.")
latex = ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccccc}\n\\toprule\nMethod & pairs & Matched & Shifted & $\\overline{\\Delta}$ [95\\% CI] & $\\Delta<0$ & Class \\\\\n\\midrule\n"
         + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\label{table:distribution_shifts_detail}\n\\end{table}\n")
out = ROOT / "tables/table:distribution_shifts_detail.tex"; out.write_text(latex)
print("\n".join(md)); print(f"\nsubsets: {SUBSETS}\nTEX -> {out}\nJSON -> {ROOT/'analysis/logs/shift_robustness.json'}")
