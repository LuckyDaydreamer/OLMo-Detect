"""Distribution-shift table in the paper's layout (Size x {Matched, Shifted} rows; 14 unsupervised MIAs + Avg columns; Avg Delta + Class rows).

AUC per (method, size, setting) = the paper's all-stage aggregation (subset -> domain mean -> stage mean -> mean of stages) restricted to the
subsets that have a shifted variant (10: DCLM, peS2o, OWM, Python, Java, SE-Math, SE-SO, SFT-Aya, SFT-WildChat, DPO), so Matched and Shifted rows
are computed on identical subsets; per method/size, subsets missing on either side are dropped from both.
Avg Delta = mean over sizes of (Shifted - Matched). Class = cluster bootstrap over subsets (B resamples; each resample recomputes the same
stage-macro statistic for all sizes): degraded if the 95% CI < 0, inflated if > 0, robust if the CI lies within [-delta, +delta], else '?'.

Usage: ../OLMo-Detect/.venv/bin/python analysis/shift_table_paper.py [--delta 0.03] [--boot 5000]
Output: tables/table:distribution_shifts.tex, analysis/logs/shift_table_paper.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT

ap = argparse.ArgumentParser()
ap.add_argument("--delta", type=float, default=0.0, help="equivalence margin; 0 (default) = 3-way scheme (R = CI includes 0), >0 = 4-way scheme with ? (inconclusive)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ptest", default="bootstrap", choices=["bootstrap","wilcoxon"], help="p-value source: paired instance bootstrap (analysis/logs/shift_bootstrap.json) or per-subset Wilcoxon")
ap.add_argument("--alpha", type=float, default=0.01, help="significance level for I/D")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; SIZES = PT.SIZES
rng = np.random.default_rng(a.seed)

# subsets with a shifted variant, with their (stage, domain)
SUB = [(k, stg, dom) for stg, dom, keys in PT.STRUCT for k in keys if PT.leaves(k, "shifted", "7b") is not None]
KEYS = [k for k, _, _ in SUB]; STG_OF = {k: s for k, s, _ in SUB}; DOM_OF = {k: d for k, _, d in SUB}
METHODS = [(n, s, c) for n, s, c in PT.METHODS if n != "SelfCrit"]
COLS = ["PPL", "Zlib", "Lower", "MinK", "MinK++", "DCPDD", "ReCaLL", "CAMIA", "PAC", "Neigh.", "EMMIA", "DCQ", "CDD", "Guided"]

def stage_macro(vals):
    """vals: list of (subset_key, value) (a key may repeat under bootstrap) -> paper aggregation (domain mean -> stage mean -> mean of stages)"""
    dom = {}
    for k, v in vals: dom.setdefault((STG_OF[k], DOM_OF[k]), []).append(v)
    stg = {}
    for (s, d), vs in dom.items(): stg.setdefault(s, []).append(np.mean(vs))
    return float(np.mean([np.mean(v) for v in stg.values()])) if stg else float("nan")

A = {}   # (name, size, split, key) -> auc
for name, stem, col in METHODS:
    for s in SIZES:
        for k in KEYS:
            for split in ("matched", "shifted"):
                v, ok, note = PT.subset_auc(name, stem, col, k, split, s)
                A[(name, s, split, k)] = v

res = {}
for name, _, _ in METHODS:
    per = {}
    for s in SIZES:
        ks = [k for k in KEYS if A[(name, s, "matched", k)] is not None and A[(name, s, "shifted", k)] is not None]
        per[s] = dict(keys=ks, matched=stage_macro([(k, A[(name, s, "matched", k)]) for k in ks]) if ks else None,
                      shifted=stage_macro([(k, A[(name, s, "shifted", k)]) for k in ks]) if ks else None)
    sizes_ok = [s for s in SIZES if per[s]["matched"] is not None]
    deltas = [per[s]["shifted"] - per[s]["matched"] for s in sizes_ok]
    avg_delta = float(np.mean(deltas)) if deltas else None
    # per-subset Delta (mean over sizes where both settings exist) -> sign test
    from scipy.stats import binomtest
    U = sorted(set(k for s in sizes_ok for k in per[s]["keys"])); dsub = {}
    for k in U:
        vs = [A[(name, s, "shifted", k)] - A[(name, s, "matched", k)] for s in sizes_ok if k in per[s]["keys"]]
        if vs: dsub[k] = float(np.mean(vs))
    from scipy.stats import wilcoxon
    dv = np.array(list(dsub.values())); pos = int((dv > 0).sum()); n = len(dv)
    pval = float(wilcoxon(dv, alternative="two-sided").pvalue) if n >= 5 and np.any(dv != 0) else float("nan")
    if a.ptest == "bootstrap":
        BJ = json.load(open(ROOT / "analysis/logs/shift_bootstrap.json")) if (ROOT / "analysis/logs/shift_bootstrap.json").exists() else {}
        pval = float(BJ[name]["p"]) if name in BJ else float("nan")
        ci99 = tuple(BJ[name]["ci99"]) if name in BJ and "ci99" in BJ[name] else None
        BOOT_B = BJ[name].get("B", 1000) if name in BJ else 1000
    else: ci99 = None; BOOT_B = None
    sig = (ci99 is not None and (ci99[0] > 0 or ci99[1] < 0)) if ci99 is not None else (not np.isnan(pval) and pval < a.alpha)
    if avg_delta is None or n == 0 or np.isnan(pval): cls = "---"
    elif sig and avg_delta > 0: cls = "I"
    elif sig and avg_delta < 0: cls = "D"
    else: cls = "R"
    res[name] = dict(per=per, avg_delta=avg_delta, pos=pos, n=n, pval=pval, cls=cls, per_subset=dsub, ci99=ci99)
# Holm step-down over the tested methods (family = all methods with a p-value)
_pm = [(n, res[n]["pval"]) for n in res if not np.isnan(res[n]["pval"])]; _pm.sort(key=lambda x: x[1]); _m = len(_pm); _run = 0.0
HOLM = {}
for _i, (_n, _p) in enumerate(_pm):
    _run = max(_run, min(1.0, (_m - _i) * _p)); HOLM[_n] = _run
for n in res: res[n]["holm"] = HOLM.get(n, float("nan")); res[n]["holm_ok"] = (res[n]["cls"] in ("I", "D") and HOLM.get(n, 1.0) < a.alpha)


# ---------- table ----------
def f(v): return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
rows = []
for s in SIZES:
    for split in ("matched", "shifted"):
        vals = [res[n]["per"][s][split] for n in COLS]
        avg = float(np.mean([v for v in vals if v is not None]))
        rows.append((s, split, vals, avg))
lines = []
for i, s in enumerate(SIZES):
    m = [r for r in rows if r[0] == s]
    lines.append(f"\\multirow{{2}}{{*}}{{{s.upper()}}}")
    for (_, split, vals, avg) in m:
        lines.append(f" & {split.capitalize()} & " + " & ".join(f(v) for v in vals) + f" & {avg:.2f} \\\\")
    if i < len(SIZES) - 1: lines.append("\\cmidrule(l){1-17}")
row_avg_delta = []
for s in SIZES:
    m = {r[1]: r[3] for r in rows if r[0] == s}; row_avg_delta.append(m["shifted"] - m["matched"])
def sd(v): return "---" if v is None else ("$+$" if v >= 0 else "$-$") + f"{abs(v):.2f}"
def sdci(n):
    r = res[n]; v = r["avg_delta"]
    if v is None or np.isnan(r["ci"][0]): return "---"
    hw = max(v - r["ci"][0], r["ci"][1] - v)
    return sd(v) + f"$\\pm${hw:.2f}"
lines.append("\\midrule")
def pv(n):
    r = res[n]; v = r["pval"]
    return "---" if np.isnan(v) else ("$<$0.01" if v < 0.01 else f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
lines.append("\\multicolumn{2}{l}{\\textbf{$\\overline{\\Delta}$}} & " + " & ".join(sd(res[n]["avg_delta"]) for n in COLS) + f" & {sd(float(np.mean(row_avg_delta)))} \\\\")
def cnt(n):
    r = res[n]; return "---" if r["n"] == 0 else f"{r['pos']}/{r['n']}"
lines.append("\\multicolumn{2}{l}{\\textbf{$p$-value}} & " + " & ".join(pv(n) for n in COLS) + " & \\\\")
lines.append("\\midrule")
lines.append("\\multicolumn{2}{l}{\\textbf{Class.}} & " + " & ".join(res[n]["cls"] + ("" if res[n]["cls"] not in ("I", "D") or res[n]["holm_ok"] else "$^{\\S}$") for n in COLS) + " & \\\\")
AL = f"{a.alpha:g}"; _Bs = {r.get("ci99") is not None for r in res.values()}; B_TXT = "10{,}000" if any(json.load(open(ROOT / "analysis/logs/shift_bootstrap.json")).get(n, {}).get("B", 0) >= 10000 for n in res) else "1{,}000"
cap = ("Comparison of unsupervised MIAs between \\textsc{OLMo-Detect} (\\textbf{Matched}) and \\textsc{OLMo-Detect (Shifted)} (\\textbf{Shifted}), using all-stage AUC over the "
       "ten subsets with shifted variants (Table~\\ref{appendix:table:olmo_detect_shifted:overall_statistics}). \\textbf{Avg} averages across methods per row. "
       "\\textbf{$\\overline{\\Delta}$}: mean of (Shifted $-$ Matched) across model sizes. "
       + (f"The $p$-value (two-sided) is from a paired instance-level bootstrap of $\\overline{{\\Delta}}$ ({B_TXT} resamples): within every subset, members and the shared "
          "non-members are resampled with replacement, the same resampled instances are used for all model sizes, and the resampled non-members are used for both AUCs. "
          if a.ptest == "bootstrap" else "The $p$-value is from a two-sided Wilcoxon signed-rank test on the same ten per-subset differences. ")
       + f" \\textbf{{Class.}}: \\textbf{{I}} (inflated) if $p<{AL}$ and $\\overline{{\\Delta}}>0$; \\textbf{{D}} (degraded) if $p<{AL}$ and "
       "$\\overline{\\Delta}<0$; \\textbf{R} (robust) otherwise. $^{\\S}$: not significant after a Holm correction over the 14 methods. "
       "The bootstrap resamples instances, so the classification refers to these ten subsets. Detailed results are in Appendix~\\ref{appendix:rq5}.")
tex = ("\\begin{table*}[h!]\n\\centering\n\\resizebox{1\\textwidth}{!}{%\n\\begin{tabular}{@{}llccccccccccccccc@{}}\n\\toprule\n"
       "\\multirow{2}{*}{\\textbf{Size}} & \\multirow{2}{*}{\\textbf{Setting}} & \\multicolumn{11}{c}{\\textbf{Likelihood-based}} & \\textbf{Pert.} & \\textbf{Out.-Dist.} & \\textbf{Prompt} & \\multirow{2}{*}{\\textbf{Method-Avg}} \\\\\n"
       "\\cmidrule(lr){3-13} \\cmidrule(lr){14-14} \\cmidrule(lr){15-15} \\cmidrule(lr){16-16}\n"
       " & & \\textbf{PPL} & \\textbf{Zlib} & \\textbf{Lower} & \\textbf{MinK} & \\textbf{MinK++} & \\textbf{DCPDD} & \\textbf{RECALL} & \\textbf{CAMIA} & \\textbf{PAC} & \\textbf{Neigh.} & \\textbf{EMMIA} & \\textbf{DCQ} & \\textbf{CDD} & \\textbf{Guided} & \\\\\n\\midrule\n"
       + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}%\n}\n\\caption{" + cap + "}\n\\label{table:distribution_shifts}\n\\end{table*}\n")
out = ROOT / "tables/table:distribution_shifts.tex"; out.write_text(tex)
(ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
json.dump({n: dict(avg_delta=r["avg_delta"], pos=r["pos"], n=r["n"], pval=r["pval"], per_subset=r["per_subset"], cls=r["cls"], per={s: {k: v for k, v in r["per"][s].items() if k != "keys"} for s in SIZES}) for n, r in res.items()},
          open(ROOT / "analysis/logs/shift_table_paper.json", "w"), indent=1)
print(tex); print(f"TEX -> {out}")
