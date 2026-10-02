"""Appendix tables of bootstrap CIs (2026-09-26): one table per (metric, model size).
Rows = the 18 MIAs in the main-table order, columns = the 9 domains, cells = point estimate (lo--hi).
Usage: ../OLMo-Detect/.venv/bin/python analysis/bootstrap_ci_tables.py [--split matched]
Output: tables/appendix_bootstrap_ci_<split>.tex
"""
import sys, json, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="matched"); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / f"analysis/logs/bootstrap_ci_{a.split}.json"))
ROWS = [("PPL","L"),("Zlib","L"),("Lower","L"),("MinK","L"),("MinK++","L"),("DCPDD","L"),("PAC","L"),("ReCaLL","L"),("EMMIA","L"),("CAMIA","L"),
        ("Neigh.","P"),("Guided","Pr"),("DCQ","Pr"),("CDD","O"),("SelfCrit","O"),("Sup-CAMIA","S"),("MIA-Tuner","S"),("FSD (PPL)","S")]
DOMS = ["DCLM","peS2o","OpenWebMath","StarCoder","GSM8K","StackExchange","SFT","DPO","RLVR"]
SHORT = {"DCLM":"DCLM","peS2o":"peS2o","OpenWebMath":"OWM","StarCoder":"SC","GSM8K":"GSM8K","StackExchange":"SE","SFT":"SFT","DPO":"DPO","RLVR":"RLVR"}
SIZES = [("1b","1B"),("7b","7B"),("13b","13B"),("32b","32B")]
def f2(x): return f"{x:.2f}".lstrip("0") if 0 <= x < 1 else f"{x:.2f}"
def cell(v, key):
    if v is None: return "---"
    return f"{v[key]:.2f}\\,({f2(v[key+'_lo'])}--{f2(v[key+'_hi'])})"
spec = "@{}l" + "c@{\;\;}" * (len(DOMS) - 1) + "c@{}"
out = []
for metric, mname in (("auc", "AUC"), ("tpr", "TPR@5\\%FPR")):
    for sz, SZ in SIZES:
        lines = ["\\begin{table*}[t]", "\\centering", "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
                 "\\textbf{Method} & " + " & ".join(f"\\textbf{{{SHORT[d]}}}" for d in DOMS) + " \\\\", "\\midrule"]
        prev = None
        for m, fam in ROWS:
            if prev and fam != prev: lines.append(f"\\cmidrule(lr){{1-{len(DOMS)+1}}}")
            lab = m + (f"$^{{\\mathrm{{{fam}}}}}$" if fam != "S" else "")
            lines.append(f"{lab} & " + " & ".join(cell(D.get(f"{m}|{d}|{sz}"), metric) for d in DOMS) + " \\\\")
            prev = fam
        B = next((v["B"] for v in D.values()), 2000)
        lines += ["\\bottomrule", "\\end{tabular}%", "}",
                  "\\caption{" + f"{mname} with 95\\% bootstrap confidence intervals on \\textsc{{OLMo-Detect}}" + ("" if a.split=="matched" else " (Shifted)") +
                  f" for OLMo~2-{SZ}, per domain. Cells give the point estimate and the $2.5$th--$97.5$th percentiles over ${B:,}$ resamples of members and non-members. "
                  "Superscripts denote method families as in Table~\\ref{table:olmo_detect_overall_unsupervised}; the last block is supervised. ``---'' marks cells without a valid evaluation." + "}",
                  f"\\label{{table:ci_{metric}_{sz}{'' if a.split=='matched' else '_shifted'}}}", "\\end{table*}", ""]
        out.append("\n".join(lines))
p = ROOT / f"tables/appendix_bootstrap_ci_{a.split}.tex"
p.write_text("\n".join(out)); print("TEX ->", p, "| tables:", len(out))
print("first label:", f"table:ci_auc_1b{'' if a.split=='matched' else '_shifted'}", "| last:", f"table:ci_tpr_32b{'' if a.split=='matched' else '_shifted'}")
