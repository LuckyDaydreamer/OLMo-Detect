"""ONE compact appendix table for the three domain-knowledge settings: per method, Model-Avg over sizes of
  Q1 given the domain : macro AUC (subsample) + accuracy with the in-domain threshold
  Q2 domain unknown   : pooled AUC            + accuracy with one threshold tuned on all 9 domains
  Q3 unseen domain    : accuracy with the threshold tuned on the other 8 domains (LODO)   [AUC is unchanged by construction]
Inputs: analysis/logs/subsample_pooled.json (analysis/subsample_pooled.py), analysis/logs/lodo_threshold_check.json (analysis/lodo_threshold_check.py).
Usage: ../OLMo-Detect/.venv/bin/python analysis/three_settings_table.py -> tables/table:three_settings.tex
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; SIZES = ["1b", "7b", "13b", "32b"]
A = json.load(open(ROOT / "analysis/logs/subsample_pooled.json"))["cells"]; T = json.load(open(ROOT / "analysis/logs/lodo_threshold_check.json"))
FAM = [("L", ["PPL", "Zlib", "Lower", "MinK", "MinK++", "DCPDD", "PAC", "ReCaLL", "EMMIA", "CAMIA"]), ("P", ["Neigh."]), ("Pr", ["Guided", "DCQ"]), ("O", ["CDD", "SelfCrit"]), ("S", ["Sup-CAMIA", "MIA-Tuner"])]
def avg(vals): v = [x for x in vals if x is not None and np.isfinite(x)]; return float(np.mean(v)) if v else None
# 2026-09-26: the supervised rows must come from runs where the CLASSIFIER is retrained per setting
# (analysis/sup_three_settings.py, analysis/mt_three_settings.py), not from in-domain scores re-thresholded.
SUPJ = {}
for nm, fn in (("Sup-CAMIA", "sup_three_settings.json"), ("MIA-Tuner", "mt_three_settings.json")):
    fp = ROOT / "analysis/logs" / fn
    if fp.exists():
        j = json.load(open(fp)).get("Model-Avg")
        if j: SUPJ[nm] = dict(auc_macro=j["macro_in"], acc_in=j["acc_in"], auc_pooled=j["pooled_unknown"],
                              acc_all=j["acc_unknown"], acc_lodo=j["acc_lodo"])
rows = {}
for fam, names in FAM:
    for n in names:
        a = [A.get(f"{n}|{s}") for s in SIZES]; t = [T.get(f"{n}|{s}") for s in SIZES]
        rows[n] = dict(fam=fam, auc_macro=avg([c and c["macro_mean"] for c in a]), auc_pooled=avg([c and c["pooled_mean"] for c in a]),
                       acc_in=avg([c and c["acc_in"] for c in t]), acc_all=avg([c and c["acc_all9"] for c in t]), acc_lodo=avg([c and c["acc_lodo"] for c in t]),
                       flag=any((c or {}).get("missing_frac", 0) > 0.01 for c in a))
        if n in SUPJ: rows[n].update(SUPJ[n]); rows[n]["flag"] = False
COLS = ["auc_macro", "acc_in", "auc_pooled", "acc_all", "acc_lodo"]   # 25 Sep: header order (was macro,pooled,acc_in -> columns 2/3 mislabeled)
best = {sup: {c: max((r[c] for r in rows.values() if (r["fam"] == "S") == sup and r[c] is not None), default=None) for c in COLS} for sup in (False, True)}
def cell(v, b, flag):
    if v is None: return "---"
    s = f"{v:.2f}"; s = f"\\textbf{{{s}}}" if b is not None and abs(v - b) < 5e-4 else s
    return s + ("$^{*}$" if flag else "")
tex, last = [], None
for n, r in rows.items():
    sup = r["fam"] == "S"
    if last and r["fam"] != last: tex.append("\\midrule" if sup else "\\cmidrule(lr){1-6}")
    if sup and last != "S":  # unsupervised Method-Avg before the supervised block
        u = [x for x in rows.values() if x["fam"] != "S"]
        tex.insert(len(tex) - 1, "\\midrule\n\\textbf{Method-Avg (unsup.)} & " + " & ".join(f"{avg([x[c] for x in u]):.2f}" for c in COLS) + " \\\\")
    last = r["fam"]; lab = (f"\\textbf{{{n}}}" if sup else f"{n}$^{{\\mathrm{{{r['fam']}}}}}$") + ("$^\\dagger$" if n == "SelfCrit" else "")
    tex.append(f"{lab} & " + " & ".join(cell(r[c], best[sup][c], r["flag"]) for c in COLS) + " \\\\")
n_ps, reps = json.load(open(ROOT / "analysis/logs/subsample_pooled.json"))["n_per_side"], json.load(open(ROOT / "analysis/logs/subsample_pooled.json"))["reps"]
cap = (f"Three evaluation settings on \\textsc{{OLMo-Detect}}, averaged over model sizes (1B--32B). Every domain is subsampled to {n_ps} members and {n_ps} non-members "
       f"({reps} random draws; multi-subset domains pooled over their subsets; methods, dev sets and hyperparameters exactly as in the main tables). "
       "\\textbf{Given the domain}: AUC per domain, macro-averaged (the paper's protocol), and accuracy with a decision threshold tuned within the evaluated domain (2-fold). "
       "\\textbf{Domain unknown}: one AUC over the pooled instances of all nine domains, and accuracy with a single threshold tuned on the pooled data; supervised MIAs are retrained on the pooled dev sets of all nine domains. "
       "\\textbf{Unseen domain}: accuracy with the threshold tuned on the other eight domains and applied to the held-out domain; supervised MIAs are additionally retrained on those eight domains (LODO), while the AUC of an unsupervised MIA is unchanged by construction. "
       "Thresholds maximise accuracy on their tuning data. "
       "Best per column and block in bold. $^{*}$partial coverage (DCQ). $^\\dagger$SelfCrit is evaluated on DPO and RLVR only. Accuracies are omitted for EMMIA, which reports the best of its four base scores per subset and thus has no single score to threshold, and for SelfCrit, whose two domains leave no other domains to tune on.")
latex = ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
         "\\multirow{2.5}{*}{\\textbf{Method}} & \\multicolumn{2}{c}{\\textbf{Given the domain}} & \\multicolumn{2}{c}{\\textbf{Domain unknown}} & \\textbf{Unseen domain} \\\\\n"
         "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-6}\n & AUC (macro) & Acc. & AUC (pooled) & Acc. & Acc. \\\\\n\\midrule\n" + "\n".join(tex)
         + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\label{table:three_settings}\n\\end{table}\n")
out = ROOT / "tables/table:three_settings.tex"; out.write_text(latex); print("TEX ->", out)
for n, r in rows.items(): print(f"{n:10s} " + " ".join(f"{c}={'---' if r[c] is None else f'{r[c]:.3f}'}" for c in COLS))
u = [x for x in rows.values() if x["fam"] != "S"]; print("unsup avg:", {c: round(avg([x[c] for x in u]), 3) for c in COLS})
