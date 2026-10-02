"""Compact FSD gain table: rows = FSD on each base detector; columns = base AUC (Model-Avg, in-domain) and the all-stage AUC gain of FSD over
its base under In-domain and LODO for 1B/7B/13B/32B + Avg. Same numbers as analysis/paper_tables.py main_sup (sup_stages 'all' = stage-macro),
just without the Pre/Mid/Post breakdown. Positive gains bold; $^{*}$ = partial coverage (as in the main tables).
Usage: ../OLMo-Detect/.venv/bin/python analysis/fsd_gain_compact.py   -> tables/table:olmo_detect_overall_fsd.tex
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT
ROOT = Path(__file__).resolve().parents[1]; SIZES = PT.SIZES
BASES = [("PPL", "loss"), ("Zlib", "zlib"), ("Lower", "lowercase"), ("MinK", "mink")]   # MinK++ excluded = original FSD setting

def allv(spec, s):
    v, ok = PT.sup_stages(spec, "matched", s)["all"]; return v, ok

def sgn(v, star): return ("\\textbf{" if v > 0.0049 else "") + ("$+$" if v >= 0 else "$-$") + f"{abs(v):.2f}" + ("$^{*}$" if star else "") + ("}" if v > 0.0049 else "")
rows = []
for lab, b in BASES:
    base = {s: allv(("latest", "fsd", f"base_{b}"), s) for s in SIZES}
    ind = {s: allv(("latest", "fsd", f"fsd_{b}_score"), s) for s in SIZES}
    lodo = {s: allv(("lodo_fsd", "fsd", f"fsd_{b}_score"), s) for s in SIZES}
    lodo_base = {s: allv(("lodo_fsd", "fsd", f"base_{b}"), s) for s in SIZES}
    if any(base[s][0] is None or ind[s][0] is None for s in SIZES): print(f"[skip] {lab}: missing in-domain cells", file=sys.stderr); continue
    bavg = np.mean([base[s][0] for s in SIZES])
    gi = {s: (ind[s][0] - base[s][0], not (ind[s][1] and base[s][1])) for s in SIZES}
    gl = {s: ((lodo[s][0] - lodo_base[s][0]) if lodo[s][0] is not None and lodo_base[s][0] is not None else None, not (lodo[s][1] and lodo_base[s][1])) for s in SIZES}
    cells = [f"{bavg:.2f}"] + [sgn(*gi[s]) for s in SIZES] + [sgn(np.mean([gi[s][0] for s in SIZES]), any(gi[s][1] for s in SIZES))]
    glv = [gl[s][0] for s in SIZES if gl[s][0] is not None]
    cells += [("---" if gl[s][0] is None else sgn(*gl[s])) for s in SIZES] + [sgn(np.mean(glv), any(gl[s][1] for s in SIZES)) if glv else "---"]
    rows.append(f"\\textbf{{FSD ({lab})}} & " + " & ".join(cells) + " \\\\")
tex = ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lc|ccccc|ccccc}\n\\toprule\n"
       "\\multirow{2.5}{*}{\\textbf{Method}} & \\multirow{2.5}{*}{\\textbf{Base AUC}} & \\multicolumn{5}{c|}{\\textbf{In-domain $\\Delta$AUC}} & \\multicolumn{5}{c}{\\textbf{LODO $\\Delta$AUC}} \\\\\n"
       "\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}\n & & 1B & 7B & 13B & 32B & Model-Avg & 1B & 7B & 13B & 32B & Model-Avg \\\\\n\\midrule\n" + "\n".join(rows) +
       "\n\\bottomrule\n\\end{tabular}\n\\caption{AUC gain ($\\Delta$AUC) of FSD over its base detector on \\textsc{OLMo-Detect} under the \\textbf{In-domain} and \\textbf{LODO} settings, "
       "averaged over all subsets (all-stage AUC). \\textbf{Base AUC} is the base detector's own AUC averaged over model sizes. Positive gains are in bold; "
       "$^{*}$ marks cells with partial coverage. Per-stage and per-domain results are in Appendix~\\ref{appendix:auc_results_sup}.}\n\\label{table:olmo_detect_overall_fsd}\n\\end{table}\n")
out = ROOT / "tables/table:olmo_detect_overall_fsd.tex"; out.write_text(tex); print(tex); print("TEX ->", out)
