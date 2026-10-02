"""Combined supervised table (cross-stage style): TOP = Sup-CAMIA / MIA-Tuner AUCs (In-domain, LODO) from tables_latest/table:olmo_detect_overall_supervised.tex,
BOTTOM = FSD gain block from tables_latest/table:olmo_detect_overall_fsd.tex; one caption/label. Run after regen_tables.sh.
-> tables/table:olmo_detect_overall_supervised_combined.tex"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; T = ROOT / "tables"
def tabular(p):
    s = (T / p).read_text(); m = re.search(r"(\\begin\{tabular\}.*?\\end\{tabular\})", s, re.S); return m.group(1)
top = tabular("table:olmo_detect_overall_supervised.tex").replace("\\multirow{2}{*}{\\textbf{Method}}","\\multirow{2.5}{*}{\\textbf{Method}}").replace("\\multirow{2}{*}{\\textbf{Setting}}","\\multirow{2.5}{*}{\\textbf{Setting}}"); bot = tabular("table:olmo_detect_overall_fsd.tex")
tex = ("\\begin{table*}[t]\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{0.95\\textwidth}{!}{%\n" + top + "%\n}\n\n\\vspace{4pt}\n\n"
       "\\resizebox{0.65\\textwidth}{!}{%  <- change the factor to scale the second part\n" + bot + "%\n}\n"
       "\\caption{Supervised MIAs on \\textsc{OLMo-Detect} under the \\textbf{In-domain} and \\textbf{LODO} settings. \\textbf{Top:} AUCs of Sup-CAMIA and MIA-Tuner; "
       "the best method per column is in bold. \\textbf{Bottom:} AUC gain ($\\Delta$AUC) of FSD over its base detector, averaged over all domains; \\textbf{Base AUC} is the base "
       "detector's AUC averaged over model sizes, and positive gains are in bold. Per-stage and per-domain results are in Appendix~\\ref{appendix:auc_results_sup}.}\n"
       "\\label{table:olmo_detect_overall_supervised}\n\\end{table*}\n")
out = T / "table:olmo_detect_overall_supervised_combined.tex"; out.write_text(tex); print(tex); print("TEX ->", out)
