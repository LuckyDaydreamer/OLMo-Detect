"""Task-4 generalization table: 5 REP methods (8 scores) on OLMo-3 / DCLM / SmolLM2 models + OLMo-2 baselines, AUC on results_latest.
Usage: ../OLMo-Detect/.venv/bin/python analysis/t4_generalization.py -> markdown stdout + tables/table:generalization.tex + analysis/logs/t4_generalization.json"""
import sys, json, os
PAPER = os.environ.get("T4_PAPER") == "1"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import paper_tables as PT
ROOT = Path(__file__).resolve().parents[1]
COLS = [("PPL","loss_zlib_lowercase","loss_score"),("Zlib","loss_zlib_lowercase","zlib_score"),("Lower","loss_zlib_lowercase","lowercase_score"),
        ("MinK","minkprob","min_k_prob_mean_logprob"),("MinK++","minkprob","min_k_pp_score"),("DCPDD","dcpdd","dc_pdd_score"),("PAC","pac","pac_score"),("CAMIA","camia_faithful","camia_faithful_score")]
DOMS = {"GSM8K": ("member_midtraining_gsm8k","non-member_midtraining_gsm8k"),
        "CoT-GSM8K": ("member_posttraining_rlvr_cot-gsm8k","non-member_posttraining_rlvr_cot-gsm8k"),
        "DCLM": ("member_pretraining_dclm_matched","non-member_pretraining_dclm"),
        "CoT-MATH": ("member_posttraining_rlvr_cot-math","non-member_posttraining_rlvr_cot-math"),
        "OpenWebMath": ("member_pretraining_openwebmath_matched","non-member_pretraining_openwebmath")}
GROUPS = [("OLMo-2 (baseline)", ["OLMo2-1B-Instruct","OLMo2-7B-Instruct","OLMo2-13B-Instruct","OLMo2-32B-Instruct"], ["GSM8K","DCLM","OpenWebMath"] if PAPER else ["GSM8K","CoT-GSM8K","DCLM","OpenWebMath"]),
          ("OLMo-3", ["Olmo-3-1025-7B","Olmo-3-7B-Instruct","Olmo-3-7B-Think","Olmo-3-7B-RL-Zero-Math","Olmo-3.1-32B-Instruct"], ["GSM8K"]),
          ("DCLM", ["DCLM-1B","DCLM-Baseline-7B"], ["DCLM","OpenWebMath"]),
          ("SmolLM2", ["SmolLM2-1.7B"], ["DCLM"])]   # 25 Sep: HF SmolLM2-1.7B (DCLM-Baseline ~1 pass + FineWeb-Edu; NM absence from FineWeb-Edu to be verified)
SKIP = {("DCLM-Baseline-7B","DCLM")}   # 25 Sep: DCLM-7B saw an unpublished 2.5T subset of the DCLM pool -> members not guaranteed; only OpenWebMath is valid for it
if PAPER:
    COLS = [c for c in COLS if c[0] not in ("Lower","MinK++","CAMIA")]; GROUPS = [g for g in GROUPS if g[0] in ("OLMo-2 (baseline)","OLMo-3","DCLM","SmolLM2")]
    KEEP = {"OLMo-2 (baseline)": ("OLMo2-7B-Instruct","OLMo2-32B-Instruct","OLMo2-1B-Instruct"), "OLMo-3": ("Olmo-3-7B-Instruct","Olmo-3.1-32B-Instruct"), "DCLM": ("DCLM-1B","DCLM-Baseline-7B"), "SmolLM2": ("SmolLM2-1.7B",)}
    GROUPS = [(g, [m for m in ms if m in KEEP.get(g, ())], d) for g, ms, d in GROUPS]
out = {}; md = ["| Group | Model | Domain | " + " | ".join(c[0] for c in COLS) + " | Avg | n |", "|---|---|---|" + "---|" * (len(COLS) + 2)]; tex = []
for g, models, doms in GROUPS:
    for m in models:
        for d in doms:
            if (m, d) in SKIP: continue
            cl, ul = DOMS[d]; row = []
            for name, stem, col in COLS:
                c = PT.read(cl, m, stem, col); u = PT.read(ul, m, stem, col); row.append((PT.auc(c, u), len(c), len(u)))
            vals = [r[0] for r in row]; ok = [v for v in vals if v is not None]; avg = sum(ok) / len(ok) if ok else None
            n = max((r[1] for r in row), default=0)
            out[f"{m}|{d}"] = dict(group=g, aucs={c[0]: v for c, v in zip(COLS, vals)}, avg=avg, n_con=n)
            f = lambda v: "---" if v is None else f"{v:.2f}"
            md.append(f"| {g} | {m} | {d} | " + " | ".join(f(v) for v in vals) + f" | {f(avg)} | {n} |")
            tex.append(f"{g} & {m.replace('-Instruct','')} & {d} & " + " & ".join(f(v) for v in vals) + f" & {f(avg)} \\\\")
print("\n".join(md))
(ROOT / "analysis/logs/t4_generalization.json").write_text(json.dumps(out, indent=1))
latex = ("\\begin{table*}[t]\n\\centering\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lllccccccccc}\n\\toprule\nGroup & Model & Domain & " + " & ".join(c[0] for c in COLS) +
         " & Avg \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}}\n\\caption{AUCs of unsupervised MIAs on models beyond OLMo~2 (test partition). OLMo-2 rows are the baselines on the same domains.}\n\\label{table:generalization" + ("" if PAPER else "_all") + "}\n\\end{table*}\n")
fn = "table:generalization.tex" if PAPER else "table:generalization_all.tex"; (ROOT / "tables" / fn).write_text(latex); print("TEX ->", fn)
