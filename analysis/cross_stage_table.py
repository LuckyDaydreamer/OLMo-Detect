#!/usr/bin/env python3
"""Cross-stage table (tables_latest/table:cross_stage.tex) from analysis/logs/cross_stage_matrix.json (18 MIAs x 4 sizes x 13 subsets, matched split).
Top: per-subset AUC (size-avg, then MIA-avg) for all MIAs / effective MIAs, dup counts. Bottom: stage aggregates + math P-S vs others with a paired
one-sided Wilcoxon test over MIAs (each MIA's size-averaged AUC, mean over the 3 math P-S subsets minus mean over the other 10)."""
import json,numpy as np
from scipy.stats import wilcoxon
M=json.load(open("analysis/logs/cross_stage_matrix.json")); names=M["names"]; KEYS=M["keys"]; STRUCT=M["struct"]; SIZES=["1b","7b","13b","32b"]
A={tuple(k.split("|")):v for k,v in M["auc"].items()}
EXCL={"FSD","SelfCrit"}   # 2026-09-26 (user caption): FSD is reported as a gain over its base detector, SelfCrit covers only 2 domains
names=[n for n in names if n not in EXCL]
EFF=[n for n in ["PPL","Zlib","Lower","MinK","DCPDD","PAC","CAMIA","Sup-CAMIA","FSD"] if n not in EXCL]; MATH=["gsm8k","rlvr_cot-gsm8k","rlvr_cot-math"]; OTH=[k for k in KEYS if k not in MATH]
DUP={"dclm":"9.00$^\\ddagger$","pes2o":"1.12$^\\ddagger$","openwebmath":"1.00","sc_python":"1.00","sc_java":"1.00","gsm8k":"6.00","se_math":"6.00","se_stackoverflow":"6.00",
     "sft_aya":"2.09","sft_wildchat":"2.00","dpo":"1.13","rlvr_cot-gsm8k":"1.25","rlvr_cot-math":"1.50"}   # 25 Sep: SFT/DPO/RLVR from analysis/dupcount_combine.py (Non_DCLM + post-training indexes); ddagger = old estimate, needs DCLM + Dolmino indexes
def m_sub(n,k):  # size-averaged AUC of MIA n on subset k (None if no size available)
    v=[A[(n,s,k)] for s in SIZES if (n,s,k) in A]; return float(np.mean(v)) if v else None
def sub_auc(ms,k): return float(np.mean([x for n in ms for x in [m_sub(n,k)] if x is not None]))
def dom_keys(): return [(stg,keys) for stg,dom,keys in STRUCT]
def stage_auc(ms,stg,excl=()): return float(np.mean([np.mean([sub_auc(ms,k) for k in keys if k not in excl]) for st,keys in dom_keys() if st==stg and any(k not in excl for k in keys)]))
def contrast(ms):
    d=[]
    for n in ms:
        a=[m_sub(n,k) for k in MATH]; b=[m_sub(n,k) for k in OTH]
        if any(x is None for x in a): continue
        d.append(np.mean(a)-np.mean([x for x in b if x is not None]))
    d=np.array(d); return float(d.mean()),int((d>0).sum()),len(d),float(wilcoxon(d,alternative="greater").pvalue)
def group_auc(ms,ks): return float(np.mean([sub_auc(ms,k) for k in ks]))
f=lambda x:(f"\\textbf{{{x:.2f}}}" if x>0.7 else f"{x:.2f}")
STG={"pre":"Pre-training","mid":"Mid-training","post":"Post-training"}
res={"all":{},"eff":{}}
for tag,ms in (("all",names),("eff",EFF)):
    res[tag]={"subset":{k:sub_auc(ms,k) for k in KEYS},"stage":{s:stage_auc(ms,s) for s in ("pre","mid","post")},"stage_wo":{s:stage_auc(ms,s,MATH) for s in ("pre","mid","post")},"math":group_auc(ms,MATH),"others":group_auc(ms,OTH),"contrast":contrast(ms)}
json.dump(res,open("analysis/logs/cross_stage_final.json","w"),indent=1)
L=["\\begin{table*}[t]","\\centering","\\resizebox{\\textwidth}{!}{%","\\begin{tabular}{@{}l ccccccccccccc@{}}","\\toprule",
   "\\textbf{Stage} & \\multicolumn{5}{c}{\\textbf{Pre-training}} & \\multicolumn{3}{c}{\\textbf{Mid-training}} & \\multicolumn{5}{c}{\\textbf{Post-training}} \\\\",
   "\\cmidrule(lr){2-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-14}",
   "\\textbf{Domain} & \\multirow{2.5}{*}{DCLM} & \\multirow{2.5}{*}{peS2o} & \\multirow{2.5}{*}{OpenWebMath} & \\multicolumn{2}{c}{StarCoder} & Math Mix & \\multicolumn{2}{c}{Stack Exchange} & \\multicolumn{2}{c}{SFT} & DPO & \\multicolumn{2}{c}{RLVR} \\\\",
   "\\cmidrule(lr){5-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12} \\cmidrule(lr){13-14}",
   "\\textbf{Subset} & & & & Python & Java & GSM8K & Math SE & SO & Aya & WildChat & UF-WildChat & CoT-GSM8K & CoT-MATH \\\\","\\midrule",
   "\\textbf{AUC} (all MIAs) & "+" & ".join(f(res["all"]["subset"][k]) for k in KEYS)+" \\\\",
   "\\textbf{AUC} (eff.\\ MIAs) & "+" & ".join(f(res["eff"]["subset"][k]) for k in KEYS)+" \\\\",
   "\\textit{Dup.\\ count} & "+" & ".join(DUP[k] for k in KEYS)+" \\\\","\\bottomrule","\\end{tabular}%","}","\\vspace{4pt}","",
   "\\resizebox{0.8\\textwidth}{!}{%  <- change the factor to scale the second part","\\begin{tabular}{@{}l ccc|ccc|ccc@{}}","\\toprule",
   " & \\multicolumn{3}{c|}{\\textbf{Stage (all subsets)}} & \\multicolumn{3}{c|}{\\textbf{Stage (w/o math P--S)}} & \\multicolumn{3}{c}{\\textbf{Math P--S vs.\\ others}} \\\\","\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
   " & Pre & Mid & Post & Pre & Mid & Post & Math P--S & Others & $\\Delta$ (\\#MIAs, $p$) \\\\","\\midrule"]
for tag,lab in (("all","all MIAs"),("eff","eff.\\ MIAs")):
    r=res[tag]; d,npos,ntot,p=r["contrast"]
    L.append(f"\\textbf{{AUC}} ({lab}) & {r['stage']['pre']:.2f} & {r['stage']['mid']:.2f} & {r['stage']['post']:.2f} & {r['stage_wo']['pre']:.2f} & {r['stage_wo']['mid']:.2f} & {r['stage_wo']['post']:.2f} & {r['math']:.2f} & {r['others']:.2f} & $+{d:.2f}$ ({npos}/{ntot}, {p:.3f}) \\\\")
L+=["\\bottomrule","\\end{tabular}%","}"]
cap=("Cross-stage analysis on \\textsc{OLMo-Detect}. \\textbf{Top:} Per-subset AUCs averaged across model sizes and over all 18 MIAs or the \\textit{effective (eff.)} subset (overall AUC $\\geq 0.60$); bold indicates AUC $>0.70$. \\textit{Dup.\\ count}: average number of occurrences of each subset's members in the training corpus. Math SE: Math Stack Exchange; SO: Stack Overflow; UF-WildChat: UltraFeedback-WildChat. "
     "\\textbf{Bottom:} Stage-level AUCs with and without the three math problem--solution (P--S) subsets (GSM8K, CoT-GSM8K, and CoT-MATH), and AUCs on the P--S subsets versus the other ten. $\\Delta$: average per-MIA AUC gain on P--S subsets over the others (MIAs missing a P--S subset are excluded); parentheses report the number of MIAs with positive gains and the one-sided Wilcoxon $p$-value.")
L+=["\\caption{"+cap+"}","\\label{table:cross_stage}","\\end{table*}"]
open("tables/table:cross_stage.tex","w").write("\n".join(L)+"\n")
print(json.dumps({t:{"stage":{k:round(v,3) for k,v in res[t]["stage"].items()},"math":round(res[t]["math"],3),"others":round(res[t]["others"],3),"contrast":res[t]["contrast"]} for t in res},indent=0))
