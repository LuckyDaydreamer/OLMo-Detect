#!/usr/bin/env python3
"""Sup-CAMIA learning curve over ALL 13 subsets (results_camia_xdomain = representative subsets, results_camia_xdomain_alt = the other subsets).
Same protocol as analysis/lc_supcamia.py (fixed 50% eval half per class, nested training subsets 10-50%, 5 seeds). DPO = mean of chosen/rejected
(as in the main tables). Domain value = macro over its subsets; Macro-Avg = mean over the 9 domains. Output analysis/logs/lc_supcamia_full.json + appendix tex."""
import sys,os,json,pickle,numpy as np
sys.path.insert(0,"."); from camia_lr_xdomain import _fit_score_lr, union_concat, _auc
SIZES=["1b","7b","13b","32b"]; F=[0.1,0.2,0.3,0.4,0.5]; SEEDS=range(5)
# (row key, pkl dir, pkl domain key)
ROWS=[("dclm","results_camia_xdomain","dclm"),("pes2o","results_camia_xdomain","pes2o"),("openwebmath","results_camia_xdomain","openwebmath"),
      ("sc_python","results_camia_xdomain_alt","starcoder"),("sc_java","results_camia_xdomain","starcoder"),("gsm8k","results_camia_xdomain","gsm8k"),
      ("se_math","results_camia_xdomain_alt","stackexchange"),("se_stackoverflow","results_camia_xdomain","stackexchange"),
      ("sft_aya","results_camia_xdomain","sft"),("sft_wildchat","results_camia_xdomain_alt","sft"),
      ("dpo_chosen","results_camia_xdomain","dpo"),("dpo_rejected","results_camia_xdomain_alt","dpo"),
      ("rlvr_cot-gsm8k","results_camia_xdomain","rlvr"),("rlvr_cot-math","results_camia_xdomain_alt","rlvr")]
DOMS=[("DCLM",["dclm"]),("peS2o",["pes2o"]),("OWM",["openwebmath"]),("StarCoder",["sc_python","sc_java"]),("GSM8K",["gsm8k"]),
      ("StackExchange",["se_math","se_stackoverflow"]),("SFT",["sft_aya","sft_wildchat"]),("DPO",["dpo_chosen","dpo_rejected"]),("RLVR",["rlvr_cot-gsm8k","rlvr_cot-math"])]
sub=lambda d,idx:{k:np.asarray(v)[idx] for k,v in d.items()}
out={}; PK={}
if os.environ.get("LC_REUSE") and os.path.exists("analysis/logs/lc_supcamia_full.json"): out=json.load(open("analysis/logs/lc_supcamia_full.json"))
for S in ([] if out else SIZES):
    out[S]={}
    for key,dr,dk in ROWS:
        if (dr,S) not in PK: PK[(dr,S)]=pickle.load(open(f"{dr}/signals_{S}.pkl","rb"))
        P=PK[(dr,S)][dk]
        con=union_concat([P["con_dev"],P["con_test"]]); unc=union_concat([P["unc_dev"],P["unc_test"]])
        Nc=len(next(iter(con.values()))); Nu=len(next(iter(unc.values()))); res={f:[] for f in F}
        for s in SEEDS:
            rng=np.random.default_rng(1000+s); pc=rng.permutation(Nc); pu=rng.permutation(Nu)
            ec,tc=pc[:Nc//2],pc[Nc//2:]; eu,tu=pu[:Nu//2],pu[Nu//2:]
            target=union_concat([sub(con,ec),sub(unc,eu)]); lab=np.r_[np.ones(len(ec)),np.zeros(len(eu))]
            for f in F:
                nc=max(2,min(len(tc),int(round(f*Nc)))); nu=max(2,min(len(tu),int(round(f*Nu))))
                lr,_,_=_fit_score_lr(sub(con,tc[:nc]),sub(unc,tu[:nu]),target); res[f].append(float(_auc(lab,lr)))
        out[S][key]={"N_per_class":[Nc,Nu],"curve":{str(f):{"n_train_per_class":int(round(f*Nc)),"auc_mean":float(np.mean(res[f])),"auc_std":float(np.std(res[f]))} for f in F}}
        print(f"{S:3s} {key:16s} N={Nc:4d} "+" ".join(f"f{f}:{np.mean(res[f]):.3f}±{np.std(res[f]):.3f}" for f in F),flush=True)
if not os.environ.get("LC_REUSE"): json.dump(out,open("analysis/logs/lc_supcamia_full.json","w"),indent=1)
# table rows: 13 (DPO = mean of chosen/rejected), + Macro-Avg over the 9 domains (multi-subset domains macro'd first)
def row_val(S,key,f,what="auc_mean"):
    if key=="dpo": return float(np.mean([out[S][k]["curve"][str(f)][what] for k in ("dpo_chosen","dpo_rejected")]))
    return out[S][key]["curve"][str(f)][what]
def row_N(S,key): return out[S]["dpo_chosen" if key=="dpo" else key]["N_per_class"][0]
TAB=[("DCLM-Baseline",["dclm"]),("peS2o",["pes2o"]),("OpenWebMath",["openwebmath"]),("StarCoder",["sc_python","sc_java"]),("Dolmino Math Mix",["gsm8k"]),
     ("Stack Exchange",["se_math","se_stackoverflow"]),("SFT",["sft_aya","sft_wildchat"]),("DPO",["dpo"]),("RLVR",["rlvr_cot-gsm8k","rlvr_cot-math"])]
def dom_val(S,ks,f,what="auc_mean"): return float(np.mean([row_val(S,k,f,what) for k in ks]))
def macro(S,f): return float(np.mean([dom_val(S,ks,f) for _,ks in TAB]))
Nvar={n:sorted(set(sum(row_N(S,k) for k in ks) for S in SIZES)) for n,ks in TAB}
L=["\\begin{table*}[t]","\\centering","\\resizebox{\\textwidth}{!}{%","\\begin{tabular}{@{}l r|ccccc|ccccc|ccccc|ccccc@{}}","\\toprule",
   "\\multirow{2}{*}{\\textbf{Domain}} & \\multirow{2}{*}{$N$} & "+" & ".join(f"\\multicolumn{{5}}{{c{'|' if S!='32b' else ''}}}{{\\textbf{{{S.upper()}}}}}" for S in SIZES)+" \\\\",
   "\\cmidrule(lr){3-7} \\cmidrule(lr){8-12} \\cmidrule(lr){13-17} \\cmidrule(lr){18-22}",
   " & & "+" & ".join(" & ".join(f"{int(f*100)}\\%" for f in F) for S in SIZES)+" \\\\","\\midrule"]
for name,ks in TAB:
    nv=Nvar[name]; nstr=str(nv[0]) if len(nv)==1 else f"{min(nv)}--{max(nv)}"
    L.append(f"{name} & {nstr} & "+" & ".join(" & ".join(f"{dom_val(S,ks,f):.2f}" for f in F) for S in SIZES)+" \\\\")
L+=["\\midrule","\\textbf{Macro-Avg} & & "+" & ".join(" & ".join(f"\\textbf{{{macro(S,f):.2f}}}" for f in F) for S in SIZES)+" \\\\","\\bottomrule","\\end{tabular}%","}"]
ALLK=[k for _,ks in TAB for k in ks]
mx=max(row_val(S,k,f,"auc_std") for S in SIZES for k in ALLK for f in F)
nlow=sum(1 for S in SIZES for k in ALLK for f in F if row_val(S,k,f,"auc_std")<0.05); ntot=len(SIZES)*len(ALLK)*len(F)
mx3=max(row_val(S,k,f,"auc_std") for S in SIZES for k in ALLK for f in F if f>=0.3)
cap=(f"AUC of Sup-CAMIA as a function of the labeled-data budget on \\textsc{{OLMo-Detect}}. For each subset and model size, half of the members and half of the non-members are held out as a fixed evaluation set; Sup-CAMIA is trained on nested subsets of the remaining half containing 10\\%--50\\% of the subset's members and non-members (10\\% equals the dev-set budget of the main tables). Entries are the mean AUC on the held-out half over five random splits, averaged over the domain's subsets (DPO over the chosen and rejected responses); $N$ is the number of members (equal to the number of non-members) in the domain. SFT and DPO members come from the size-specific post-training mixes (SFT: one mix shared by 1B and 32B and another by 7B and 13B; DPO: one preference mix per size), so their $N$ varies across model sizes and is given as a range. Macro-Avg is the mean over the nine domains. At the subset level the standard deviation across splits is below 0.05 in {nlow} of {ntot} cases; it reaches {mx:.2f} for the smallest subsets (at most 100 members) at the 10\\%--20\\% budgets and is at most {mx3:.2f} at budgets of 30\\% or more. Absolute values are not comparable to Table~\\ref{{table:olmo_detect_overall_supervised}}, which uses the 90\\% test split.")
L+=["\\caption{"+cap+"}","\\label{table:lc_supcamia}","\\end{table*}"]
open("../ICLR2027_OLMo-Detect/tables/table:lc_supcamia.tex","w").write("\n".join(L)+"\n"); print("WRITTEN tables_latest/table:lc_supcamia.tex; max std",round(mx,3))
print("MACRO:",{S:[round(macro(S,f),3) for f in F] for S in SIZES})
