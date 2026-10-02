#!/usr/bin/env python
"""Paper tables from FINAL scores (ICLR2027_OLMo-Detect/results_latest; EMMIA from results_iclr/emmia_raw; SelfCrit DPO from the
combined DPO dirs). MACRO-averaging: subset -> domain (mean of subsets) -> stage (mean of domains) -> All (mean of Pre/Mid/Post);
Model-Avg = mean over sizes; Method-Avg = mean over method rows with a value. IFEval excluded (dropped from the benchmark).
A cell gets a star when any leaf feeding it is not fully scored on the CURRENT benchmark test set (missing file or partial
coverage); values are computed from what is available. Usage: paper_tables.py main_unsup [--split matched|shifted] [--tex out.tex]"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
import config
import json,math,glob,sys,argparse,hashlib
from pathlib import Path
from sklearn.metrics import roc_auc_score
B=config.BENCHMARK
RL=config.RESULTS; RI=RL; EM=RL/"emmia"
SIZES=["1b","7b","13b","32b"]; SD={s:f"OLMo2-{s.upper()}-Instruct" for s in SIZES}
def nlines(globs):
    """number of UNIQUE texts (consolidation dedups by text; StarCoder matched members contain duplicate texts)"""
    t=set()
    for g in globs:
        for p in glob.glob(str(g)):
            for l in open(p):
                if l.strip():
                    r=json.loads(l); t.add(r.get("text") or (r.get("chosen_text","")+"\x1f"+r.get("rejected_text","")))
    return len(t)
_bn={}
def bench_n(kind,*a):
    """benchmark TEST size for a leaf"""
    key=(kind,)+a
    if key in _bn: return _bn[key]
    if kind=="unc_pre": g=[B/f"pretraining/{a[0]}/non-member/test/*.jsonl"]
    elif kind=="con_pre": g=[B/f"pretraining/{a[0]}/member/{a[1]}/test/*.jsonl"]
    elif kind=="unc_sc": g=[B/f"pretraining/starcoder/non-member/test/*{a[0]}*.jsonl"]
    elif kind=="con_sc": g=[B/f"pretraining/starcoder/member/{a[1]}/test/*{a[0]}*.jsonl"]
    elif kind=="unc_gsm": g=[B/"midtraining/gsm8k/non-member/test/*.jsonl"]
    elif kind=="con_gsm": g=[B/"midtraining/gsm8k/member/test/*.jsonl"]
    elif kind=="unc_se": g=[B/f"midtraining/stackexchange/non-member/test/*{a[0]}*.jsonl"]
    elif kind=="con_se": g=[B/f"midtraining/stackexchange/member/{a[1]}/test/*{a[0]}*.jsonl"]
    elif kind=="unc_rlvr": g=[B/f"posttraining/rlvr/non-member/test/*{a[0]}*.jsonl"]
    elif kind=="con_rlvr": g=[B/f"posttraining/rlvr/member/test/*{a[0]}*.jsonl"]
    elif kind=="unc_sft": g=[B/f"posttraining/sft/{a[0]}/non-member/test/*{a[1]}*.jsonl"]
    elif kind=="con_sft": g=[B/f"posttraining/sft/{a[0]}/member/{a[2]}/test/*{a[1]}*.jsonl"]
    elif kind=="unc_dpo": g=[B/f"posttraining/dpo/non-member/test/*_dpo_{a[0]}_{a[1]}_test.jsonl"]
    elif kind=="con_dpo": g=[B/f"posttraining/dpo/member/{a[2]}/test/*_dpo_{a[0]}_{a[1]}_{a[2]}_test.jsonl"]
    elif kind=="unc_dpo_pair": g=[B/f"posttraining/dpo/non-member/test/*_dpo_{a[0]}_pair_test.jsonl"]
    elif kind=="con_dpo_pair": g=[B/f"posttraining/dpo/member/{a[1]}/test/*_dpo_{a[0]}_pair_{a[1]}_test.jsonl"]
    else: raise ValueError(kind)
    _bn[key]=nlines(g); return _bn[key]
def valid(v):
    try: x=float(v); return not math.isnan(x)
    except Exception: return False
def read(leaf,sd,stem,col,dirs=(RL,)):
    for D in dirs:
        p=D/leaf/sd/f"{stem}.json"
        if p.exists():
            d=json.loads(p.read_text()); return [float(s[col]) for s in d.get("result",{}).get("samples",[]) if valid(s.get(col))]
    return []
def text_key(t):
    """join key for a raw benchmark text"""
    return hashlib.sha1((t or "").strip().encode("utf-8")).hexdigest()[:20]
def rec_key(s):
    """Join key for one scored record. Freshly scored results carry the document text;
    the released scores carry only its hash, the text itself being in benchmark/."""
    t=s.get("text")
    return text_key(t) if t else (s.get("text_sha1") or "")
METRIC="auc"   # "auc" | "tpr"  (TPR@5% FPR; convention = AUC_TPR@5%FPR.py: np.interp on roc_curve, degenerate -> 0.05)
MET={"auc":dict(name="AUC",main="table:olmo_detect_overall_unsupervised",lab=lambda l:f"table:{l}"),
     "tpr":dict(name="TPR@5\\% FPR",main="table:tpr_unsup:all",lab=lambda l:"table:tpr_unsup:"+l.replace("auc_","",1))}
def auc(con,unc):
    if not con or not unc: return None
    y=[1]*len(con)+[0]*len(unc)
    if METRIC=="auc": return float(roc_auc_score(y,con+unc))
    import numpy as _np; from sklearn.metrics import roc_curve as _rc
    sc=con+unc
    if len(set(sc))<2: return 0.05
    fpr,tpr,_=_rc(y,sc); return float(_np.interp(0.05,fpr,tpr))
# ---------- leaves ----------
# each leaf(split,size) -> list of (con_leafdir, unc_leafdir, con_N, unc_N)  (DPO pools chosen+rejected)
def leaves(dom_key,split,size):
    m="matched" if split=="matched" else "shifted"
    if dom_key in ("dclm","pes2o","openwebmath"):
        return [((f"member_pretraining_{dom_key}_{m}",bench_n("con_pre",dom_key,m)),(f"non-member_pretraining_{dom_key}",bench_n("unc_pre",dom_key)))]
    if dom_key.startswith("sc_"):
        sub=dom_key[3:]; return [((f"member_pretraining_starcoder_{sub}_{m}",bench_n("con_sc",sub,m)),(f"non-member_pretraining_starcoder_{sub}",bench_n("unc_sc",sub)))]
    if dom_key=="gsm8k":
        if split!="matched": return None
        return [(("member_midtraining_gsm8k",bench_n("con_gsm")),("non-member_midtraining_gsm8k",bench_n("unc_gsm")))]
    if dom_key.startswith("se_"):
        sub=dom_key[3:]; return [((f"member_midtraining_stackexchange_{sub}_{m}",bench_n("con_se",sub,m)),(f"non-member_midtraining_stackexchange_{sub}",bench_n("unc_se",sub)))]
    if dom_key.startswith("rlvr_"):
        if split!="matched": return None
        sub=dom_key[5:]; return [((f"member_posttraining_rlvr_{sub}",bench_n("con_rlvr",sub)),(f"non-member_posttraining_rlvr_{sub}",bench_n("unc_rlvr",sub)))]
    if dom_key.startswith("sft_"):
        sub=dom_key[4:]; pair="1b-32b" if size in ("1b","32b") else "7b-13b"
        return [((f"member_posttraining_sft_{pair}_{sub}_{m}",bench_n("con_sft",pair,sub,m)),(f"non-member_posttraining_sft_{pair}_{sub}",bench_n("unc_sft",pair,sub)))]
    if dom_key=="dpo":
        return [((f"member_posttraining_dpo_{size}_{side}_{m}",bench_n("con_dpo",size,side,m)),(f"non-member_posttraining_dpo_{size}_{side}",bench_n("unc_dpo",size,side))) for side in ("chosen","rejected")]
    raise ValueError(dom_key)
STRUCT=[("pre","DCLM",["dclm"]),("pre","peS2o",["pes2o"]),("pre","OpenWebMath",["openwebmath"]),("pre","StarCoder",["sc_python","sc_java"]),
        ("mid","GSM8K",["gsm8k"]),("mid","StackExchange",["se_math","se_stackoverflow"]),
        ("post","SFT",["sft_aya","sft_wildchat"]),("post","DPO",["dpo"]),("post","RLVR",["rlvr_cot-gsm8k","rlvr_cot-math"])]
METHODS=[("PPL","loss_zlib_lowercase","loss_score"),("Zlib","loss_zlib_lowercase","zlib_score"),("Lower","loss_zlib_lowercase","lowercase_score"),
 ("MinK","minkprob","min_k_prob_mean_logprob"),("MinK++","minkprob","min_k_pp_score"),("DCPDD","dcpdd","dc_pdd_score"),("ReCaLL","recall","recall_score"),
 ("CAMIA","camia_faithful","camia_faithful_score"),("PAC","pac","pac_score"),("Neigh.","neighborhood_attack","score"),("EMMIA","__emmia__",None),
 ("DCQ","dcq","score"),("CDD","cdd","peak"),("Guided","guided_instruction","score"),("SelfCrit","selfcrit","self_critique_score")]
SELFCRIT_DOMS={"DPO","RLVR"}
EMMIA_INITS=("loss","zlib","mink","minkpp")
_ei={}
def emmia_init(size):
    """EMMIA base score selected on the pooled MATCHED dev split (analysis/emmia_dev_select.py).
    One init per size; the shifted split reuses the matched-selected value, as for MinK's k.
    Set EMMIA_MAX_ON_TEST=1 to restore the pre-2026-09-28 max-over-inits-on-the-evaluated-set rule."""
    import os
    if os.environ.get("EMMIA_MAX_ON_TEST"): return None
    if not _ei:
        p=Path(__file__).resolve().parents[1]/"analysis/logs/emmia_dev_select.json"
        if p.exists():
            sel=json.loads(p.read_text()).get("selection",{})
            for s,v in sel.items(): _ei[s]=v.get("matched_dev_pick")
        _ei.setdefault("__loaded__",True)
    return _ei.get(size)
def emmia_tag(con_leaf):
    t=con_leaf.replace("member_","",1)
    if "_dpo_" in t: return t   # posttraining_dpo_1b_matched_chosen
    if t.startswith("posttraining_rlvr_") or t.startswith("midtraining_gsm8k"): return t+"_matched" if not t.endswith("_matched") else t
    return t if t.endswith(("_matched","_shifted")) else t+"_matched"
def subset_auc(name,stem,col,dom_key,split,size):
    """-> (auc or None, complete:bool, note)"""
    lv=leaves(dom_key,split,size)
    if lv is None: return None,True,"n/a-split"
    sd=SD[size]
    if stem=="__emmia__":
        samples=[]; ok=True
        for (cl,cn),(ul,un) in lv:
            p=EM/emmia_tag(cl)/f"emmia_{size}.json"
            if not p.exists() and (EM/(emmia_tag(cl)+"_sub300")/f"emmia_{size}.json").exists(): p=EM/(emmia_tag(cl)+"_sub300")/f"emmia_{size}.json"   # 2026-09-25 SO shifted subsample
            if not p.exists(): ok=False; continue
            ss=json.loads(p.read_text()).get("samples",[]); samples+=ss
            if sum(1 for s in ss if int(s["label"])==1)!=cn or sum(1 for s in ss if int(s["label"])==0)!=un: ok=False
        if not samples: return None,False,"emmia missing"
        y=[int(s["label"]) for s in samples]
        if len(set(y))<2: return None,False,"emmia one-class"
        k=emmia_init(size)   # 2026-09-28: init selected on the pooled DEV split, not max over the evaluated set
        if k is None:        # no dev selection available -> old rule, flagged
            a=max(auc([float(s["final_scores"][kk] ) for s in samples if int(s["label"])==1],[float(s["final_scores"][kk]) for s in samples if int(s["label"])==0]) for kk in EMMIA_INITS)
            return a,ok,"emmia max-on-test (no dev selection)"
        a=auc([float(s["final_scores"][k]) for s in samples if int(s["label"])==1],[float(s["final_scores"][k]) for s in samples if int(s["label"])==0])
        return a,ok,""
    if stem=="selfcrit" and dom_key=="dpo":   # combined DPO dirs (results_iclr), pair-level instances
        m="matched" if split=="matched" else "shifted"
        con=read(f"member_posttraining_dpo_{size}_pair_{m}",sd,stem,col,dirs=(RL,RI)); unc=read(f"non-member_posttraining_dpo_{size}_pair",sd,stem,col,dirs=(RL,RI))
        ok=len(con)==bench_n("con_dpo_pair",size,m) and len(unc)==bench_n("unc_dpo_pair",size)
        return auc(con,unc),ok,""
    con=[];unc=[];ok=True
    for (cl,cn),(ul,un) in lv:
        c=read(cl,sd,stem,col,dirs=(RL,RI) if stem=="selfcrit" else (RL,)); u=read(ul,sd,stem,col,dirs=(RL,RI) if stem=="selfcrit" else (RL,))
        if len(c)!=cn or len(u)!=un: ok=False
        # vintage check: member and non-member files from runs >5 days apart = different numeric setup (e.g. Sep-12 4-bit vs fp16 overwrite) -> provisional
        pc=RI/cl/sd/f"run_manifest_{stem}.json"; pu=RI/ul/sd/f"run_manifest_{stem}.json"   # manifests: not rewritten by patch merges (23 Sep)
        if not pc.exists(): pc=RI/cl/sd/f"{stem}.json"
        if not pu.exists(): pu=RI/ul/sd/f"{stem}.json"
        if pc.exists() and pu.exists():
            def _num(pth):
                try:
                    import ast as _ast; c=json.loads(pth.read_text()).get("config",{}); c=_ast.literal_eval(c) if isinstance(c,str) else c
                    return tuple(sorted((k,str(v)) for k,v in c.items() if any(t in k.lower() for t in ("bit","quant","dtype","fp16","precision"))))
                except Exception: return None
            nc,nu=_num(pc),_num(pu)
            if nc and nu: ok = ok and (nc==nu)                       # numeric setup recorded: compare it
            elif size=="32b" and abs(pc.stat().st_mtime-pu.stat().st_mtime)>5*86400: ok=False   # else age rule only where 4-bit vs fp16 runs coexist
        con+=c; unc+=u
    return auc(con,unc),ok,""
def build(split):
    """-> table[(method,size)] = {stage:(val,complete)}, plus notes"""
    T={}; notes=[]
    for name,stem,col in METHODS:
        for s in SIZES:
            row={}
            for st in ("pre","mid","post"):
                doms=[]; comp=True
                for stg,dom,keys in STRUCT:
                    if stg!=st: continue
                    if name=="SelfCrit" and dom not in SELFCRIT_DOMS: continue
                    vals=[]
                    for k in keys:
                        a,ok,note=subset_auc(name,stem,col,k,split,s)
                        if note=="n/a-split": continue
                        if a is None: comp=False; notes.append(f"{name} {s} {dom}/{k}: no data"); continue
                        if not ok: comp=False; notes.append(f"{name} {s} {dom}/{k}: partial")
                        vals.append(a)
                    if vals: doms.append(sum(vals)/len(vals))
                row[st]=(sum(doms)/len(doms) if doms else None,comp)
            allv=[v for v,_ in row.values() if v is not None]
            row["all"]=(sum(allv)/len(allv) if allv else None, all(c for _,c in row.values()))
            T[(name,s)]=row
    return T,notes
def fmt(v,c,bold=False):
    if v is None: return "---"
    s=f"{v:.2f}"; s=f"\\textbf{{{s}}}" if bold else s
    return s+("$^{*}$" if not c else "")
def main_unsup(split,tex=None):
    T,notes=build(split); cols=["pre","mid","post","all"]
    # model-avg per method
    MA={}
    for name,_,_ in METHODS:
        MA[name]={}
        for c in cols:
            vs=[T[(name,s)][c] for s in SIZES if T[(name,s)][c][0] is not None]
            MA[name][c]=(sum(v for v,_ in vs)/len(vs), all(k for _,k in vs)) if vs else (None,True)
    best={s:max((T[(n,s)]["all"][0] or -1,n) for n,_,_ in METHODS)[1] for s in SIZES}; best["MA"]=max((MA[n]["all"][0] or -1,n) for n,_,_ in METHODS)[1]
    lines=[]
    for name,_,_ in METHODS:
        cells=[]
        for s in SIZES:
            for c in cols: v,k=T[(name,s)][c]; cells.append(fmt(v,k,bold=(c=="all" and best[s]==name)))
        for c in cols: v,k=MA[name][c]; cells.append(fmt(v,k,bold=(c=="all" and best["MA"]==name)))
        lab=f"\\textbf{{{name}}}"+("$^\\dagger$" if name=="SelfCrit" else "")
        lines.append(lab+" & "+" & ".join(cells)+" \\\\")
    # method-avg
    cells=[]
    for s in SIZES+["MA"]:
        for c in cols:
            vs=[(T[(n,s)][c] if s!="MA" else MA[n][c]) for n,_,_ in METHODS]; vs=[x for x in vs if x[0] is not None]
            cells.append(fmt(sum(v for v,_ in vs)/len(vs), all(k for _,k in vs)) if vs else "---")
    lines.append("\\midrule\n\\textbf{Method-Avg} & "+" & ".join(cells)+" \\\\")
    body="\n".join(lines)
    out=body
    if tex: Path(tex).write_text(body+"\n")
    print(body); print("\n% incomplete cells (leaf-level):"); 
    for n in sorted(set(notes)): print("%  "+n)
# ---------------- Appendix I: per-domain / per-subset AUC tables ----------------
DOM_TABLES=[("DCLM-Baseline","auc_dclm",[("dclm","")]),("peS2o","auc_pes2o",[("pes2o","")]),("StarCoder","auc_starcoder",[("sc_python","Python"),("sc_java","Java")]),
 ("OpenWebMath","auc_openwebmath",[("openwebmath","")]),("GSM8K","auc_gsm8k",[("gsm8k","")]),("Stack Exchange","auc_stackexchange",[("se_math","Math Stack Exchange"),("se_stackoverflow","Stack Overflow")]),
 ("SFT","auc_sft",[("sft_aya","Aya"),("sft_wildchat","WildChat")]),("DPO","auc_dpo",[("dpo","")]),("RLVR","auc_rlvr",[("rlvr_cot-gsm8k","CoT-GSM8K"),("rlvr_cot-math","CoT-MATH")])]
def cell_vals(keys,split):
    """-> {(method,size): (auc,complete)} macro over the given subset keys"""
    out={}
    for name,stem,col in METHODS:
        for s in SIZES:
            vals=[];comp=True;any_=False
            for k in keys:
                a,ok,note=subset_auc(name,stem,col,k,split,s)
                if note=="n/a-split": continue
                any_=True
                if a is None: comp=False; continue
                if not ok: comp=False
                vals.append(a)
            out[(name,s)]=(sum(vals)/len(vals) if vals else None, comp) if any_ else (None,True)
    return out
def block_rows(cellsets,methods):
    """cellsets: list of {(m,s):(v,c)} (one per column group). -> rows of latex cells with bold best per column"""
    rows={m:[] for m in methods}; 
    for cs in cellsets:
        MA={m:None for m in methods}
        for m in methods:
            vs=[cs[(m,s)] for s in SIZES if cs[(m,s)][0] is not None]
            MA[m]=(sum(v for v,_ in vs)/len(vs), all(c for _,c in vs)) if vs else (None,True)
        best={s:max(((cs[(m,s)][0] if cs[(m,s)][0] is not None else -1),m) for m in methods)[1] for s in SIZES}; best["MA"]=max(((MA[m][0] if MA[m][0] is not None else -1),m) for m in methods)[1]
        for m in methods:
            for s in SIZES: v,c=cs[(m,s)]; rows[m].append(fmt(v,c,bold=(best[s]==m and v is not None)))
            v,c=MA[m]; rows[m].append(fmt(v,c,bold=(best["MA"]==m and v is not None)))
        # method-avg
        avg=[]
        for s in SIZES+["MA"]:
            vs=[(cs[(m,s)] if s!="MA" else MA[m]) for m in methods]; vs=[x for x in vs if x[0] is not None]
            avg.append(fmt(sum(v for v,_ in vs)/len(vs), all(c for _,c in vs)) if vs else "---")
        rows.setdefault("__avg__",[]).extend(avg)
    return rows
STAR_NOTE=""   # provisional-star legend removed from captions (user edit 22 Sep); stars stay in cells until runs finish
def appendix_domains(split="matched",tex=None):
    out=[]
    for dom,label,subs in DOM_TABLES:
        methods=[m for m,_,_ in METHODS if (m!="SelfCrit" or dom in SELFCRIT_DOMS)]
        cs=cell_vals([k for k,_ in subs],split); rows=block_rows([cs],methods)
        body="\n".join(f"\\textbf{{{m}}}"+("$^\\dagger$" if m=="SelfCrit" else "")+" & "+" & ".join(rows[m])+" \\\\" for m in methods)
        body+="\n\\midrule\n\\textbf{Method-Avg} & "+" & ".join(rows["__avg__"])+" \\\\"
        dag=" $^\\dagger$SelfCrit is evaluated only on DPO and RLVR." if dom in SELFCRIT_DOMS else ""
        out.append(f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{0.7\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l|cccc|c}}
\\toprule
\\textbf{{Method}} & \\textbf{{1B}} & \\textbf{{7B}} & \\textbf{{13B}} & \\textbf{{32B}} & \\textbf{{Model-Avg}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Comparison of unsupervised MIAs on the {dom} domain of \\textsc{{OLMo-Detect}}, evaluated via AUC{' (macro-averaged over its subsets)' if len(subs)>1 else ''}. \\textbf{{Model-Avg}} averages across model sizes, while \\textbf{{Method-Avg}} averages the methods with a value in that column. The best-performing method in each column is highlighted in bold.{dag}{STAR_NOTE}}}
\\label{{table:{label}}}
\\end{{table*}}
""")
    for dom,label,subs in DOM_TABLES:
        if len(subs)<2: continue
        methods=[m for m,_,_ in METHODS if (m!="SelfCrit" or dom in SELFCRIT_DOMS)]
        rows=block_rows([cell_vals([k],split) for k,_ in subs],methods)
        body="\n".join(f"\\textbf{{{m}}}"+("$^\\dagger$" if m=="SelfCrit" else "")+" & "+" & ".join(rows[m])+" \\\\" for m in methods)
        body+="\n\\midrule\n\\textbf{Method-Avg} & "+" & ".join(rows["__avg__"])+" \\\\"
        n=len(subs); colspec="l|"+"|".join(["ccccc"]*n)
        head=" & ".join(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{sl}}}}}" for _,sl in subs)
        cm=" ".join(f"\\cmidrule(lr){{{2+5*i}-{6+5*i}}}" for i in range(n))
        sub=" & ".join(["1B & 7B & 13B & 32B & Model-Avg"]*n)
        dag=" $^\\dagger$SelfCrit is evaluated only on DPO and RLVR." if dom in SELFCRIT_DOMS else ""
        out.append(f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{colspec}}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & {head} \\\\
{cm}
 & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Comparison of unsupervised MIAs on the subsets of the {dom} domain of \\textsc{{OLMo-Detect}}, evaluated via AUC. \\textbf{{Model-Avg}} averages across model sizes, while \\textbf{{Method-Avg}} averages the methods with a value in that column. The best-performing method in each column is highlighted in bold.{dag}{STAR_NOTE}}}
\\label{{table:{label}_subsets}}
\\end{{table*}}
""")
    txt="\n".join(out)
    if tex: Path(tex).write_text(txt)
    print(txt)


def appendix_stages(split="matched",tex=None):
    """Stage table in the main-table layout, grouped by stage: Pre | Mid | Post | All, each with 1B/7B/13B/32B/Model-Avg."""
    T,notes=build(split); methods=[m for m,_,_ in METHODS]
    cellsets=[{(m,s):T[(m,s)][st] for m in methods for s in SIZES} for st in ("pre","mid","post","all")]
    rows=block_rows(cellsets,methods)
    def row(m):
        lab="\\textbf{"+m+"}"+("$^\\dagger$" if m=="SelfCrit" else "")
        cells=rows[m]; groups=[" & ".join(cells[i*5:(i+1)*5]) for i in range(4)]
        return lab+" & "+" & ".join(groups)+" \\\\"
    body="\n".join(row(m) for m in methods)
    avg=rows["__avg__"]; body+="\n\\midrule\n\\textbf{Method-Avg} & "+" & ".join(" & ".join(avg[i*5:(i+1)*5]) for i in range(4))+" \\\\"
    head=" & ".join(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{sl}}}}}" for sl in ("Pre-training","Mid-training","Post-training","All"))
    cm=" ".join(f"\\cmidrule(lr){{{2+5*i}-{6+5*i}}}" for i in range(4))
    sub=" & ".join(["1B & 7B & 13B & 32B & Model-Avg"]*4)
    txt=f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l|ccccc|ccccc|ccccc|ccccc}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & {head} \\\\
{cm}
 & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Comparison of unsupervised MIAs on \\textsc{{OLMo-Detect}} by training stage, evaluated via AUC. \\textbf{{Pre-training}}, \\textbf{{Mid-training}}, and \\textbf{{Post-training}} report macro-averaged AUCs over the domains of each stage (subset AUCs are first averaged within each domain), while \\textbf{{All}} macro-averages the three stages; these are the \\textbf{{Pre}}/\\textbf{{Mid}}/\\textbf{{Post}}/\\textbf{{\\textit{{All}}}} values of Table~\\ref{{table:olmo_detect_overall_unsupervised}} regrouped by stage. \\textbf{{Model-Avg}} averages across model sizes, while \\textbf{{Method-Avg}} averages the methods with a value in that column. The best-performing method in each column is highlighted in bold. $^\\dagger$SelfCrit is evaluated only on DPO and RLVR and thus has only \\textbf{{Post-training}} and \\textbf{{All}} values.{STAR_NOTE}}}
\\label{{table:auc_stages}}
\\end{{table*}}
"""
    if tex: Path(tex).write_text(txt)
    print(txt)

# ---------------- Supervised overall table (in-domain vs LODO) ----------------
RM=config.LODO_MIA_TUNER; RF=config.LODO_FSD; RX=config.LODO_CAMIA
FSD_BASES=[("PPL","loss"),("Zlib","zlib"),("Lower","lowercase"),("MinK","mink"),("MinK++","minkpp")]
XDOM={"DCLM":"dclm","peS2o":"pes2o","OpenWebMath":"openwebmath","StarCoder":"starcoder","GSM8K":"gsm8k","StackExchange":"stackexchange","SFT":"sft","DPO":"dpo","RLVR":"rlvr"}
def sup_subset_auc(spec,dom_key,split,size):
    """spec: ('latest'|'lodo_mt'|'lodo_fsd', stem, col) -> (auc, complete)"""
    src,stem,col=spec; lv=leaves(dom_key,split,size)
    if lv is None: return None,True
    sd=SD[size]; dirs={"latest":(RL,),"lodo_mt":(RM,),"lodo_fsd":(RF,)}[src]
    con=[];unc=[];ok=True
    for (cl,cn),(ul,un) in lv:
        c=read(cl,sd,stem,col,dirs=dirs); u=read(ul,sd,stem,col,dirs=dirs)
        if len(c)!=cn or len(u)!=un: ok=False
        con+=c; unc+=u
    return auc(con,unc),ok
def sup_stages(spec,split,size):
    """{stage:(val,complete)} incl. 'all' for a supervised spec (macro subset->domain->stage->all)"""
    out={}
    for st in ("pre","mid","post"):
        doms=[];comp=True
        for stg,dom,keys in STRUCT:
            if stg!=st: continue
            vals=[]
            for k in keys:
                a,ok=sup_subset_auc(spec,k,split,size)
                if a is None: comp=False; continue
                if not ok: comp=False
                vals.append(a)
            if vals: doms.append(sum(vals)/len(vals))
        out[st]=(sum(doms)/len(doms) if doms else None,comp)
    v=[x for x,_ in out.values() if x is not None]; out["all"]=(sum(v)/len(v) if v else None, all(c for _,c in out.values()))
    return out
def camia_lodo_stages(size):
    p=RX/f"camia_lr_xdomain_{size}.json"; out={}
    if not p.exists():
        for st in ("pre","mid","post","all"): out[st]=(None,False)
        return out
    pd=json.loads(p.read_text()).get("per_domain",{})
    for st,doms in [("pre",["DCLM","peS2o","OpenWebMath","StarCoder"]),("mid",["GSM8K","StackExchange"]),("post",["SFT","DPO","RLVR"])]:
        v=[pd[XDOM[x]]["lodo_lr"] for x in doms if XDOM[x] in pd]; out[st]=(sum(v)/len(v) if v else None, len(v)==len(doms))
    v=[x for x,_ in out.values() if x is not None]; out["all"]=(sum(v)/len(v) if v else None, all(c for _,c in out.values()))
    return out
def main_sup(split="matched",tex=None,compact=True,gains=True,ref=True,label=None,delta=False):
    STG=["all"] if compact else ["pre","mid","post","all"]
    # rows: (method label, setting label, {size:{stage:(v,c)}}, {size:{stage:base}} or None)
    rows=[]
    ref_rows={s:sup_stages(("latest","loss_zlib_lowercase","zlib_score"),split,s) for s in SIZES}
    if not compact and ref: rows.append(("\\textit{Zlib (unsup.)}","---",ref_rows,None))
    rows.append(("Sup-CAMIA","In-domain",{s:sup_stages(("latest","camia_lr","camia_lr_score"),split,s) for s in SIZES},None))
    rows.append(("Sup-CAMIA","LODO",{s:camia_lodo_stages(s) for s in SIZES},None))
    rows.append(("MIA-Tuner","In-domain",{s:sup_stages(("latest","mia_tuner","mia_tuner_score"),split,s) for s in SIZES},None))
    rows.append(("MIA-Tuner","LODO",{s:sup_stages(("lodo_mt","mia_tuner","mia_tuner_score"),split,s) for s in SIZES},None))
    for lab,b in FSD_BASES:
        for setting,src in [("In-domain","latest"),("LODO","lodo_fsd")]:
            v={s:sup_stages((src,"fsd",f"fsd_{b}_score"),split,s) for s in SIZES}; base={s:sup_stages((src,"fsd",f"base_{b}"),split,s) for s in SIZES}
            for s in SIZES:
                for st in STG:
                    x,c=v[s][st]; v[s][st]=(x,c and base[s][st][1])
            rows.append((f"FSD ({lab})",setting,v,{s:{st:base[s][st][0] for st in STG} for s in SIZES}))
    def ma(vals):
        v=[x for x in vals if x[0] is not None]; return (sum(x[0] for x in v)/len(v), all(x[1] for x in v)) if v else (None,True)
    def cell(row,s,st):
        v=row[2]; base=row[3]
        if s=="MA":
            x=ma([v[z][st] for z in SIZES]); b=ma([(base[z][st],True) for z in SIZES])[0] if base else None
        else: x=v[s][st]; b=base[s][st] if base else None
        return x[0],x[1],b
    cols=[(s,st) for s in SIZES+["MA"] for st in STG]
    best={}   # best per column WITHIN each setting (in-domain rows vs in-domain rows, LODO vs LODO)
    for s,st in cols:
        for setting in ("In-domain","LODO"):
            best[(s,st,setting)]=max(((cell(r,s,st)[0] if cell(r,s,st)[0] is not None else -1),i) for i,r in enumerate(rows) if r[1]==setting)[1]
    lines=[]; prev=None
    for i,row in enumerate(rows):
        cells=[]
        for s,st in cols:
            x,c,b=cell(row,s,st)
            if x is None: cells.append("---"); continue
            if delta and b is not None:
                t=f"{x-b:+.2f}".replace("-0.00","+0.00").replace("-","$-$").replace("+","$+$")
                if not c: t+="$^{*}$"
                cells.append(t); continue
            t=f"{x:.2f}"
            if row[1]!="---" and best[(s,st,row[1])]==i: t=f"\\textbf{{{t}}}"
            sub=(f"{x-b:+.2f}".replace("-0.00","+0.00") if (b is not None and not compact and gains) else ""); sup="*" if not c else ""
            if sub or sup: t+="$"+(f"_{{{sub}}}" if sub else "")+(f"^{{{sup}}}" if sup else "")+"$"
            cells.append(t)
        lab=row[0] if row[0]!=prev else ""
        if row[0]!=prev and row[1]!="---": lab=f"\\multirow{{2}}{{*}}{{\\textbf{{{row[0]}}}}}"
        if row[0]!=prev and prev is not None: lines.append("\\midrule" if row[0].startswith(("Sup","MIA")) or prev.startswith(("Sup","MIA","Zlib")) else ("\\cmidrule(lr){1-7}" if compact else "\\cmidrule(lr){1-22}"))
        if delta and row[0].startswith("FSD") and not (prev or "").startswith("FSD"):
            ncol=7 if compact else 22
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{FSD: gain in AUC over its base detector ($\\Delta$AUC; base AUCs in Table~\\ref{{table:olmo_detect_overall_unsupervised}}, absolute values in Table~\\ref{{table:app:supervised_stages}})}}}} \\\\")
        lines.append(f"{lab} & {row[1]} & "+" & ".join(cells)+" \\\\"); prev=row[0]
    body="\n".join(lines)
    if compact:
        txt=f"""\\begin{{table}}[t]
\\centering
\\resizebox{{0.72\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|cccc|c}}
\\toprule
\\textbf{{Method}} & \\textbf{{Setting}} & \\textbf{{1B}} & \\textbf{{7B}} & \\textbf{{13B}} & \\textbf{{32B}} & \\textbf{{Model-Avg}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Comparison of supervised MIAs on \\textsc{{OLMo-Detect}}, evaluated via AUC macro-averaged over all stages (the \\textbf{{\\textit{{All}}}} aggregation of Table~\\ref{{table:olmo_detect_overall_unsupervised}}), under the \\textbf{{In-domain}} setting (trained on the development set of the evaluated domain) and the \\textbf{{LODO}} setting (leave-one-domain-out: trained on the development sets of all other domains). FSD is built on top of an existing detector, given in parentheses. \\textbf{{Model-Avg}} averages across model sizes. Within each setting, the best-performing method in each column is highlighted in bold. Per-stage results and the gains of FSD over its base detectors are reported in Table~\\ref{{table:app:supervised_stages}}.{STAR_NOTE}}}
\\label{{table:olmo_detect_overall_supervised}}
\\end{{table}}
"""
    else:
        head=" & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{s}}}}}" for s in ["1B","7B","13B","32B","Model-Avg"])
        cm=" ".join(f"\\cmidrule(lr){{{3+4*i}-{6+4*i}}}" for i in range(5))
        sub=" & ".join(["Pre & Mid & Post & All"]*5)
        if label=="main":
            cap=("Comparison of supervised MIAs on \\textsc{OLMo-Detect}, evaluated via AUC, under the \\textbf{In-domain} setting (trained on the development set of the "
                 "evaluated domain) and the \\textbf{LODO} setting (leave-one-domain-out: trained on the development sets of all other domains). \\textbf{Pre}/\\textbf{Mid}/"
                 "\\textbf{Post}/\\textbf{\\textit{All}} and \\textbf{Model-Avg} are aggregated exactly as in Table~\\ref{table:olmo_detect_overall_unsupervised}. FSD is built on top "
                 "of an existing detector, given in parentheses; "+("its rows report the gain in AUC over that base detector measured in the same run (absolute values in Table~\\ref{table:app:supervised_stages})" if delta else "its gains over the base detectors are reported in Table~\\ref{table:app:supervised_stages}")+". Within each setting, the "
                 "best-performing method in each column is highlighted in bold"+(" (Sup-CAMIA and MIA-Tuner rows)" if delta else "")+"."+STAR_NOTE)
            lab="table:olmo_detect_overall_supervised"
        else:
            cap=("Per-stage comparison of supervised MIAs on \\textsc{OLMo-Detect} under the \\textbf{In-domain} and \\textbf{LODO} settings, evaluated via AUC. \\textbf{Pre}/"
                 "\\textbf{Mid}/\\textbf{Post}/\\textbf{\\textit{All}} and \\textbf{Model-Avg} are aggregated exactly as in Table~\\ref{table:olmo_detect_overall_unsupervised}. FSD is "
                 "built on top of an existing detector (given in parentheses); the subscript reports its gain over that base detector's own AUC measured in the same run. The first "
                 "row repeats the best unsupervised method, which requires no training, as a reference. Within each setting, the best-performing entry in each column is highlighted in bold."+STAR_NOTE)
            lab="table:app:supervised_stages"
        txt=f"""\\begin{{table*}}[t]
\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|cccc|cccc|cccc|cccc|cccc}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Setting}}}} & {head} \\\\
{cm}
 & & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{{cap}}}
\\label{{{lab}}}
\\end{{table*}}
"""
    if tex: Path(tex).write_text(txt)
    print(txt)

# ---------------- Supervised: two tables (Sup-CAMIA + MIA-Tuner absolute AUC; FSD as gain over base) ----------------
def sup_two_tables(split="matched",tex=None):
    STG=["pre","mid","post","all"]; cols=[(s,st) for s in SIZES+["MA"] for st in STG]
    def ma(vals):
        v=[x for x in vals if x[0] is not None]; return (sum(x[0] for x in v)/len(v), all(x[1] for x in v)) if v else (None,True)
    def render(rows,delta):
        """rows: (label, setting, {size:{st:(v,c)}}, base or None) -> body lines; bold = best (largest) per column within setting"""
        def cell(row,s,st):
            v,base=row[2],row[3]
            if s=="MA":
                x=ma([v[z][st] for z in SIZES]); b=ma([(base[z][st],True) for z in SIZES])[0] if base else None
            else: x=v[s][st]; b=base[s][st] if base else None
            val=(x[0]-b) if (delta and x[0] is not None and b is not None) else x[0]
            return val,x[1]
        best={}
        for s,st in cols:
            for setting in ("In-domain","LODO"):
                best[(s,st,setting)]=max(((cell(r,s,st)[0] if cell(r,s,st)[0] is not None else -9),i) for i,r in enumerate(rows) if r[1]==setting)[1]
        lines=[];prev=None
        for i,row in enumerate(rows):
            cells=[]
            for s,st in cols:
                x,c=cell(row,s,st)
                if x is None: cells.append("---"); continue
                t=(f"{x:+.2f}".replace("-0.00","+0.00").replace("-","$-$").replace("+","$+$")) if delta else f"{x:.2f}"
                if (delta and x>0.005) or (not delta and best[(s,st,row[1])]==i): t=f"\\textbf{{{t}}}"
                if not c: t+="$^{*}$"
                cells.append(t)
            if row[0]!=prev and prev is not None: lines.append("\\midrule")
            lab=f"\\multirow{{2}}{{*}}{{\\textbf{{{row[0]}}}}}" if row[0]!=prev else ""
            lines.append(f"{lab} & {row[1]} & "+" & ".join(cells)+" \\\\"); prev=row[0]
        return "\n".join(lines)
    head=" & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{s}}}}}" for s in ["1B","7B","13B","32B","Model-Avg"])
    cm=" ".join(f"\\cmidrule(lr){{{3+4*i}-{6+4*i}}}" for i in range(5)); sub=" & ".join(["Pre & Mid & Post & All"]*5)
    def wrap(body,cap,lab):
        return f"""\\begin{{table*}}[t]
\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|cccc|cccc|cccc|cccc|cccc}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Setting}}}} & {head} \\\\
{cm}
 & & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{{cap}}}
\\label{{{lab}}}
\\end{{table*}}
"""
    # Table 1
    rows=[("Sup-CAMIA","In-domain",{s:sup_stages(("latest","camia_lr","camia_lr_score"),split,s) for s in SIZES},None),
          ("Sup-CAMIA","LODO",{s:camia_lodo_stages(s) for s in SIZES},None),
          ("MIA-Tuner","In-domain",{s:sup_stages(("latest","mia_tuner","mia_tuner_score"),split,s) for s in SIZES},None),
          ("MIA-Tuner","LODO",{s:sup_stages(("lodo_mt","mia_tuner","mia_tuner_score"),split,s) for s in SIZES},None)]
    t1=wrap(render(rows,False),
        "Comparison of Sup-CAMIA and MIA-Tuner on \\textsc{OLMo-Detect}, evaluated via AUC, under the \\textbf{In-domain} setting (trained on the dev set of the evaluated domain) and the \\textbf{LODO} setting (leave-one-domain-out: trained on the dev sets of all other domains). \\textbf{Pre}/\\textbf{Mid}/\\textbf{Post}/\\textbf{\\textit{All}}, \\textbf{Model-Avg}, and the highlighting rule follow Table~\\ref{table:olmo_detect_overall_unsupervised} (best per column within each setting). FSD is reported separately in Table~\\ref{table:olmo_detect_overall_fsd}."+STAR_NOTE,
        "table:olmo_detect_overall_supervised")
    # Table 2 (FSD as gain)
    rows=[]
    for lab,b in FSD_BASES:
        for setting,src in [("In-domain","latest"),("LODO","lodo_fsd")]:
            v={s:sup_stages((src,"fsd",f"fsd_{b}_score"),split,s) for s in SIZES}; base={s:sup_stages((src,"fsd",f"base_{b}"),split,s) for s in SIZES}
            for s in SIZES:
                for st in STG: x,c=v[s][st]; v[s][st]=(x,c and base[s][st][1])
            rows.append((f"FSD ({lab})",setting,v,{s:{st:base[s][st][0] for st in STG} for s in SIZES}))
    t2=wrap(render(rows,True),
        "Gain in AUC ($\\Delta$AUC) of FSD over its base detector on \\textsc{OLMo-Detect}, under the \\textbf{In-domain} setting (fine-tuned on the non-members of the evaluated domain's dev set) and the \\textbf{LODO} setting (fine-tuned on the non-members of all other domains' dev sets). Each row applies FSD on top of the base detector given in parentheses; the gain is measured against that base detector's own AUC in the same run (absolute values are reported in Table~\\ref{table:app:supervised_stages}). \\textbf{Pre}/\\textbf{Mid}/\\textbf{Post}/\\textbf{\\textit{All}} and \\textbf{Model-Avg} follow Table~\\ref{table:olmo_detect_overall_unsupervised}; positive gains, i.e., cases where FSD improves on its base detector, are highlighted in bold."+STAR_NOTE,
        "table:olmo_detect_overall_fsd")
    txt=t1+"\n"+t2
    if tex: Path(tex).write_text(txt)
    print(txt)


# ---------------- Main table, GROUPED layout (methods as columns by family, size x stage rows) — added 22 Sep 2026 ----------------
FAMILIES=[("Likelihood-based",["PPL","Zlib","Lower","MinK","MinK++","DCPDD","PAC","ReCaLL","EMMIA","CAMIA"]),
          ("Pert.",["Neigh."]),("Prompt",["Guided","DCQ"]),("Out.-Dist.",["CDD","SelfCrit"])]   # 22 Sep FINAL: PAC stays likelihood (Min-K% extension), Neigh.=perturbation, DCQ=prompt
def main_unsup_grouped(split="matched",tex=None):
    T,notes=build(split); cols=["pre","mid","post","all"]; order=[m for _,ms in FAMILIES for m in ms]
    MA={}
    for name in order:
        MA[name]={}
        for c in cols:
            vs=[T[(name,s)][c] for s in SIZES if T[(name,s)][c][0] is not None]
            MA[name][c]=(sum(v for v,_ in vs)/len(vs), all(k for _,k in vs)) if vs else (None,True)
    def get(s,name,c): return T[(name,s)][c] if s!="MA" else MA[name][c]
    def row(s,c,label):
        vals={n:get(s,n,c) for n in order}; have=[n for n in order if vals[n][0] is not None]
        best=max(have,key=lambda n:vals[n][0]) if have else None
        cells=[fmt(vals[n][0],vals[n][1],bold=(n==best)) for n in order]
        cells.append(fmt(sum(vals[n][0] for n in have)/len(have), all(vals[n][1] for n in have)) if have else "---")
        return f" & {label} & "+" & ".join(cells)+" \\\\"
    nm=len(order); ncol=nm+3
    spec="@{}l@{\;\;}l@{\;\;\;}"+"c"*nm+"@{\;\;\;}c@{}"
    hdr1=["\\multirow{2}{*}{\\textbf{Model}}","\\multirow{2}{*}{\\textbf{Stage}}"]; rules=[]; col=3
    for fam,ms in FAMILIES:
        hdr1.append(f"\\multicolumn{{{len(ms)}}}{{c}}{{\\textbf{{{fam}}}}}" if len(ms)>1 else f"\\textbf{{{fam}}}")
        rules.append(f"\\cmidrule(lr){{{col}-{col+len(ms)-1}}}"); col+=len(ms)
    hdr1.append("\\multirow{2}{*}{\\makecell{\\textbf{Method-}\\\\\\textbf{Avg}}}")
    hdr2=" & & "+" & ".join(f"\\textbf{{{m}}}"+("$^\\dagger$" if m=="SelfCrit" else "") for m in order)+" & \\\\"
    L=["\\begin{table*}[t]","\\centering","\\resizebox{0.99\\textwidth}{!}{%",f"\\begin{{tabular}}{{{spec}}}","\\toprule",
       " & ".join(hdr1)+" \\\\"," ".join(rules),hdr2,"\\midrule"]
    blocks=[(s,s.upper()) for s in SIZES]+[("MA","\\makecell{\\textbf{Model-}\\\\\\textbf{Avg}}")]
    for i,(s,lab) in enumerate(blocks):
        if s=="MA": L.append("\\midrule")
        L.append(f"\\multirow{{4}}{{*}}{{{lab}}}")
        L.append(row(s,"pre","Pre ")); L.append(row(s,"mid","Mid ")); L.append(row(s,"post","Post")+f" \\cmidrule(l){{2-{ncol}}}")
        L.append(row(s,"all","All "))
        if s!="MA" and i<len(SIZES)-1: L.append(f"\\cmidrule(l){{1-{ncol}}}")
    L+=["\\bottomrule","\\end{tabular}%","}",
        "\\caption{Comparison of unsupervised MIAs on \\textsc{OLMo-Detect}, evaluated via "+MET[METRIC]["name"]+". Methods are grouped by family: likelihood-based, "
        "perturbation-based (\\textbf{Pert.}), prompt-based and output-distribution-based (\\textbf{Out.-Dist.}). \\textbf{Pre}/\\textbf{Mid}/\\textbf{Post} report macro-averaged AUCs "
        "over the domains of each stage, while \\textbf{All} macro-averages the three stages. \\textbf{Model-Avg} averages across model sizes, while "
        "\\textbf{Method-Avg} averages the methods with a value in that row. The best-performing method in each row is highlighted in bold. "
        "$^\\dagger$SelfCrit is evaluated only on DPO and RLVR and thus has only \\textbf{Post} and \\textbf{All} values, averaged over these two domains. "
        "Per-domain AUC results are provided in Appendix~\\ref{appendix:auc_results_unsup}.}",
        "\\label{"+MET[METRIC]["main"]+"}","\\end{table*}"]
    out="\n".join(L)+"\n"
    if tex: Path(tex).write_text(out)
    print(out); print("% incomplete cells (leaf-level):")
    for n in sorted(set(notes)): print("%  "+n)


def main_unsup_transposed(split="matched",tex=None,caption=None):
    """Compact main table: rows = methods grouped by family (family column + vertical rule, no horizontal family rules); columns = size x stage + Model-Avg."""
    T,notes=build(split); cols=["pre","mid","post","all"]; order=[m for _,ms in FAMILIES for m in ms]
    MA={}
    for name in order:
        MA[name]={}
        for c in cols:
            vs=[T[(name,s)][c] for s in SIZES if T[(name,s)][c][0] is not None]
            MA[name][c]=(sum(v for v,_ in vs)/len(vs), all(k for _,k in vs)) if vs else (None,True)
    def get(s,name,c): return T[(name,s)][c] if s!="MA" else MA[name][c]
    groups=[(s,s.upper()) for s in SIZES]+[("MA","Model-Avg")]
    colkeys=[(s,c) for s,_ in groups for c in cols]
    best={}
    for s,c in colkeys:
        have=[n for n in order if get(s,n,c)[0] is not None]; best[(s,c)]=max(have,key=lambda n:get(s,n,c)[0]) if have else None
    spec="@{}l|"+"|".join("cccc" for _ in groups)+"@{}"
    hdr1="\\multirow{2}{*}{\\textbf{Method}} & "+" & ".join(f"\\multicolumn{{4}}{{c{'|' if i<len(groups)-1 else ''}}}{{\\textbf{{{lab}}}}}" for i,(_,lab) in enumerate(groups))+" \\\\"
    rules=" ".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(len(groups)))
    hdr2=" & "+" & ".join("\\textbf{Pre} & \\textbf{Mid} & \\textbf{Post} & \\textbf{All}" for _ in groups)+" \\\\"
    L=["\\begin{table*}[t]","\\centering","\\resizebox{0.99\\textwidth}{!}{%",f"\\begin{{tabular}}{{{spec}}}","\\toprule",hdr1,rules,hdr2,"\\midrule"]
    MARK={"Likelihood-based":"L","Pert.":"P","Prompt":"Pr","Out.-Dist.":"O"}   # superscript family markers (legend in caption)
    ncol=1+4*len(groups)
    for fi,(fam,ms) in enumerate(FAMILIES):
        for m in ms:
            cells=[fmt(get(s,m,c)[0],get(s,m,c)[1],bold=(best[(s,c)]==m)) for s,c in colkeys]
            sup=MARK[fam]+("\\dagger" if m=="SelfCrit" else "")
            L.append(f"{m}$^{{\\mathrm{{{MARK[fam]}}}"+("\\dagger" if m=="SelfCrit" else "")+"}$ & "+" & ".join(cells)+" \\\\")
        if fi<len(FAMILIES)-1: L.append(f"\\cmidrule(lr){{1-{ncol}}}")
    L.append("\\midrule")
    avg=[]
    for s,c in colkeys:
        vs=[get(s,n,c) for n in order if get(s,n,c)[0] is not None]
        avg.append(fmt(sum(v for v,_ in vs)/len(vs), all(k for _,k in vs)) if vs else "---")
    L.append("\\textbf{Method-Avg} & "+" & ".join(avg)+" \\\\")
    cap=caption or ("Comparison of unsupervised MIAs on \\textsc{OLMo-Detect}, evaluated via "+MET[METRIC]["name"]+".")
    L+=["\\bottomrule","\\end{tabular}%","}","\\caption{"+cap+"}","\\label{"+MET[METRIC]["main"]+"}","\\end{table*}"]
    out="\n".join(L)+"\n"
    if tex: Path(tex).write_text(out)
    print(out); print("% incomplete cells (leaf-level):")
    for n in sorted(set(notes)): print("%  "+n)

# ---------------- Appendix I, GROUPED layout (family-grouped method columns; rows = sizes, or subset x sizes) — added 22 Sep 2026 ----------------
def _grouped_cols(methods):
    """-> (order, header_line1_cells, cmidrules, header_line2_cells) for the families restricted to `methods`; first data column index = c0"""
    fams=[(f,[m for m in ms if m in methods]) for f,ms in FAMILIES]; fams=[(f,ms) for f,ms in fams if ms]
    order=[m for _,ms in fams for m in ms]
    return fams,order
def _grouped_rows(cs,order,label_cells,ncol,avg_label="\\textit{Avg}"):
    """cs: {(m,s):(v,c)}; label_cells: dict s-> leading cells string. -> list of latex rows for SIZES + Model-Avg (bold = best per row)"""
    MA={}
    for m in order:
        vs=[cs[(m,s)] for s in SIZES if cs[(m,s)][0] is not None]
        MA[m]=(sum(v for v,_ in vs)/len(vs), all(c for _,c in vs)) if vs else (None,True)
    L=[]
    for s in SIZES+["MA"]:
        vals={m:(cs[(m,s)] if s!="MA" else MA[m]) for m in order}; have=[m for m in order if vals[m][0] is not None]
        best=max(have,key=lambda m:vals[m][0]) if have else None
        cells=[fmt(vals[m][0],vals[m][1],bold=(m==best)) for m in order]
        cells.append(fmt(sum(vals[m][0] for m in have)/len(have), all(vals[m][1] for m in have)) if have else "---")
        if s=="MA": L.append(f"\\cmidrule(l){{{2 if '&' not in label_cells['MA'] else 2}-{ncol}}}")
        L.append(label_cells[s]+" & "+" & ".join(cells)+" \\\\")
    return L
def appendix_domains_grouped(split="matched",tex=None):
    out=[]
    def header(fams,order,lead):
        h1=[f"\\multirow{{2}}{{*}}{{\\textbf{{{x}}}}}" for x in lead]; rules=[]; col=len(lead)+1
        for f,ms in fams:
            h1.append(f"\\multicolumn{{{len(ms)}}}{{c}}{{\\textbf{{{f}}}}}" if len(ms)>1 else f"\\textbf{{{f}}}")
            rules.append(f"\\cmidrule(lr){{{col}-{col+len(ms)-1}}}"); col+=len(ms)
        h1.append("\\multirow{2}{*}{\\makecell{\\textbf{Method-}\\\\\\textbf{Avg}}}")
        h2=" & "*len(lead)+" & ".join(f"\\textbf{{{m}}}"+("$^\\dagger$" if m=="SelfCrit" else "") for m in order)+" & \\\\"
        return " & ".join(h1)+" \\\\", " ".join(rules), h2
    for dom,label,subs in DOM_TABLES:
        methods=[m for m,_,_ in METHODS if (m!="SelfCrit" or dom in SELFCRIT_DOMS)]
        fams,order=_grouped_cols(methods); ncol=len(order)+2
        cs=cell_vals([k for k,_ in subs],split)
        lab={s:f"\\textbf{{{s.upper()}}}" for s in SIZES}; lab["MA"]="\\textbf{Model-Avg}"
        body="\n".join(_grouped_rows(cs,order,lab,ncol))
        h1,rules,h2=header(fams,order,["Model"])
        dag=" $^\\dagger$SelfCrit is evaluated only on DPO and RLVR." if dom in SELFCRIT_DOMS else ""
        spec="@{}l@{\;\;\;}"+"c"*len(order)+"@{\;\;\;}c@{}"
        out.append(f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{0.99\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{spec}}}
\\toprule
{h1}
{rules}
{h2}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Comparison of unsupervised MIAs on the {dom} domain of \\textsc{{OLMo-Detect}}, evaluated via {MET[METRIC]['name']}{' (macro-averaged over its subsets)' if len(subs)>1 else ''}. Methods are grouped by family as in Table~\\ref{{{MET[METRIC]['main']}}}. \\textbf{{Model-Avg}} averages across model sizes, while \\textbf{{Method-Avg}} averages the methods with a value in that row. The best-performing method in each row is highlighted in bold.{dag}}}
\\label{{{MET[METRIC]['lab'](label)}}}
\\end{{table*}}
""")
    for dom,label,subs in DOM_TABLES:
        if len(subs)<2: continue
        methods=[m for m,_,_ in METHODS if (m!="SelfCrit" or dom in SELFCRIT_DOMS)]
        fams,order=_grouped_cols(methods); ncol=len(order)+3
        blocks=[]
        for i,(k,sl) in enumerate(subs):
            cs=cell_vals([k],split)
            lab={s:(f"\\multirow{{5}}{{*}}{{\\textbf{{{sl}}}}} & {s.upper()}" if s==SIZES[0] else f" & {s.upper()}") for s in SIZES}; lab["MA"]=" & \\textit{Model-Avg}"
            rows=_grouped_rows(cs,order,lab,ncol)
            rows=[r.replace(f"\\cmidrule(l){{2-{ncol}}}",f"\\cmidrule(l){{2-{ncol}}}") for r in rows]
            blocks.append("\n".join(rows))
        body=f"\n\\cmidrule(l){{1-{ncol}}}\n".join(blocks)
        h1,rules,h2=header(fams,order,["Subset","Model"])
        dag=" $^\\dagger$SelfCrit is evaluated only on DPO and RLVR." if dom in SELFCRIT_DOMS else ""
        spec="@{}l@{\;\;}l@{\;\;\;}"+"c"*len(order)+"@{\;\;\;}c@{}"
        out.append(f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{0.99\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{spec}}}
\\toprule
{h1}
{rules}
{h2}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Comparison of unsupervised MIAs on the subsets of the {dom} domain of \\textsc{{OLMo-Detect}}, evaluated via {MET[METRIC]['name']}. Methods are grouped by family as in Table~\\ref{{{MET[METRIC]['main']}}}. \\textbf{{Model-Avg}} averages across model sizes, while \\textbf{{Method-Avg}} averages the methods with a value in that row. The best-performing method in each row is highlighted in bold.{dag}}}
\\label{{{MET[METRIC]['lab'](label)}_subsets}}
\\end{{table*}}
""")
    txt="\n".join(out)
    if tex: Path(tex).write_text(txt)
    print(txt)


# ---------------- No-CoT ablation (RLVR): CoT vs no-CoT variants of GSM8K and MATH, grouped layout — added 22 Sep 2026 ----------------
NOCOT=[("GSM8K","rlvr_cot-gsm8k","rlvr_gsm8k_no_cot"),("MATH","rlvr_cot-math","rlvr_math_no_cot")]
def nocot_ablation(split="matched",tex=None):
    methods=[m for m,_,_ in METHODS]; fams,order=_grouped_cols(methods); ncol=len(order)+4
    def fmtd(d,c):
        if d is None: return "---"
        if abs(d)<0.005: d=0.0
        return f"{d:+.2f}"+("$^{*}$" if not c else "")
    blocks=[]
    for sub,kc,kn in NOCOT:
        cc=cell_vals([kc],split); cn=cell_vals([kn],split)
        MAc={};MAn={}
        for m in order:
            for cs,MA in ((cc,MAc),(cn,MAn)):
                vs=[cs[(m,s)] for s in SIZES if cs[(m,s)][0] is not None]
                MA[m]=(sum(v for v,_ in vs)/len(vs), all(c for _,c in vs)) if vs else (None,True)
        rows=[]
        for i,s in enumerate(SIZES+["MA"]):
            lab=f"\\multirow{{{3*5}}}{{*}}{{\\textbf{{{sub}}}}}" if i==0 else ""
            slab=(f"\\multirow{{3}}{{*}}{{{s.upper()}}}" if s!="MA" else "\\multirow{3}{*}{\\textit{Model-Avg}}")
            if s=="MA": rows.append(f"\\cmidrule(l){{2-{ncol}}}")
            for var,cs in (("CoT",cc),("No-CoT",cn)):
                vals={m:(cs[(m,s)] if s!="MA" else (MAc if var=="CoT" else MAn)[m]) for m in order}; have=[m for m in order if vals[m][0] is not None]
                best=max(have,key=lambda m:vals[m][0]) if have else None
                cells=[fmt(vals[m][0],vals[m][1],bold=(m==best)) for m in order]
                cells.append(fmt(sum(vals[m][0] for m in have)/len(have), all(vals[m][1] for m in have)) if have else "---")
                rows.append(f"{lab if var=='CoT' else ''} & {slab if var=='CoT' else ''} & {var} & "+" & ".join(cells)+" \\\\")
            dcell=[]; ds=[]
            for m in order:
                a=(cc[(m,s)] if s!="MA" else MAc[m]); b=(cn[(m,s)] if s!="MA" else MAn[m])
                if a[0] is None or b[0] is None: dcell.append("---")
                else: d=b[0]-a[0]; ds.append((d,a[1] and b[1])); dcell.append(fmtd(d,a[1] and b[1]))
            dcell.append(fmtd(sum(d for d,_ in ds)/len(ds), all(c for _,c in ds)) if ds else "---")
            rows.append(" & & $\\Delta$ & "+" & ".join(dcell)+" \\\\"+("" if s=="MA" else f" \\cmidrule(l){{3-{ncol}}}"))
        blocks.append("\n".join(rows))
    body=f"\n\\cmidrule(l){{1-{ncol}}}\n".join(blocks)
    h1=["\\multirow{2}{*}{\\textbf{Subset}}","\\multirow{2}{*}{\\textbf{Model}}","\\multirow{2}{*}{\\textbf{Variant}}"]; rules=[]; col=4
    for f,ms in fams:
        h1.append(f"\\multicolumn{{{len(ms)}}}{{c}}{{\\textbf{{{f}}}}}" if len(ms)>1 else f"\\textbf{{{f}}}"); rules.append(f"\\cmidrule(lr){{{col}-{col+len(ms)-1}}}"); col+=len(ms)
    h1.append("\\multirow{2}{*}{\\makecell{\\textbf{Method-}\\\\\\textbf{Avg}}}")
    h2=" & & & "+" & ".join(f"\\textbf{{{m}}}"+("$^\\dagger$" if m=="SelfCrit" else "") for m in order)+" & \\\\"
    spec="@{}l@{\;\;}l@{\;\;}l@{\;\;\;}"+"c"*len(order)+"@{\;\;\;}c@{}"
    out="\n".join(["\\begin{table*}[t]","\\centering","\\resizebox{0.99\\textwidth}{!}{%",f"\\begin{{tabular}}{{{spec}}}","\\toprule",
        " & ".join(h1)+" \\\\"," ".join(rules),h2,"\\midrule",body,"\\bottomrule","\\end{tabular}%","}",
        "\\caption{No-CoT ablation on the RLVR domain of \\textsc{OLMo-Detect}, evaluated via "+MET[METRIC]["name"]+". \\textbf{CoT} scores the original problem--solution instances (CoT-GSM8K, CoT-MATH); "
        "\\textbf{No-CoT} scores the same instances with the chain-of-thought removed (question and final answer only); $\\Delta$ = No-CoT $-$ CoT. Methods are grouped by family as in "
        "Table~\\ref{"+MET[METRIC]["main"]+"}. \\textbf{Model-Avg} averages across model sizes, while \\textbf{Method-Avg} averages the methods with a value in that row. "
        "The best-performing method in each CoT/No-CoT row is highlighted in bold. $^\\dagger$SelfCrit is evaluated only on DPO and RLVR.}",
        "\\label{table:nocot_ablation}" if METRIC=="auc" else "\\label{table:tpr_unsup:nocot_ablation}","\\end{table*}"])+"\n"
    if tex: Path(tex).write_text(out)
    print(out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("--split",default="matched"); ap.add_argument("--tex"); ap.add_argument("--metric",default="auc",choices=["auc","tpr"]); ap.add_argument("--caption-file"); a=ap.parse_args(); METRIC=a.metric
    if a.table=="main_unsup": main_unsup(a.split,a.tex)
    elif a.table=="main_unsup_grouped": main_unsup_grouped(a.split,a.tex)
    elif a.table=="main_unsup_transposed": main_unsup_transposed(a.split,a.tex,caption=(open(a.caption_file).read().strip() if a.caption_file else None))
    elif a.table=="appendix_domains": appendix_domains(a.split,a.tex)
    elif a.table=="appendix_domains_grouped": appendix_domains_grouped(a.split,a.tex)
    elif a.table=="nocot_ablation": nocot_ablation(a.split,a.tex)
    elif a.table=="appendix_stages": appendix_stages(a.split,a.tex)
    elif a.table=="main_sup": main_sup(a.split,a.tex,compact=True)
    elif a.table=="appendix_sup": main_sup(a.split,a.tex,compact=False)
    elif a.table=="main_sup_stages": main_sup(a.split,a.tex,compact=False,gains=False,ref=False,label="main")
    elif a.table=="sup_two": sup_two_tables(a.split,a.tex)
    elif a.table=="main_sup_delta": main_sup(a.split,a.tex,compact=False,gains=False,ref=False,label="main",delta=True)



