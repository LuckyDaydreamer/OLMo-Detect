#!/usr/bin/env python
"""Consolidate ALL latest, instance-verified scores across EVERY setting into ICLR2027_OLMo-Detect/results_latest,
and emit per-setting coverage tables (unscored member/non-member instances + unscored methods).

Settings:
  matched_olmo2 (unsup+sup) : OLMo2-{1,7,13,32}B-Instruct, member_*_matched vs non-member_*
  shifted_olmo2 (unsup)     : same models, member_*_shifted (no gsm8k/rlvr)
  olmo3 (representative)     : Olmo-3-* variants, matched, domains {dclm,gsm8k,cot-gsm8k}
  nonolmo (representative)   : DCLM-Baseline, matched, domains {dclm,gsm8k,cot-gsm8k}
Instance-verify: a score file's sample `text` set is matched against the CURRENT benchmark TEST instances;
only current-instance samples are copied. Coverage = covered/current-test-N."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
import config
import os,glob,json,hashlib,shutil
from datetime import datetime
ROOT=str(config.REPO); B=str(config.BENCHMARK)
OUT=f"{ROOT}/ICLR2027_OLMo-Detect/results_latest"
SRC=[f"{ROOT}/OLMo-Detect/results_iclr",f"{ROOT}/OLMo-Detect/results_olmo3",f"{ROOT}/OLMo-Detect/results_olmo",
     f"{ROOT}/ICLR2027_OLMo-Detect/results_merged",f"{ROOT}/OLMo-Detect/results"]
SRC=[d for d in SRC if os.path.isdir(d)]
h=lambda t:hashlib.md5(t.encode("utf-8","ignore")).hexdigest()
import sys as _sys; _sys.path.insert(0,f"{ROOT}/OLMo-Detect"); from data_utils import extract_text as _xt, extract_sample_id as _xid
ALT={}   # FIX 2026-09-21: alt key -> raw-text hash. Scorers store the FORMATTED text (data_utils.extract_text: StarCoder header,
         # DPO prompt+response, ...), and always a sample_id. Map h(formatted) and "id:<sample_id>" back to the canonical raw-text hash
         # so instance verification no longer drops valid samples ("17/13 unscored" artifacts). Benchmark counts stay = raw-text set.
ALT_ID={}  # FIX 2026-09-22 16:55: "id:<sample_id>" -> raw hash must be PER LEAF (ids like sample_0 repeat in every file; the global
           # setdefault let the first benchmark file win and silently dropped GEN results of every later leaf). Keyed by id(bt-set).
def bset(globs):
    s=set(); idmap={}; ALT_ID[id(s)]=idmap
    for g in (globs or []):
        for f in glob.glob(g):
            if f.endswith(".jsonl"):
                for i,l in enumerate(open(f)):
                    if l.strip():
                        try:
                            r=json.loads(l); raw=h(r.get("text") or r.get("content") or ""); s.add(raw)
                            try:
                                ft=_xt(r)
                                if ft: ALT.setdefault(h(ft),raw)
                            except Exception: pass
                            try: idmap.setdefault("id:"+str(_xid(r,i)),raw)
                            except Exception: pass
                        except: pass
    return s
UNSUP=["loss_zlib_lowercase","minkprob","dcpdd","recall","camia_faithful","pac","neighborhood_attack","emmia","dcq","cdd","guided_instruction"]
SUP=["fsd","mia_tuner","camia_lr"]; REP=["loss_zlib_lowercase","minkprob","dcpdd","pac","camia_faithful"]
OLMO2=[f"OLMo2-{s}-Instruct" for s in ["1B","7B","13B","32B"]]
OLMO3=["Olmo-3-7B-Instruct","Olmo-3.1-32B-Instruct","Olmo-3-1025-7B","Olmo-3-7B-Think","Olmo-3-7B-RL-Zero-Math"]
NONOLMO=["DCLM-1B","DCLM-Baseline-7B"]
# ---- leaves: (label, unc_dir, con_matched_dir, con_shifted_dir|None, bench_unc_glob, bench_conM_glob, bench_conS_glob|None) ----
def tv(p,sub=""): return [f"{B}/{p}/test/{sub}"] if sub else [f"{B}/{p}/test/*.jsonl"]
def sg(p,sub): return [f"{B}/{p}/test/*{sub}*.jsonl"]
L=[]
def add(label,unc,cm,cs,bu,bm,bs): L.append(dict(label=label,unc=unc,cm=cm,cs=cs,bu=bu,bm=bm,bs=bs))
for dom in ["dclm","pes2o","openwebmath"]:
    add(dom,f"non-member_pretraining_{dom}",f"member_pretraining_{dom}_matched",f"member_pretraining_{dom}_shifted",
        tv(f"pretraining/{dom}/non-member"),tv(f"pretraining/{dom}/member/matched"),tv(f"pretraining/{dom}/member/shifted"))
for sub in ["java","python"]:
    add(f"starcoder_{sub}",f"non-member_pretraining_starcoder_{sub}",f"member_pretraining_starcoder_{sub}_matched",f"member_pretraining_starcoder_{sub}_shifted",
        sg("pretraining/starcoder/non-member",sub),sg("pretraining/starcoder/member/matched",sub),sg("pretraining/starcoder/member/shifted",sub))
add("gsm8k","non-member_midtraining_gsm8k","member_midtraining_gsm8k",None,
    tv("midtraining/gsm8k/non-member"),tv("midtraining/gsm8k/member"),None)
for sub in ["math","stackoverflow"]:
    add(f"se_{sub}",f"non-member_midtraining_stackexchange_{sub}",f"member_midtraining_stackexchange_{sub}_matched",f"member_midtraining_stackexchange_{sub}_shifted",
        sg("midtraining/stackexchange/non-member",sub),sg("midtraining/stackexchange/member/matched",sub),sg("midtraining/stackexchange/member/shifted",sub))
for sub in ["cot-gsm8k","cot-math","gsm8k_no_cot","math_no_cot"]:   # *_no_cot = no-CoT ablation leaves (22 Sep); excluded from the main-table STRUCT in paper_tables.py
    add(f"rlvr_{sub}",f"non-member_posttraining_rlvr_{sub}",f"member_posttraining_rlvr_{sub}",None,
        sg("posttraining/rlvr/non-member",sub),sg("posttraining/rlvr/member",sub),None)
for pair in ["1b-32b","7b-13b"]:
    for sub in ["aya","wildchat"]:
        add(f"sft_{pair}_{sub}",f"non-member_posttraining_sft_{pair}_{sub}",f"member_posttraining_sft_{pair}_{sub}_matched",f"member_posttraining_sft_{pair}_{sub}_shifted",
            sg(f"posttraining/sft/{pair}/non-member",sub),sg(f"posttraining/sft/{pair}/member/matched",sub),sg(f"posttraining/sft/{pair}/member/shifted",sub))
for s in ["1b","7b","13b","32b"]:
    for side in ["chosen","rejected"]:
        add(f"dpo_{s}_{side}",f"non-member_posttraining_dpo_{s}_{side}",f"member_posttraining_dpo_{s}_matched_{side}",f"member_posttraining_dpo_{s}_shifted_{side}",
            [f"{B}/posttraining/dpo/non-member/test/*_dpo_{s}_{side}_*.jsonl"],[f"{B}/posttraining/dpo/member/matched/test/*_dpo_{s}_matched_{side}_*.jsonl"],[f"{B}/posttraining/dpo/member/shifted/test/*_dpo_{s}_shifted_{side}_*.jsonl"])
# ---- valid OLMo2 models per leaf (DPO/SFT are size/pair-specific) ----
for lf in L:
    lab=lf["label"]
    if lab.startswith("dpo_"): lf["vm"]=[f"OLMo2-{lab.split('_')[1].upper()}-Instruct"]
    elif lab.startswith("sft_1b-32b"): lf["vm"]=["OLMo2-1B-Instruct","OLMo2-32B-Instruct"]
    elif lab.startswith("sft_7b-13b"): lf["vm"]=["OLMo2-7B-Instruct","OLMo2-13B-Instruct"]
    else: lf["vm"]=None   # all sizes valid
# ---- benchmark text-sets ----
for lf in L:
    lf["btu"]=bset(lf["bu"]); lf["btm"]=bset(lf["bm"]); lf["bts"]=bset(lf["bs"]) if lf["bs"] else set()
# ---- consolidation: for each dir, each model present across SRC, each stem: keep current-instance samples ----
# FIX 2026-09-21: NEVER rmtree(OUT) — results_latest also holds hand-written docs (TASK*.md, OUTPUT_DIRS.md, ...). Remove only leaf dirs.
os.makedirs(OUT,exist_ok=True)
for _d in glob.glob(f"{OUT}/*"):
    if os.path.isdir(_d) and (os.path.basename(_d).startswith("member_") or os.path.basename(_d).startswith("non-member_")): shutil.rmtree(_d)
def consolidate(dirname,bt):
    """copy latest instance-verified score files for all models under dirname; return {model:{stem:covered}}"""
    if not bt: return {}
    cov={}
    models=set()
    for src in SRC:
        d=f"{src}/{dirname}"
        if os.path.isdir(d): models.update(os.path.basename(x) for x in glob.glob(f"{d}/*") if os.path.isdir(x))
    for model in models:
        stem_samp={}
        for src in SRC:
            dd=f"{src}/{dirname}/{model}"
            if not os.path.isdir(dd): continue
            for jf in glob.glob(f"{dd}/*.json"):
                stem=os.path.basename(jf)[:-5]
                if stem.startswith("run_manifest"): continue
                try: d=json.loads(open(jf).read())
                except: continue
                mt=os.path.getmtime(jf)
                for x in d.get("result",{}).get("samples",[]):
                    t=x.get("text")
                    # FIX 2026-09-22: generative-method outputs (neighborhood/dcq/cdd/guided) carry NO text, only sample_id -> match by id
                    hh=h(t) if t is not None else None
                    if hh not in bt: hh=(ALT.get(hh) if hh is not None else None) or ALT_ID.get(id(bt),{}).get("id:"+str(x.get("sample_id")))
                    if hh in bt:
                        cur=stem_samp.setdefault(stem,{})
                        if hh not in cur or mt>cur[hh][0]: cur[hh]=(mt,x)
        cov[model]={}
        for stem,hmap in stem_samp.items():
            od=f"{OUT}/{dirname}/{model}"; os.makedirs(od,exist_ok=True)
            json.dump({"result":{"samples":[v[1] for v in hmap.values()]}},open(f"{od}/{stem}.json","w"))
            cov[model][stem]=len(hmap)
    return cov
COV={}   # (dirname)-> {model:{stem:covered}}
for lf in L:
    COV[lf["unc"]]=consolidate(lf["unc"],lf["btu"])
    COV[lf["cm"]]=consolidate(lf["cm"],lf["btm"])
    if lf["cs"]: COV[lf["cs"]]=consolidate(lf["cs"],lf["bts"])
# ---- coverage tables ----
def cov_of(dirname,model,stems):
    c=COV.get(dirname,{}).get(model,{}); return {s:c.get(s,0) for s in stems}
def unscored_row(lf,model,split,methods):
    con_dir=lf["cm"] if split=="matched" else lf["cs"]; con_bt=lf["btm"] if split=="matched" else lf["bts"]
    if con_dir is None or not con_bt: return None
    cm=cov_of(con_dir,model,methods); um=cov_of(lf["unc"],model,methods)
    ctn=len(con_bt); utn=len(lf["btu"])
    mM=ctn-max(cm.values() or [0]); mN=utn-max(um.values() or [0])
    miss=[s for s in methods if min(cm[s],um[s])<min(ctn,utn)]
    return ctn,utn,mM,mN,miss
def emit(title,leaves,models,methods,split):
    print(f"\n{'='*90}\n{title}\n{'='*90}")
    print(f"{'domain':22} {'model':22} {'testN(M/NM)':>12} {'unscored(M/NM)':>15} {'#meth-miss':>10}/{len(methods)}")
    lines=[f"## {title}",f"| domain | model | testN M/NM | unscored M/NM | methods missing ({len(methods)}) |","|---|---|---|---|---|"]
    for lf in leaves:
        mlist=[m for m in models if (lf.get("vm") is None or m in lf["vm"])] if set(models)<=set(OLMO2) else models
        for model in mlist:
            r=unscored_row(lf,model,split,methods)
            if r is None: continue
            ctn,utn,mM,mN,miss=r
            # only show rows that have ANY data or ANY gap
            has=any(cov_of((lf['cm'] if split=='matched' else lf['cs']),model,methods).values()) or any(cov_of(lf['unc'],model,methods).values())
            if not has and mM==ctn and mN==utn: tag="(never scored)"
            else: tag=""
            print(f"{lf['label']:22} {model:22} {str(ctn)+'/'+str(utn):>12} {str(mM)+'/'+str(mN):>15} {len(miss):>10} {tag}")
            lines.append(f"| {lf['label']} | {model} | {ctn}/{utn} | {mM}/{mN} | {len(miss)}: {','.join(m.split('_')[0] for m in miss)} |")
    return lines
alllines=[f"# ALL-SETTINGS coverage (unscored instances + methods)  generated {datetime.now():%Y-%m-%d %H:%M}",
          "unscored = current TEST instances with no score (member/non-member). methods missing = target methods below full coverage.\n"]
matched_leaves=L
alllines+=emit("SETTING 1a: MATCHED / OLMo-2 / UNSUPERVISED (11 methods)",matched_leaves,OLMO2,UNSUP,"matched")
alllines+=emit("SETTING 1b: MATCHED / OLMo-2 / SUPERVISED (3 methods)",matched_leaves,OLMO2,SUP,"matched")
shifted_leaves=[lf for lf in L if lf["cs"]]
alllines+=emit("SETTING 2: SHIFTED / OLMo-2 / UNSUPERVISED (11 methods)",shifted_leaves,OLMO2,UNSUP,"shifted")
rep_leaves=[lf for lf in L if lf["label"] in ("dclm","gsm8k","rlvr_cot-gsm8k")]
alllines+=emit("SETTING 3: OLMo-3 / representative methods (4)",rep_leaves,OLMO3,REP,"matched")
alllines+=emit("SETTING 4: non-OLMo models / representative methods (4)",rep_leaves,NONOLMO,REP,"matched")
open(f"{OUT}/COVERAGE_ALL.md","w").write("\n".join(alllines))
print(f"\nconsolidated -> {OUT} | tables -> {OUT}/COVERAGE_ALL.md")
print("ALL_DONE")
