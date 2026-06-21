#!/usr/bin/env python
"""
Compute OLMo-Detect AUC and TPR@5%FPR for ONE detection method, at every
granularity: all-stage (Overall), per-stage, per-domain, and per-subset.
Works both for reproducing our reported methods and for evaluating a new one.

This covers the 14 "per-sample" methods, whose scope AUC is computed by POOLING
the contaminated (label 1) and uncontaminated (label 0) instances across the
scope and taking a single ROC-AUC — exactly as in the paper (Table 4 "Overall"
pools instances across stages; domain/stage numbers pool their members).

  Perplexity, Zlib, Lowercase, Min-K%, Min-K%++, DC-PDD, RECALL, CAMIA, PAC,
  Neighborhood, DCQ, CDD, Guided Instruction, Self-Critique

Method-specific data handling (this is where methods differ — handled for you):
  * DPO          → contaminated = chosen ∪ rejected pooled (likewise uncontam.).
  * SFT          → split across model-pair dirs (1B/32B vs 7B/13B); merged here.
  * Self-Critique→ only defined for DPO and RLVR; DPO reads the combined dir
                   (selfcrit.json sits there, not in chosen/rejected). All other
                   (method, domain) combinations that don't exist print "n/a".

Usage:
  python evaluate.py --method ppl
  python evaluate.py --method camia --fpr 0.05      # also prints TPR@FPR
  python evaluate.py --list                          # list method keys
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS = Path(__file__).resolve().parent / "results"
SIZES = ["1b", "7b", "13b", "32b"]
SDIR = {s: f"OLMo2-{s.upper()}-Instruct" for s in SIZES}

# method key -> (file stem, score column).  "selfcrit" handled specially for DPO.
METHODS = {
    "ppl":        ("loss_zlib_lowercase", "loss_score"),
    "zlib":       ("loss_zlib_lowercase", "zlib_score"),
    "lowercase":  ("loss_zlib_lowercase", "lowercase_score"),
    "mink":       ("minkprob", "min_k_prob_mean_logprob"),
    "minkpp":     ("minkprob", "min_k_pp_score"),
    "dcpdd":      ("dcpdd", "dc_pdd_score"),
    "recall":     ("recall", "recall_score"),
    "camia":      ("camia", "camia_score"),
    "pac":        ("pac", "pac_score"),
    "neighborhood": ("neighborhood_attack", "score"),
    "dcq":        ("dcq", "score"),
    "cdd":        ("cdd", "peak"),
    "guided":     ("guided_instruction", "score"),
    "selfcrit":   ("selfcrit", "self_critique_score"),
}
ALIASES = {"perplexity":"ppl","lower":"lowercase","min-k":"mink","min_k":"mink",
           "minkprob":"mink","min-k++":"minkpp","minkpp_":"minkpp","dc-pdd":"dcpdd",
           "neigh":"neighborhood","guided_instruction":"guided","self-critique":"selfcrit",
           "self_critique":"selfcrit"}

# ── benchmark structure: leaves carry (stage, domain, subset). ───────────────
# kind decides how the (con, uncon) score lists are built for a (method,size).
#   simple : one con dir + one uncon dir (size = subdir)
#   dpo    : pooled chosen∪rejected  (selfcrit: the combined dir instead)
#   sft    : model-pair dir chosen by size (1b/32b vs 7b/13b)
def L(stage, domain, subset, kind, **kw):
    return {"stage": stage, "domain": domain, "subset": subset, "kind": kind, **kw}

LEAVES = [
    L("pre","DCLM",None,"simple", con="contaminated_pretraining_dclm_matched", unc="uncontaminated_pretraining_dclm"),
    L("pre","peS2o",None,"simple", con="contaminated_pretraining_pes2o_matched", unc="uncontaminated_pretraining_pes2o"),
    L("pre","OpenWebMath",None,"simple", con="contaminated_pretraining_openwebmath_matched", unc="uncontaminated_pretraining_openwebmath"),
    L("pre","StarCoder","Java","simple", con="contaminated_pretraining_starcoder_java_matched", unc="uncontaminated_pretraining_starcoder_java"),
    L("pre","StarCoder","Assembly","simple", con="contaminated_pretraining_starcoder_assembly_matched", unc="uncontaminated_pretraining_starcoder_assembly"),
    L("mid","GSM8K",None,"simple", con="contaminated_midtraining_gsm8k", unc="uncontaminated_midtraining_gsm8k"),
    L("mid","StackExchange","Math","simple", con="contaminated_midtraining_stackexchange_math_matched", unc="uncontaminated_midtraining_stackexchange_math"),
    L("mid","StackExchange","StackOverflow","simple", con="contaminated_midtraining_stackexchange_stackoverflow_matched", unc="uncontaminated_midtraining_stackexchange_stackoverflow"),
    L("post","SFT","Aya","sft", sub="aya"),
    L("post","SFT","WildChat","sft", sub="wildchat"),
    L("post","DPO",None,"dpo"),
    L("post","RLVR","CoT-GSM8K","simple", con="contaminated_posttraining_rlvr_cot-gsm8k", unc="uncontaminated_posttraining_rlvr_cot-gsm8k"),
    L("post","RLVR","CoT-MATH","simple", con="contaminated_posttraining_rlvr_cot-math", unc="uncontaminated_posttraining_rlvr_cot-math"),
    L("post","RLVR","IFEval","simple", con="contaminated_posttraining_rlvr_ifeval", unc="uncontaminated_posttraining_rlvr_ifeval"),
]
STAGE_NAME = {"pre":"Pre-training","mid":"Mid-training","post":"Post-training"}

def _valid(v):
    return not (v is None or v == "-Infinity" or (isinstance(v,float) and math.isnan(v)))

def _read(path: Path, col):
    if not path.exists(): return []
    d = json.loads(path.read_text())
    return [float(s[col]) for s in d.get("result",{}).get("samples",[])
            if col in s and _valid(s.get(col))]

def leaf_scores(leaf, method, size):
    """Return (con_scores, uncon_scores) for one leaf at one size, or ([],[])."""
    stem, col = METHODS[method]
    sd = SDIR[size]
    if leaf["kind"] == "simple":
        return (_read(RESULTS/leaf["con"]/sd/f"{stem}.json", col),
                _read(RESULTS/leaf["unc"]/sd/f"{stem}.json", col))
    if leaf["kind"] == "dpo":
        if method == "selfcrit":           # selfcrit lives in the combined dir
            return (_read(RESULTS/f"contaminated_posttraining_dpo_{size}_matched"/sd/f"{stem}.json", col),
                    _read(RESULTS/f"uncontaminated_posttraining_dpo_{size}"/sd/f"{stem}.json", col))
        con = (_read(RESULTS/f"contaminated_posttraining_dpo_{size}_matched_chosen"/sd/f"{stem}.json", col) +
               _read(RESULTS/f"contaminated_posttraining_dpo_{size}_matched_rejected"/sd/f"{stem}.json", col))
        unc = (_read(RESULTS/f"uncontaminated_posttraining_dpo_{size}_chosen"/sd/f"{stem}.json", col) +
               _read(RESULTS/f"uncontaminated_posttraining_dpo_{size}_rejected"/sd/f"{stem}.json", col))
        return con, unc
    if leaf["kind"] == "sft":
        pair = "1b-32b" if size in ("1b","32b") else "7b-13b"
        sub = leaf["sub"]
        return (_read(RESULTS/f"contaminated_posttraining_sft_{pair}_{sub}_matched"/sd/f"{stem}.json", col),
                _read(RESULTS/f"uncontaminated_posttraining_sft_{pair}_{sub}"/sd/f"{stem}.json", col))
    raise ValueError(leaf["kind"])

def auc_tpr(con, unc, fpr):
    if not con or not unc: return None, None
    y = np.array([1]*len(con)+[0]*len(unc)); s = np.array(con+unc, dtype=float)
    auc = float(roc_auc_score(y, s))
    if np.all(s == s[0]): return auc, float(fpr)
    f, t, _ = roc_curve(y, s); return auc, float(np.interp(fpr, f, t))

def scope_metrics(leaves, method, fpr):
    """Pool the given leaves; return per-size (auc,tpr) + cross-size averages."""
    per = {}
    for sz in SIZES:
        con, unc = [], []
        for lf in leaves:
            c, u = leaf_scores(lf, method, sz)
            con += c; unc += u
        per[sz] = auc_tpr(con, unc, fpr)
    aucs = [v[0] for v in per.values() if v[0] is not None]
    tprs = [v[1] for v in per.values() if v[1] is not None]
    avg = (sum(aucs)/len(aucs) if aucs else None, sum(tprs)/len(tprs) if tprs else None)
    return per, avg

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method"); ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--show-tpr", action="store_true")
    ap.add_argument("--results-dir", default="results",
                    help="Score directory to read (default: results/; use results_repro/ to score a fresh run_all.slurm).")
    a = ap.parse_args()
    if a.list or not a.method:
        print("methods:", ", ".join(METHODS)); return
    global RESULTS
    RESULTS = Path(a.results_dir) if Path(a.results_dir).is_absolute() else Path(__file__).resolve().parent / a.results_dir
    if not RESULTS.is_dir():
        ap.error(f"results dir not found: {RESULTS}")
    m = a.method.lower(); m = ALIASES.get(m, m)
    if m not in METHODS: ap.error(f"unknown method '{a.method}'. Try --list.")

    rows = []  # (indent_label, leaves)
    rows.append(("ALL-STAGE (Overall)", LEAVES))
    for st in ["pre","mid","post"]:
        rows.append((f"  stage: {STAGE_NAME[st]}", [l for l in LEAVES if l["stage"]==st]))
    # per-domain (preserve order of first appearance)
    seen = []
    for l in LEAVES:
        if l["domain"] not in seen: seen.append(l["domain"])
    for dom in seen:
        dleaves = [l for l in LEAVES if l["domain"]==dom]
        rows.append((f"  domain: {dom}", dleaves))
        if any(l["subset"] for l in dleaves):
            for l in dleaves:
                rows.append((f"      subset: {l['subset']}", [l]))

    hdr = f"{'scope':34s} " + " ".join(f"{s.upper():>8s}" for s in SIZES) + f" {'Avg':>8s}"
    print(f"Method: {m}   (AUC{' / TPR@'+format(a.fpr*100,'g')+'%' if a.show_tpr else ''})")
    print(hdr); print("-"*len(hdr))
    for label, leaves in rows:
        per, (aa, at) = scope_metrics(leaves, m, a.fpr)
        def cell(sz):
            au, tp = per[sz]
            if au is None: return "n/a"
            return f"{au:.4f}/{tp:.4f}" if a.show_tpr else f"{au:.4f}"
        avg = "n/a" if aa is None else (f"{aa:.4f}/{at:.4f}" if a.show_tpr else f"{aa:.4f}")
        print(f"{label:34s} " + " ".join(f"{cell(s):>8s}" if not a.show_tpr else f"{cell(s):>16s}" for s in SIZES)
              + (f" {avg:>16s}" if a.show_tpr else f" {avg:>8s}"))

if __name__ == "__main__":
    main()
