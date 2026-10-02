#!/usr/bin/env python
"""Retune on the SHIFTED dev split and re-evaluate the shifted test split (2026-09-26).

Covers the five unsupervised MIAs whose hyperparameter can be re-applied from data already on
disk, with no new GPU work:
  MinK, MinK++  -- results_mink_grid stores every k (11 values) for members and non-members,
                   matched and shifted, dev and test; k is selected here on the shifted dev split.
  DCQ, Neigh., Guided -- the shifted dev sweeps in OLMo-Detect/iclr_tune already record the
                   selected value, and the stored per-sample fields let the score be recomputed.

For each method it reports, per domain, the matched-to-shifted change under the paper's protocol
(both sides on the matched-selected value) and under per-setting tuning (shifted side on the
shifted-selected value), so the I/D pattern can be compared directly.

Usage: ../OLMo-Detect/.venv/bin/python analysis/shifted_retune.py -> analysis/logs/shifted_retune.json
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
import paper_tables as PT
GRID = ROOT.parent / "OLMo-Detect/results_mink_grid"
TUNE = ROOT.parent / "OLMo-Detect/iclr_tune"
SIZES = PT.SIZES
KS = ["0.05","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]

def auc(m, u):
    if len(m) == 0 or len(u) == 0: return None
    r = rankdata(np.concatenate([m, u])); n1, n0 = len(m), len(u)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def dev_name(leaf):
    """results dataset name of the dev counterpart of a test leaf"""
    return leaf + "_dev" if not leaf.endswith("_shifted") else leaf[:-len("_shifted")] + "_dev_shifted"

# ---------- MinK / MinK++ from the k-grid -------------------------------------------------
def grid_scores(leaf, size, field):
    p = GRID / leaf / PT.SD[size] / "minkgrid.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text())["samples"]
    return {k: np.array([float(s[field][k]) for s in ss]) for k in KS}

def mink_pick(field, size):
    """select k on the pooled SHIFTED dev split"""
    best, bk = -1, None
    for k in KS:
        m, u = [], []
        for stg, dom, keys in PT.STRUCT:
            for key in keys:
                lv = PT.leaves(key, "shifted", size)
                if not lv: continue
                for (cl, _), (ul, _) in lv:
                    a = grid_scores(dev_name(cl), size, field); b = grid_scores(dev_name(ul), size, field)
                    if a is None or b is None: continue
                    m.append(a[k]); u.append(b[k])
        if not m: continue
        v = auc(np.concatenate(m), np.concatenate(u))
        if v is not None and v > best: best, bk = v, k
    return bk, best

def mink_dom(field, k, dom_keys, split, size):
    vals = []
    for key in dom_keys:
        lv = PT.leaves(key, split, size)
        if not lv: continue
        m, u = [], []
        for (cl, _), (ul, _) in lv:
            a = grid_scores(cl, size, field); b = grid_scores(ul, size, field)
            if a is None or b is None: return None
            m.append(a[k]); u.append(b[k])
        v = auc(np.concatenate(m), np.concatenate(u))
        if v is not None: vals.append(v)
    return sum(vals) / len(vals) if vals else None

# ---------- DCQ / Neigh. / Guided recomputed from stored fields ---------------------------
def read_samples(leaf, size, stem):
    p = PT.RL / leaf / PT.SD[size] / f"{stem}.json"
    if not p.exists(): return None
    return json.loads(p.read_text())["result"]["samples"]

def neigh_score(ss, st):
    out = []
    for s in ss:
        nb = [v for v in s.get("neighbour_lls", []) if np.isfinite(v)]
        t = s.get("target_ll")
        if not nb or t is None or not np.isfinite(t): continue
        mu = float(np.mean(nb))
        if st == "d": out.append(t - mu)
        else:
            sd = float(np.std(nb)) if len(nb) > 1 else 1.0
            out.append((t - mu) / (sd if sd else 1.0))
    return np.array(out)

def guided_score(ss, st):
    f = "rouge_guided" if st == "rouge_guided" else "rouge_delta"
    return np.array([float(s[f]) for s in ss if s.get(f) is not None])

def dcq_score(ss, st):
    pe = {L: [] for L in "ABCD"}
    for s in ss:
        b = s.get("bdq_probs") or {}
        for L in "ABCD":
            if b.get(L) is not None: pe[L].append(float(b[L]))
    pe = {L: (sum(v) / len(v) if v else 0.0) for L, v in pe.items()}
    npl = [L for L in "ABCD" if pe[L] < 0.2]
    out = []
    for s in ss:
        pc = s.get("pcorrect_per_slot") or {}
        if not pc: continue
        if st == "mean_pcorrect": out.append(sum(pc.values()) / len(pc))
        elif st == "min_pcorrect": out.append(min(pc.values()))
        else:
            Ls = list(pc) if st == "pcorrect_kappa" else [L for L in pc if L in npl]
            ks = [(float(pc[L]) - pe.get(L, 0.0)) / max(1e-6, 1.0 - pe.get(L, 0.0)) for L in Ls]
            out.append(sum(ks) / len(ks) if ks else float("nan"))
    return np.array([v for v in out if np.isfinite(v)])

SCORERS = {"Neigh.": ("neighborhood_attack", neigh_score), "Guided": ("guided_instruction", guided_score),
           "DCQ": ("dcq", dcq_score)}

def generic_dom(name, st, dom_keys, split, size):
    stem, fn = SCORERS[name]
    vals = []
    for key in dom_keys:
        lv = PT.leaves(key, split, size)
        if not lv: continue
        m, u = [], []
        for (cl, _), (ul, _) in lv:
            a = read_samples(cl, size, stem); b = read_samples(ul, size, stem)
            if a is None or b is None: return None
            m.append(fn(a, st)); u.append(fn(b, st))
        v = auc(np.concatenate(m), np.concatenate(u))
        if v is not None: vals.append(v)
    return sum(vals) / len(vals) if vals else None

MATCHED_CFG = {"Neigh.": {s: "d" for s in SIZES},
               "Guided": {"1b": "rouge_delta", "7b": "rouge_delta", "13b": "rouge_delta", "32b": "rouge_guided"},
               "DCQ": {"1b": "min_pcorrect", "7b": "min_pcorrect", "13b": "pcorrect_kappa_np", "32b": "pcorrect_kappa"}}
def sweep_pick(fam):
    out = {}
    for s in SIZES:
        p = TUNE / f"{fam}_tune_shifted_{s}.json"
        out[s] = json.load(open(p))["best"]["score_type"] if p.exists() else None
    return out
SHIFTED_CFG = {"Neigh.": sweep_pick("neighborhood"), "Guided": sweep_pick("guided"), "DCQ": sweep_pick("dcq")}

DOMS7 = [(d, keys) for stg, d, keys in PT.STRUCT if d not in ("GSM8K", "RLVR")]
out = {}
print(f"{'method':8s}{'size':5s}{'matched cfg':>14s}{'shifted cfg':>14s}   per-domain delta (old -> new)")
for name in ["MinK", "MinK++", "Neigh.", "Guided", "DCQ"]:
    res = {}
    for size in SIZES:
        if name in ("MinK", "MinK++"):
            field = "mink" if name == "MinK" else "minkpp"
            mcfg = "0.2"; scfg, _ = mink_pick(field, size)
            f = lambda dk, sp, cfg: mink_dom(field, cfg, dk, sp, size)
        else:
            mcfg = MATCHED_CFG[name][size]; scfg = SHIFTED_CFG[name][size]
            f = lambda dk, sp, cfg: generic_dom(name, cfg, dk, sp, size)
        if scfg is None: continue
        for dom, keys in DOMS7:
            mt = f(keys, "matched", mcfg); sh_old = f(keys, "shifted", mcfg); sh_new = f(keys, "shifted", scfg)
            if None in (mt, sh_old, sh_new): continue
            res.setdefault(dom, {})[size] = dict(matched=mt, shifted_old=sh_old, shifted_new=sh_new)
        print(f"{name:8s}{size:5s}{str(mcfg):>14s}{str(scfg):>14s}   {len(res)} domains")
    out[name] = res
json.dump(out, open(ROOT / "analysis/logs/shifted_retune.json", "w"), indent=1)
print("\n-> analysis/logs/shifted_retune.json")
