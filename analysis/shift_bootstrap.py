"""Paired instance-level bootstrap for the distribution-shift table statistic.

For each method: Delta_bar = mean over sizes of [stage-macro over the 10 shifted-variant subsets of (AUC_shifted - AUC_matched)].
Bootstrap (B resamples): within every leaf, resample matched members, shifted members and the SHARED non-members with replacement;
the same resampled instances are used for all four model sizes (instances are aligned across sizes by text / sample_id / EMMIA index),
and the resampled non-members are used for both AUCs of a subset (paired). p (two-sided) = 2 * min(P(Delta*<=0), P(Delta*>=0)) with the
(k+1)/(B+1) correction; 95% CI = percentiles. Fixed-effects claim: "on these domains the method's AUC is inflated/degraded".

Usage: ../OLMo-Detect/.venv/bin/python analysis/shift_bootstrap.py [--boot 1000] -> analysis/logs/shift_bootstrap.json + markdown to stdout
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT
ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; RL, RI, EM, SD, SIZES = PT.RL, PT.RI, PT.EM, PT.SD, PT.SIZES
rng = np.random.default_rng(a.seed)
SUB = [(k, stg, dom) for stg, dom, keys in PT.STRUCT for k in keys if PT.leaves(k, "shifted", "7b") is not None]
STG_OF = {k: s for k, s, _ in SUB}; DOM_OF = {k: d for k, _, d in SUB}
METHODS = [(n, s, c) for n, s, c in PT.METHODS if n != "SelfCrit"]

def auc_fast(pos, neg):
    """rank-based AUC, ties counted 1/2"""
    x = np.concatenate([pos, neg]); r = np.empty(len(x)); order = np.argsort(x, kind="mergesort"); xs = x[order]
    i = 0; n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def T(s): return PT.rec_key(s)
def scores_by_key(leaf, size, stem, col):
    p = RL / leaf / SD[size] / f"{stem}.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text()).get("result", {}).get("samples", [])
    if not ss: return None
    by_text = bool(T(ss[0]))
    out = {}
    for s in ss:
        if PT.valid(s.get(col)): out[T(s) if by_text else str(s["sample_id"])] = float(s[col])
    return out
def emmia_by_key(con_leaf, size):
    """EMMIA: {('c', idx): {k:score}}, {('u', idx): ...} from the emmia_raw file of the con leaf's tag"""
    p = EM / PT.emmia_tag(con_leaf) / f"emmia_{size}.json"
    remap = None
    if not p.exists():
        # 2026-09-25: Stack Overflow shifted 7b/13b/32b were run on a 150+150 subsample (data/emmia_sub, seed 0);
        # map the subsample positions back to the full-leaf indices so that they align with the matched run.
        p2 = EM / (PT.emmia_tag(con_leaf) + "_sub300") / f"emmia_{size}.json"
        mp = ROOT.parent / "OLMo-Detect/data/emmia_sub/stackoverflow_sub300.map.json"
        if not (p2.exists() and mp.exists()): return None, None
        p = p2; remap = json.loads(mp.read_text())
    ss = json.loads(p.read_text()).get("samples", []); n_con = sum(1 for s in ss if int(s["label"]) == 1)
    c, u = {}, {}
    for s in ss:
        lab = int(s["label"]); idx = int(s["index"]) - (0 if lab == 1 else n_con)
        if remap is not None: idx = (remap["con_idx"] if lab == 1 else remap["unc_idx"])[idx]
        (c if lab == 1 else u)[idx] = {k: float(v) for k, v in s["final_scores"].items()}
    return c, u

def build(name, stem, col):
    """-> {subset: [ (M, S, U, size_rows) ... ]}: one entry per (leaf index, group of sizes sharing the same leaves); arrays are rows=sizes in the
    group x instances (aligned across those sizes by text / sample_id / EMMIA index); EMMIA arrays are dicts of base keys."""
    EK = ("loss", "zlib", "mink", "minkpp"); data = {}
    for k, stg, dom in SUB:
        triples = []
        for li in range(len(PT.leaves(k, "matched", SIZES[0]))):
            groups = {}
            for si, s in enumerate(SIZES):
                (cm, _), (um, _) = PT.leaves(k, "matched", s)[li]; (cs, _), (us, _) = PT.leaves(k, "shifted", s)[li]
                groups.setdefault((cm, cs, um), []).append(si)
            for (cm, cs, um), sis in groups.items():
                per = []
                for si in sis:
                    s = SIZES[si]
                    if stem == "__emmia__":
                        mc, mu = emmia_by_key(cm, s); sc, su = emmia_by_key(cs, s)
                        if mc is None or sc is None: per = None; break
                        per.append((mc, sc, mu, su))   # EMMIA: each run's OWN non-member scores (scores are only comparable within a run)
                    else:
                        M = scores_by_key(cm, s, stem, col); S = scores_by_key(cs, s, stem, col); U = scores_by_key(um, s, stem, col)
                        if M is None or S is None or U is None: per = None; break
                        per.append((M, S, U))
                if per is None: continue
                NP = len(per[0])
                keys = [sorted(set.intersection(*[set(x[j]) for x in per]), key=str) for j in range(NP)]
                if stem == "__emmia__":   # same non-member instances in both runs (a subsampled shifted run restricts the matched one)
                    common = sorted(set(keys[2]) & set(keys[3]), key=str); keys[2] = keys[3] = common
                if min(len(kk) for kk in keys) < 5: continue
                if stem == "__emmia__":
                    arrs = [{e: np.array([[per[i][j][x][e] for x in keys[j]] for i in range(len(per))]) for e in EK} for j in range(NP)]
                else:
                    arrs = [np.array([[per[i][j][x] for x in keys[j]] for i in range(len(per))]) for j in range(NP)]
                triples.append(tuple(arrs) + (sis,))
        if triples: data[k] = triples
    return data

def stat(data, stem, idx=None):
    """Delta_bar (mean over sizes of stage-macro of subset deltas); leaves of a subset are POOLED before the AUC (as in paper_tables);
    idx = optional resample indices per (subset, triple, part)"""
    per_size = []
    for si in range(len(SIZES)):
        vals = []
        for k, triples in data.items():
            ek = None; parts = None
            for ti, trip in enumerate(triples):
                *AP, sis = trip
                if parts is None: parts = [[] for _ in AP]
                if si not in sis: continue
                r = sis.index(si)
                for j, A in enumerate(AP):
                    if stem == "__emmia__":
                        ek = list(A); parts[j].append({e: (A[e][r][idx[(k, ti, j)]] if idx is not None else A[e][r]) for e in ek})
                    else:
                        parts[j].append(A[r][idx[(k, ti, j)]] if idx is not None else A[r])
            if not parts or not parts[0]: continue
            if stem == "__emmia__":
                best_m = best_s = None
                for e in ek:
                    m = np.concatenate([x[e] for x in parts[0]]); s_ = np.concatenate([x[e] for x in parts[1]]); um = np.concatenate([x[e] for x in parts[2]]); us = np.concatenate([x[e] for x in parts[3]])
                    am, as_ = auc_fast(m, um), auc_fast(s_, us)
                    if best_m is None or am > best_m: best_m = am
                    if best_s is None or as_ > best_s: best_s = as_
                vals.append((k, best_s - best_m))
            else:
                m = np.concatenate(parts[0]); s_ = np.concatenate(parts[1]); u = np.concatenate(parts[2])
                vals.append((k, auc_fast(s_, u) - auc_fast(m, u)))
        dom = {}
        for k, v in vals: dom.setdefault((STG_OF[k], DOM_OF[k]), []).append(v)
        stg = {}
        for (s, d), vs in dom.items(): stg.setdefault(s, []).append(np.mean(vs))
        if stg: per_size.append(np.mean([np.mean(v) for v in stg.values()]))
    return float(np.mean(per_size))

import os
_only = os.environ.get('ONLY')
if _only: METHODS = [m for m in METHODS if m[0] in _only.split(',')]
out = {}
print(f"{'method':8s} {'Δ̄':>6s} {'boot p':>7s} {'95% CI':>16s}  subsets")
for name, stem, col in METHODS:
    data = build(name, stem, col)
    if not data: print(f"{name:8s} ---"); continue
    obs = stat(data, stem); boots = []
    sizes_of = {(k, ti, part): trip[part] for k, triples in data.items() for ti, trip in enumerate(triples) for part in range(len(trip) - 1)}
    nof = {key: (next(iter(v.values())).shape[1] if stem == "__emmia__" else v.shape[1]) for key, v in sizes_of.items()}
    for b in range(a.boot):
        idx = {key: rng.integers(0, n, n) for key, n in nof.items()}
        for key in list(idx):
            if key[2] == 3: idx[key] = idx[(key[0], key[1], 2)]   # EMMIA: same resampled non-member instances in both runs
        boots.append(stat(data, stem, idx))
    boots = np.array(boots); lo, hi = np.percentile(boots, [2.5, 97.5]); lo99, hi99 = np.percentile(boots, [0.5, 99.5])
    p = 2 * min((np.sum(boots <= 0) + 1) / (a.boot + 1), (np.sum(boots >= 0) + 1) / (a.boot + 1)); p = min(p, 1.0)
    out[name] = dict(delta=obs, p=float(p), ci=(float(lo), float(hi)), ci99=(float(lo99), float(hi99)), B=int(a.boot), n_subsets=len(data))
    print(f"{name:8s} {obs:+6.3f} {p:7.3f} [{lo:+.3f}, {hi:+.3f}]  {len(data)}")
(ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
_pj = ROOT / "analysis/logs/shift_bootstrap.json"
if _only and _pj.exists():
    _old = json.load(open(_pj)); _old.update(out); out = _old
json.dump(out, open(_pj, "w"), indent=1)
