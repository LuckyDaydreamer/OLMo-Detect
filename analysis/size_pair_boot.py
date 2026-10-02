"""Paired instance-level bootstrap for a model-size comparison of the aggregate (2026-09-26).
Per (subset, size) the member/non-member index vectors are drawn once and reused by every
method with the same record count, and shared across the two model sizes whenever the counts
match, so the size comparison is paired wherever the data allow it. The aggregate follows the
main tables: subset -> domain mean -> stage mean -> mean of the three stages -> mean over methods.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
import config
import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--split", default="matched")
ap.add_argument("--sizes", default="13b,32b")
a = ap.parse_args()
SA, SB = a.sizes.split(",")
rng = np.random.default_rng(0)
EMKEYS = ("loss", "zlib", "mink", "minkpp")
LOG = config.LOGS

def auc(m, u):
    r = rankdata(np.concatenate([m, u])); n1, n0 = len(m), len(u)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def scores(leaf, size, stem, col):
    p = PT.RL / leaf / PT.SD[size] / f"{stem}.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text()).get("result", {}).get("samples", [])
    v = [float(s[col]) for s in ss if PT.valid(s.get(col))]
    return np.array(v) if v else None

def emmia(cl, size):
    tag = PT.emmia_tag(cl); p = PT.EM / tag / f"emmia_{size}.json"
    if not p.exists(): p = PT.EM / (tag + "_sub300") / f"emmia_{size}.json"
    if not p.exists(): return None, None
    ss = json.loads(p.read_text()).get("samples", [])
    c = {k: np.array([float(s["final_scores"][k]) for s in ss if int(s["label"]) == 1]) for k in EMKEYS}
    u = {k: np.array([float(s["final_scores"][k]) for s in ss if int(s["label"]) == 0]) for k in EMKEYS}
    return c, u

# ---- load every (method, subset, size) score pair -------------------------------------------
data = {}
for name, stem, col in PT.METHODS:
    if name == "SelfCrit": continue
    for stg, dom, keys in PT.STRUCT:
        for k in keys:
            for size in (SA, SB):
                L = PT.leaves(k, a.split, size)
                if not L: continue
                M, U, EM_ = [], [], None
                bad = False
                for (cl, _), (ul, _) in L:
                    if stem == "__emmia__":
                        c, u = emmia(cl, size)
                        if c is None: bad = True; break
                        EM_ = (c, u) if EM_ is None else (
                            {q: np.concatenate([EM_[0][q], c[q]]) for q in EMKEYS},
                            {q: np.concatenate([EM_[1][q], u[q]]) for q in EMKEYS})
                    else:
                        sm, su = scores(cl, size, stem, col), scores(ul, size, stem, col)
                        if sm is None or su is None: bad = True; break
                        M.append(sm); U.append(su)
                if bad: continue
                if stem == "__emmia__":
                    data[(name, k, size)] = (EM_[0], EM_[1], len(EM_[0]["loss"]), len(EM_[1]["loss"]), True)
                else:
                    M, U = np.concatenate(M), np.concatenate(U)
                    data[(name, k, size)] = (M, U, len(M), len(U), False)

METH = sorted({n for n, _, _ in PT.METHODS if n != "SelfCrit"},
              key=[n for n, _, _ in PT.METHODS].index)
SUBSETS = [k for _, _, ks in PT.STRUCT for k in ks]
# which (method, subset) have equal counts at both sizes -> can be paired
nmatch = {}
for m in METH:
    for k in SUBSETS:
        A_, B_ = data.get((m, k, SA)), data.get((m, k, SB))
        if A_ and B_: nmatch[(m, k)] = (A_[2] == B_[2] and A_[3] == B_[3])
unpaired = sorted({k for (m, k), v in nmatch.items() if not v})
missing = sorted({(m, k) for m in METH for k in SUBSETS if (m, k) not in nmatch})
print("methods:", len(METH), METH)
print("subsets with counts that differ between sizes (resampled independently):", unpaired)
print("missing (method, subset):", missing)

def aggregate(getidx):
    out = {}
    for size in (SA, SB):
        per_m = []
        for m in METH:
            stages = {}
            for stg, dom, keys in PT.STRUCT:
                vals = []
                for k in keys:
                    d = data.get((m, k, size))
                    if d is None: continue
                    X, Y, nm, nu, is_em = d
                    im, iu = getidx(k, size, nm, nu, nmatch.get((m, k), False))
                    if is_em: vals.append(max(auc(X[q][im], Y[q][iu]) for q in EMKEYS))
                    else: vals.append(auc(X[im], Y[iu]))
                if vals: stages.setdefault(stg, []).append(sum(vals) / len(vals))
            if len(stages) == 3:
                per_m.append(sum(sum(v) / len(v) for v in stages.values()) / 3)
        out[size] = sum(per_m) / len(per_m)
    return out

point = aggregate(lambda k, sz, nm, nu, p: (np.arange(nm), np.arange(nu)))
print(f"point: {SA}={point[SA]:.4f}  {SB}={point[SB]:.4f}  delta={point[SB]-point[SA]:+.4f}")

ds = []
for b in range(a.boot):
    cache = {}
    def getidx(k, sz, nm, nu, paired, _c=cache):
        key = (k, nm, nu, "P" if paired else sz)
        if key not in _c: _c[key] = (rng.integers(0, nm, nm), rng.integers(0, nu, nu))
        return _c[key]
    g = aggregate(getidx); ds.append(g[SB] - g[SA])
ds = np.array(ds)
lo95, hi95 = np.quantile(ds, [0.025, 0.975]); lo99, hi99 = np.quantile(ds, [0.005, 0.995])
print(f"delta boot mean={ds.mean():+.4f}  95% CI [{lo95:+.4f}, {hi95:+.4f}]  99% CI [{lo99:+.4f}, {hi99:+.4f}]")
print(f"P(delta<=0) = {float((ds <= 0).mean()):.4f}")
json.dump(dict(split=a.split, sizes=[SA, SB], methods=METH, unpaired_subsets=unpaired,
               point={SA: point[SA], SB: point[SB]}, delta=point[SB] - point[SA],
               ci95=[float(lo95), float(hi95)], ci99=[float(lo99), float(hi99)],
               p_le0=float((ds <= 0).mean()), B=a.boot),
          open(LOG / f"size_pair_boot_{SA}_{SB}_{a.split}.json", "w"), indent=1)
