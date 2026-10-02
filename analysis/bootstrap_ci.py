"""Nonparametric bootstrap CIs for AUC and TPR@5%FPR (2026-09-26).
For every (method, domain, model size): resample members and non-members with replacement B times,
recompute AUC and TPR@5%FPR, report the 2.5th/97.5th percentiles. Scores come from results_latest
(EMMIA from results_iclr/emmia_raw, best of its four base scores; supervised from their own dirs),
pooling the subsets of a domain exactly as the main tables do.
Usage: ../OLMo-Detect/.venv/bin/python analysis/bootstrap_ci.py [--boot 2000] -> analysis/logs/bootstrap_ci.json
"""
import sys, json, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import paper_tables as PT
ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="matched"); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(a.seed)
EMKEYS = ("loss", "zlib", "mink", "minkpp")
SUP = [("Sup-CAMIA", "camia_lr", "camia_lr_score"), ("MIA-Tuner", "mia_tuner", "mia_tuner_score"), ("FSD (PPL)", "fsd", "fsd_loss_score")]

def auc_tpr(m, u, fpr=0.05):
    """rank AUC (ties 1/2) + TPR at the given FPR (threshold = (1-fpr) quantile of non-members)"""
    x = np.concatenate([m, u]); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = len(m), len(u)
    A = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    t = np.quantile(u, 1 - fpr)
    return float(A), float(np.mean(m > t))

def scores(leaf, size, stem, col):
    p = PT.RL / leaf / PT.SD[size] / f"{stem}.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text()).get("result", {}).get("samples", [])
    v = [float(s[col]) for s in ss if PT.valid(s.get(col))]
    return np.array(v) if v else None

def emmia(con_leaf, size):
    tag = PT.emmia_tag(con_leaf); p = PT.EM / tag / f"emmia_{size}.json"
    if not p.exists(): p = PT.EM / (tag + "_sub300") / f"emmia_{size}.json"
    if not p.exists(): return None, None
    ss = json.loads(p.read_text()).get("samples", [])
    c = {k: np.array([float(s["final_scores"][k]) for s in ss if int(s["label"]) == 1]) for k in EMKEYS}
    u = {k: np.array([float(s["final_scores"][k]) for s in ss if int(s["label"]) == 0]) for k in EMKEYS}
    return c, u

out = {}
METHODS = [(n, s, c) for n, s, c in PT.METHODS] + SUP
for name, stem, col in METHODS:
    for stg, dom, keys in PT.STRUCT:
        for size in PT.SIZES:
            M, U, EM_ = [], [], None
            ok = True
            for k in keys:
                L = PT.leaves(k, a.split, size)
                if not L: ok = False; break
                for (cl, _), (ul, _) in L:
                    if stem == "__emmia__":
                        c, u = emmia(cl, size)
                        if c is None: ok = False; break
                        EM_ = (c, u) if EM_ is None else ({kk: np.concatenate([EM_[0][kk], c[kk]]) for kk in EMKEYS},
                                                          {kk: np.concatenate([EM_[1][kk], u[kk]]) for kk in EMKEYS})
                    else:
                        sm, su = scores(cl, size, stem, col), scores(ul, size, stem, col)
                        if sm is None or su is None: ok = False; break
                        M.append(sm); U.append(su)
                if not ok: break
            if not ok: continue
            if stem == "__emmia__":
                if EM_ is None: continue
                cM, cU = EM_; nM, nU = len(cM["loss"]), len(cU["loss"])
            else:
                if not M: continue
                M = np.concatenate(M); U = np.concatenate(U); nM, nU = len(M), len(U)
            if nM < 5 or nU < 5: continue
            As, Ts = [], []
            for b in range(a.boot):
                im = rng.integers(0, nM, nM); iu = rng.integers(0, nU, nU)
                if stem == "__emmia__":
                    best = max(((auc_tpr(cM[k][im], cU[k][iu])) for k in EMKEYS), key=lambda z: z[0])
                    As.append(best[0]); Ts.append(best[1])
                else:
                    A, T = auc_tpr(M[im], U[iu]); As.append(A); Ts.append(T)
            As = np.array(As); Ts = np.array(Ts)
            if stem == "__emmia__":
                obs = max((auc_tpr(cM[k], cU[k]) for k in EMKEYS), key=lambda z: z[0])
            else:
                obs = auc_tpr(M, U)
            out[f"{name}|{dom}|{size}"] = dict(auc=obs[0], auc_lo=float(np.percentile(As, 2.5)), auc_hi=float(np.percentile(As, 97.5)),
                                               tpr=obs[1], tpr_lo=float(np.percentile(Ts, 2.5)), tpr_hi=float(np.percentile(Ts, 97.5)),
                                               n_mem=int(nM), n_non=int(nU), B=int(a.boot))
    print(f"{name}: {sum(1 for k in out if k.startswith(name + '|'))} cells", flush=True)
(ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
json.dump(out, open(ROOT / f"analysis/logs/bootstrap_ci_{a.split}.json", "w"), indent=1)
print("cells:", len(out), "->", f"analysis/logs/bootstrap_ci_{a.split}.json")
