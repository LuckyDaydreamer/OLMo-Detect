"""MIA-Tuner under the three evaluation settings (2026-09-26), analogous to analysis/sup_three_settings.py.
The soft prompt is retrained per setting, not merely re-thresholded:
  given the domain : trained on that domain's dev        -> results_latest/<leaf>/<model>/mia_tuner.json
  domain unknown   : trained on the pooled dev of ALL 9  -> results_mia_tuner_pooled9/
  unseen domain    : trained on the OTHER 8 (LODO)       -> results_mia_tuner_lodo_iclr/
Same 100+100 x 20-draw subsampling as analysis/subsample_pooled.py. -> analysis/logs/mt_three_settings.json
"""
import sys, json, argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils")); import paper_tables as PT
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed); SIZES = PT.SIZES
DIRS = {"in": PT.RL, "pooled": ROOT.parent / "OLMo-Detect/results_mia_tuner_pooled9", "lodo": ROOT.parent / "OLMo-Detect/results_mia_tuner_lodo_iclr"}
def best_thr(y, s):
    o = np.argsort(s); ys = np.asarray(y)[o]; ss = np.asarray(s)[o]
    n1 = ys.sum(); tp = n1 - np.cumsum(ys); tn = np.cumsum(1 - ys)
    acc = (tp + tn) / len(ys); i = int(np.argmax(acc)); return float(ss[i])
def acc_at(y, s, t): return float(np.mean((np.asarray(s) > t).astype(int) == np.asarray(y)))
def load(tag, leaf, size):
    p = DIRS[tag] / leaf / PT.SD[size] / "mia_tuner.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text()).get("result", {}).get("samples", [])
    v = [float(s["mia_tuner_score"]) for s in ss if PT.valid(s.get("mia_tuner_score"))]
    return np.array(v) if v else None
out = {}
for size in SIZES:
    sc = {}
    for stg, dom, keys in PT.STRUCT:
        r = {}
        for tag in DIRS:
            M, U = [], []
            ok = True
            for k in keys:
                L = PT.leaves(k, "matched", size)
                if not L: ok = False; break
                for (cl, _), (ul, _) in L:
                    m, u = load(tag, cl, size), load(tag, ul, size)
                    if m is None or u is None: ok = False; break
                    M.append(m); U.append(u)
                if not ok: break
            if not ok: break
            r[tag] = (np.concatenate(M), np.concatenate(U))
        if len(r) == 3: sc[dom] = r
    if not sc: print(f"{size}: no complete domain", flush=True); continue
    doms = sorted(sc)
    macro_in, pooled_un, macro_lo, acc_in, acc_un, acc_lo = [], [], [], [], [], []
    for rep in range(a.reps):
        draw = {T: (rng.choice(len(sc[T]["in"][0]), min(a.n, len(sc[T]["in"][0])), replace=False),
                    rng.choice(len(sc[T]["in"][1]), min(a.n, len(sc[T]["in"][1])), replace=False)) for T in doms}
        def ys(T, tag):
            ci, ui = draw[T]; c, u = sc[T][tag]
            ci = ci[ci < len(c)]; ui = ui[ui < len(u)]
            return [1]*len(ci) + [0]*len(ui), np.concatenate([c[ci], u[ui]])
        macro_in.append(np.mean([roc_auc_score(*ys(T, "in")) for T in doms]))
        macro_lo.append(np.mean([roc_auc_score(*ys(T, "lodo")) for T in doms]))
        Y = sum((ys(T, "pooled")[0] for T in doms), []); X = np.concatenate([ys(T, "pooled")[1] for T in doms])
        pooled_un.append(roc_auc_score(Y, X))
        accs = []
        for T in doms:
            y, s = ys(T, "in"); y = np.array(y); idx = rng.permutation(len(y)); h = len(y)//2
            for tr, te in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
                accs.append(acc_at(y[te], s[te], best_thr(y[tr], s[tr])))
        acc_in.append(np.mean(accs))
        t = best_thr(Y, X); acc_un.append(np.mean([acc_at(*ys(T, "pooled"), t) for T in doms]))
        al = []
        for T in doms:
            oth = [d for d in doms if d != T]
            Yo = sum((ys(d, "lodo")[0] for d in oth), []); Xo = np.concatenate([ys(d, "lodo")[1] for d in oth])
            al.append(acc_at(*ys(T, "lodo"), best_thr(Yo, Xo)))
        acc_lo.append(np.mean(al))
    out[size] = dict(n_domains=len(doms), macro_in=float(np.mean(macro_in)), acc_in=float(np.mean(acc_in)), pooled_unknown=float(np.mean(pooled_un)),
                     acc_unknown=float(np.mean(acc_un)), macro_lodo=float(np.mean(macro_lo)), acc_lodo=float(np.mean(acc_lo)))
    print(f"{size:4s} ({len(doms)} dom) given: AUC {out[size]['macro_in']:.3f} acc {out[size]['acc_in']:.3f} | unknown: AUC {out[size]['pooled_unknown']:.3f} acc {out[size]['acc_unknown']:.3f} | unseen: AUC {out[size]['macro_lodo']:.3f} acc {out[size]['acc_lodo']:.3f}", flush=True)
if out:
    m = lambda k: float(np.mean([out[s][k] for s in out]))
    out["Model-Avg"] = {k: m(k) for k in ("macro_in","acc_in","pooled_unknown","acc_unknown","macro_lodo","acc_lodo")}
    print("AVG ", {k: round(v,3) for k,v in out["Model-Avg"].items()})
json.dump(out, open(ROOT / "analysis/logs/mt_three_settings.json","w"), indent=1)
