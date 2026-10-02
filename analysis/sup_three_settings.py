"""Sup-CAMIA under the three evaluation settings (2026-09-26). The CLASSIFIER changes per setting, not only the threshold:
  given the domain : LR trained on that domain's dev            -> macro AUC + accuracy (2-fold threshold within the domain)
  domain unknown   : LR trained on the pooled dev of ALL 9      -> pooled AUC + accuracy (one threshold on the pooled draw)
  unseen domain    : LR trained on the OTHER 8 domains (LODO)   -> macro AUC + accuracy (threshold from the other 8)
Signals: OLMo-Detect/results_camia_xdomain/signals_<size>.pkl (per domain: con_dev/unc_dev/con_test/unc_test, 71 keys).
Subsampling matches analysis/subsample_pooled.py (n per side per domain, reps draws). -> analysis/logs/sup_three_settings.json
"""
import sys, json, pickle, argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "OLMo-Detect"))
from methods.camia_lr_method import _fit_score_lr
from camia_xdomain_pooled import union_concat
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); SIZES = ["1b", "7b", "13b", "32b"]; rng = np.random.default_rng(a.seed)

def best_thr(y, s):
    o = np.argsort(s); ys = np.array(y)[o]; ss = np.array(s)[o]
    n1 = ys.sum(); n0 = len(ys) - n1
    tp = n1 - np.cumsum(ys); tn = np.cumsum(1 - ys)          # predict member if score > cut
    acc = (tp + tn) / len(ys); i = int(np.argmax(acc))
    return float(ss[i]), float(acc[i])
def acc_at(y, s, t): return float(np.mean((np.array(s) > t).astype(int) == np.array(y)))

out = {}
for size in SIZES:
    S = pickle.load(open(ROOT.parent / f"OLMo-Detect/results_camia_xdomain/signals_{size}.pkl", "rb"))
    doms = sorted(S)
    sc = {}   # dom -> setting -> (scores_con, scores_unc)
    for T in doms:
        others = [d for d in doms if d != T]
        in_c, in_u = S[T]["con_dev"], S[T]["unc_dev"]
        po_c = union_concat([S[d]["con_dev"] for d in doms]); po_u = union_concat([S[d]["unc_dev"] for d in doms])
        lo_c = union_concat([S[d]["con_dev"] for d in others] + [S[d]["con_test"] for d in others])
        lo_u = union_concat([S[d]["unc_dev"] for d in others] + [S[d]["unc_test"] for d in others])
        r = {}
        for tag, (C, U) in (("in", (in_c, in_u)), ("pooled", (po_c, po_u)), ("lodo", (lo_c, lo_u))):
            s_con, _, _ = _fit_score_lr(C, U, S[T]["con_test"]); s_unc, _, _ = _fit_score_lr(C, U, S[T]["unc_test"])
            r[tag] = (np.asarray(s_con, float), np.asarray(s_unc, float))
        sc[T] = r
    macro_in, pooled_un, macro_lo, acc_in, acc_un, acc_lo = [], [], [], [], [], []
    for rep in range(a.reps):
        draw = {T: (rng.choice(len(sc[T]["in"][0]), min(a.n, len(sc[T]["in"][0])), replace=False),
                    rng.choice(len(sc[T]["in"][1]), min(a.n, len(sc[T]["in"][1])), replace=False)) for T in doms}
        def ys(T, tag):
            ci, ui = draw[T]; c, u = sc[T][tag]
            return [1] * len(ci) + [0] * len(ui), np.concatenate([c[ci], u[ui]])
        macro_in.append(np.mean([roc_auc_score(*ys(T, "in")) for T in doms]))
        macro_lo.append(np.mean([roc_auc_score(*ys(T, "lodo")) for T in doms]))
        Y = sum((ys(T, "pooled")[0] for T in doms), []); X = np.concatenate([ys(T, "pooled")[1] for T in doms])
        pooled_un.append(roc_auc_score(Y, X))
        accs = []
        for T in doms:                                    # in-domain threshold, 2-fold
            y, s = ys(T, "in"); y = np.array(y); idx = rng.permutation(len(y)); h = len(y) // 2
            for tr, te in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
                t, _ = best_thr(y[tr], s[tr]); accs.append(acc_at(y[te], s[te], t))
        acc_in.append(np.mean(accs))
        t_un, _ = best_thr(Y, X); acc_un.append(np.mean([acc_at(*ys(T, "pooled"), t_un) for T in doms]))
        al = []
        for T in doms:                                    # LODO: threshold from the other 8 (LODO classifier)
            oth = [d for d in doms if d != T]
            Yo = sum((ys(d, "lodo")[0] for d in oth), []); Xo = np.concatenate([ys(d, "lodo")[1] for d in oth])
            t, _ = best_thr(Yo, Xo); al.append(acc_at(*ys(T, "lodo"), t))
        acc_lo.append(np.mean(al))
    out[size] = dict(macro_in=float(np.mean(macro_in)), acc_in=float(np.mean(acc_in)), pooled_unknown=float(np.mean(pooled_un)),
                     acc_unknown=float(np.mean(acc_un)), macro_lodo=float(np.mean(macro_lo)), acc_lodo=float(np.mean(acc_lo)))
    print(f"{size:4s} given: AUC {out[size]['macro_in']:.3f} acc {out[size]['acc_in']:.3f} | unknown: AUC {out[size]['pooled_unknown']:.3f} acc {out[size]['acc_unknown']:.3f} | unseen: AUC {out[size]['macro_lodo']:.3f} acc {out[size]['acc_lodo']:.3f}", flush=True)
m = lambda k: float(np.mean([out[s][k] for s in SIZES]))
out["Model-Avg"] = {k: m(k) for k in ("macro_in", "acc_in", "pooled_unknown", "acc_unknown", "macro_lodo", "acc_lodo")}
print("AVG ", {k: round(v, 3) for k, v in out["Model-Avg"].items()})
json.dump(out, open(ROOT / "analysis/logs/sup_three_settings.json", "w"), indent=1)
