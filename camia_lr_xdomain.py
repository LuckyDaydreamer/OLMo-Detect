#!/usr/bin/env python
"""
CAMIA-LR (SUPERVISED, Appendix B) cross-domain generalization via LODO, offline from cache.

Reuses the cached per-domain signal dicts (results_camia_xdomain/signals_<sz>.pkl; keys
con_dev/unc_dev/con_test/unc_test) and the exact LR core (methods.camia_lr_method._fit_score_lr),
so no forward passes are needed. Two settings per held-out domain T (scored on T's test):

  in-domain : train LR on T's OWN dev  (members=con_dev[T],  non-members=unc_dev[T])
  LODO      : train LR on ALL OTHER domains' labeled data (dev+test), T never seen in training

Reports plain-LR and LR+PCA, per-domain + macro-average + pooled, and the in-domain − LODO gap
(how much the supervised attack loses on an unseen domain). Usage:
  python camia_lr_xdomain.py --size 1b
"""
import os, json, pickle, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from camia_xdomain_pooled import union_concat, STAGES
from methods.camia_lr_method import _fit_score_lr


def _auc(labels, scores):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    m = np.isfinite(scores)
    if m.sum() < 2 or len(set(labels[m].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels[m], scores[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1b")
    args = ap.parse_args()
    cache = f"results_camia_xdomain/signals_{args.size}.pkl"
    if not os.path.exists(cache):
        raise SystemExit(f"missing cache {cache} (only sizes with cached signals can run offline)")
    S = pickle.load(open(cache, "rb"))
    domains = [d for st in STAGES.values() for d in st if d in S]

    rows = {}
    for T in domains:
        con_t, unc_t = S[T]["con_test"], S[T]["unc_test"]
        target = union_concat([con_t, unc_t])
        n_c = len(next(iter(con_t.values()))); n_u = len(next(iter(unc_t.values())))
        lab = np.r_[np.ones(n_c), np.zeros(n_u)]

        others = [C for C in domains if C != T]
        pooled_con = union_concat([S[C]["con_dev"] for C in others] + [S[C]["con_test"] for C in others])
        pooled_unc = union_concat([S[C]["unc_dev"] for C in others] + [S[C]["unc_test"] for C in others])

        lr_lodo, pca_lodo, _ = _fit_score_lr(pooled_con, pooled_unc, target)
        lr_in,   pca_in,   _ = _fit_score_lr(S[T]["con_dev"], S[T]["unc_dev"], target)
        rows[T] = {"in_lr": _auc(lab, lr_in),   "in_pca": _auc(lab, pca_in),
                   "lodo_lr": _auc(lab, lr_lodo), "lodo_pca": _auc(lab, pca_lodo),
                   "_lodo_lr_scores": lr_lodo, "_in_lr_scores": lr_in, "_lab": lab}

    def macro(k):
        v = [rows[d][k] for d in domains if np.isfinite(rows[d][k])]
        return float(np.mean(v)) if v else float("nan")

    def pooled(kscore):
        sc = np.concatenate([rows[d][kscore] for d in domains])
        lb = np.concatenate([rows[d]["_lab"] for d in domains])
        return _auc(lb, sc)

    print("=" * 78)
    print(f"CAMIA-LR (SUPERVISED) cross-domain — {args.size}, matched, per-domain AUC")
    print("=" * 78)
    print(f"{'domain':14s} {'in:LR':>7s} {'in:PCA':>7s} {'LODO:LR':>8s} {'LODO:PCA':>9s} {'Δ(in−LODO)LR':>13s}")
    for T in domains:
        r = rows[T]
        print(f"{T:14s} {r['in_lr']:7.3f} {r['in_pca']:7.3f} {r['lodo_lr']:8.3f} "
              f"{r['lodo_pca']:9.3f} {r['in_lr']-r['lodo_lr']:+13.3f}")
    print("-" * 78)
    print(f"{'MACRO-avg':14s} {macro('in_lr'):7.3f} {macro('in_pca'):7.3f} "
          f"{macro('lodo_lr'):8.3f} {macro('lodo_pca'):9.3f} {macro('in_lr')-macro('lodo_lr'):+13.3f}")
    print(f"{'POOLED':14s} {pooled('_in_lr_scores'):7.3f} {'':7s} {pooled('_lodo_lr_scores'):8.3f}")

    out = {"size": args.size,
           "per_domain": {T: {k: rows[T][k] for k in ("in_lr","in_pca","lodo_lr","lodo_pca")}
                          for T in domains},
           "macro": {k: macro(k) for k in ("in_lr","in_pca","lodo_lr","lodo_pca")},
           "pooled": {"in_lr": pooled("_in_lr_scores"), "lodo_lr": pooled("_lodo_lr_scores")}}
    os.makedirs("results_camia_xdomain", exist_ok=True)
    json.dump(out, open(f"results_camia_xdomain/camia_lr_xdomain_{args.size}.json", "w"), indent=2)
    print(f"\nwrote results_camia_xdomain/camia_lr_xdomain_{args.size}.json")


if __name__ == "__main__":
    main()
