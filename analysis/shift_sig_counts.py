"""Per-domain significance of the shift effect (2026-09-25): for every (method, domain) the size-averaged Delta = AUC_shifted - AUC_matched with a
paired instance bootstrap (members of each setting and the shared non-members resampled with replacement; same resample for both AUCs of a size).
Counts per method: I = domains with the CI entirely > 0, D = entirely < 0, at 95% and 99%. EMMIA: each run's own non-members, best of its 4 base scores per run.
Output analysis/logs/shift_sig_counts.json. Usage: python analysis/shift_sig_counts.py [--boot 2000]"""
import sys, json, argparse, numpy as np
from pathlib import Path
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent)); import paper_tables as PT
ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; RL, EM, SD, SIZES = PT.RL, PT.EM, PT.SD, PT.SIZES
rng = np.random.default_rng(a.seed)
SUB = [(k, stg, dom) for stg, dom, keys in PT.STRUCT for k in keys if PT.leaves(k, "shifted", "7b") is not None]
METHODS = [(n, s, c) for n, s, c in PT.METHODS if n != "SelfCrit"]
def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = rankdata(x); n1, n0 = len(pos), len(neg); return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def T(s): return PT.rec_key(s)
def sc(leaf, size, stem, col):
    p = RL / leaf / SD[size] / f"{stem}.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text()).get("result", {}).get("samples", [])
    if not ss: return None
    return np.array([float(s[col]) for s in ss if PT.valid(s.get(col))])
def emmia(con_leaf, size):
    tag = PT.emmia_tag(con_leaf); p = EM / tag / f"emmia_{size}.json"
    if not p.exists(): p = EM / (tag + "_sub300") / f"emmia_{size}.json"
    if not p.exists(): return None
    ss = json.loads(p.read_text()).get("samples", [])
    keys = ("loss", "zlib", "mink", "minkpp")
    c = np.array([[float(s["final_scores"][k]) for k in keys] for s in ss if int(s["label"]) == 1]); u = np.array([[float(s["final_scores"][k]) for k in keys] for s in ss if int(s["label"]) == 0])
    return c, u
out = {}
for name, stem, col in METHODS:
    res = {}
    for k, stg, dom in SUB:
        per = []
        for s in SIZES:
            L = PT.leaves(k, "matched", s); Ls = PT.leaves(k, "shifted", s)
            if not L or not Ls: continue
            if stem == "__emmia__":
                em = [emmia(cm, s) for (cm, _), _ in L]; es = [emmia(cs, s) for (cs, _), _ in Ls]
                if any(x is None for x in em + es): continue
                per.append(("emmia", np.concatenate([x[0] for x in em]), np.concatenate([x[1] for x in em]), np.concatenate([x[0] for x in es]), np.concatenate([x[1] for x in es])))
            else:
                M = [sc(cm, s, stem, col) for (cm, _), _ in L]; U = [sc(um, s, stem, col) for _, (um, _) in L]; S = [sc(cs, s, stem, col) for (cs, _), _ in Ls]
                if any(x is None or len(x) == 0 for x in M + U + S): continue
                per.append(("std", np.concatenate(M), np.concatenate(S), np.concatenate(U)))
        if not per: continue
        def delta(idx=None):
            vals = []
            for t in per:
                if t[0] == "std":
                    _, m, s_, u = t
                    if idx is not None: m = m[rng.integers(0, len(m), len(m))]; s_ = s_[rng.integers(0, len(s_), len(s_))]; u = u[rng.integers(0, len(u), len(u))]
                    vals.append(auc(s_, u) - auc(m, u))
                else:
                    _, mc, mu, sc_, su = t
                    if idx is not None: mc = mc[rng.integers(0, len(mc), len(mc))]; mu = mu[rng.integers(0, len(mu), len(mu))]; sc_ = sc_[rng.integers(0, len(sc_), len(sc_))]; su = su[rng.integers(0, len(su), len(su))]
                    am = max(auc(mc[:, j], mu[:, j]) for j in range(4)); as_ = max(auc(sc_[:, j], su[:, j]) for j in range(4)); vals.append(as_ - am)
            return float(np.mean(vals))
        obs = delta(); boots = np.array([delta(1) for _ in range(a.boot)])
        lo95, hi95 = np.percentile(boots, [2.5, 97.5]); lo99, hi99 = np.percentile(boots, [0.5, 99.5])
        res[k] = dict(delta=obs, ci95=(float(lo95), float(hi95)), ci99=(float(lo99), float(hi99)), boots=boots, dom=(stg, dom))
    cnt = {lv: (sum(1 for r in res.values() if r[lv][0] > 0), sum(1 for r in res.values() if r[lv][1] < 0)) for lv in ("ci95", "ci99")}
    # DOMAIN level (7 domains): mean of the subset deltas of a (stage, domain) per bootstrap draw (same draws -> paired)
    dres = {}
    for (stg, dom) in sorted(set(r["dom"] for r in res.values())):
        ks = [k for k, r in res.items() if r["dom"] == (stg, dom)]
        ob = float(np.mean([res[k]["delta"] for k in ks])); bb = np.mean([res[k]["boots"] for k in ks], axis=0)
        dres[dom] = dict(delta=ob, subsets=ks, ci95=tuple(map(float, np.percentile(bb, [2.5, 97.5]))), ci99=tuple(map(float, np.percentile(bb, [0.5, 99.5]))))
    dcnt = {lv: (sum(1 for r in dres.values() if r[lv][0] > 0), sum(1 for r in dres.values() if r[lv][1] < 0)) for lv in ("ci95", "ci99")}
    for r in res.values(): r.pop("boots"); r["dom"] = list(r["dom"])
    out[name] = dict(per_domain=res, I95=cnt["ci95"][0], D95=cnt["ci95"][1], I99=cnt["ci99"][0], D99=cnt["ci99"][1], n=len(res), mean_abs=float(np.mean([abs(r["delta"]) for r in res.values()])),
                     per_domain7=dres, I95_dom=dcnt["ci95"][0], D95_dom=dcnt["ci95"][1], I99_dom=dcnt["ci99"][0], D99_dom=dcnt["ci99"][1], n_dom=len(dres))
    print(f"{name:8s} subsets n={len(res):2d} I/D 95%={cnt['ci95'][0]}/{cnt['ci95'][1]} 99%={cnt['ci99'][0]}/{cnt['ci99'][1]} | DOMAINS n={len(dres)} I/D 95%={dcnt['ci95'][0]}/{dcnt['ci95'][1]} 99%={dcnt['ci99'][0]}/{dcnt['ci99'][1]} | mean|Δ|={out[name]['mean_abs']:.3f}", flush=True)
json.dump(out, open(ROOT / "analysis/logs/shift_sig_counts.json", "w"), indent=1); print("DONE")
