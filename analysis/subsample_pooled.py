"""Subsampling experiment (NOT a main-paper table).

For every domain (9 domains of STRUCT in paper_tables.py; multi-subset domains are pooled over their subsets) and every model size,
draw N_PER_SIDE members + N_PER_SIDE non-members uniformly without replacement (same instances for every method), then report
  * POOLED AUC : one AUC over the union of all 9 domains' subsampled instances (raw scores, no per-domain normalisation)
  * MACRO AUC  : mean over the 9 domains of the per-domain AUC on the same subsample (reference; paper uses macro on the full data)
averaged over R random draws (mean, std). Scores come from results_latest (FINAL dir) exactly like analysis/paper_tables.py; EMMIA from
results_iclr/emmia_raw (best of its 4 base scores, chosen on the pooled/macro objective respectively, as the paper does per subset);
SelfCrit exists only for DPO/RLVR (pooled over those two domains; DPO SelfCrit is pair-level -> the pair score is used for both the chosen and
the rejected instance, mapped by record position).
Instance keys: text-bearing files (FAST/supervised) are matched by stripped text; id-only files (neighborhood, dcq, cdd, guided) by sample_id,
where the canonical id of an instance is its loss-file sample_id, replaced by the benchmark id (looked up by text) in the StarCoder leaves whose
patched records carry generic ids. EMMIA samples carry only a positional index (members first, then non-members) -> mapped through
record_index where unique, else through the benchmark file order (by text).

Usage: ../OLMo-Detect/.venv/bin/python analysis/subsample_pooled.py [--n 100] [--reps 20] [--split matched]
Outputs: tables/table:subsample_pooled.tex, analysis/logs/subsample_pooled.json, markdown to stdout.
"""
import sys, json, math, argparse, glob
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--split", default="matched")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
RL, RI, EM, SD, SIZES, B = PT.RL, PT.RI, PT.EM, PT.SD, PT.SIZES, PT.B
ROOT = Path(__file__).resolve().parents[1]

UNSUP = PT.METHODS
SUP = [("Sup-CAMIA", "camia_lr", "camia_lr_score"), ("MIA-Tuner", "mia_tuner", "mia_tuner_score"), ("FSD (MinK++)", "fsd", "fsd_minkpp_score")]
EMKEYS = ("loss", "zlib", "mink", "minkpp")

_cache = {}
def T(s): return PT.rec_key(s)

def samples(leaf, size, stem):
    k = ("__s__", leaf, size, stem)
    if k in _cache: return _cache[k]
    out = []
    for D in ((RL, RI) if stem == "selfcrit" else (RL,)):
        p = D / leaf / SD[size] / f"{stem}.json"
        if p.exists(): out = json.loads(p.read_text()).get("result", {}).get("samples", []); break
    _cache[k] = out; return out

def scores(leaf, size, stem, col):
    """-> ('text'|'id', {key: score}); text-keyed when the file stores texts, else keyed by sample_id"""
    k = (leaf, size, stem, col)
    if k in _cache: return _cache[k]
    ss = samples(leaf, size, stem)
    if ss and T(ss[0]): out = ("text", {T(s): float(s[col]) for s in ss if PT.valid(s.get(col)) and T(s)})
    else: out = ("id", {str(s["sample_id"]): float(s[col]) for s in ss if PT.valid(s.get(col))})
    _cache[k] = out; return out

def bench_file(leaf):
    """benchmark jsonl for a leaf dir name (needed for StarCoder: canonical ids + EMMIA line order)"""
    m = "shifted" if leaf.endswith("_shifted") else "matched"
    if "starcoder" in leaf:
        sub = "python" if "python" in leaf else "java"
        g = B / f"pretraining/starcoder/{'member/'+m if leaf.startswith('member') else 'non-member'}/test/*{sub}*.jsonl"
        fs = glob.glob(str(g)); return fs[0] if len(fs) == 1 else None
    return None

def bench_index(leaf):
    """{text: (line, id)} from the benchmark file (StarCoder only), else None"""
    k = ("__bi__", leaf)
    if k in _cache: return _cache[k]
    bf = bench_file(leaf); out = None
    if bf:
        out = {}
        for n, l in enumerate(x for x in open(bf) if x.strip()):
            r = json.loads(l); t = PT.text_key(r.get("text")); out.setdefault(t, (n, str(r.get("id", r.get("sample_id", f"sample_{n}")))))
    _cache[k] = out; return out

def universe(leaf, size):
    """[(text, canonical_id, record_index)] of the leaf (loss file), canonical id from the benchmark for StarCoder"""
    k = ("__u__", leaf, size)
    if k in _cache: return _cache[k]
    bi = bench_index(leaf); out = []
    for s in samples(leaf, size, "loss_zlib_lowercase"):
        if not PT.valid(s.get("loss_score")): continue
        t = T(s); cid = str(s["sample_id"])
        if bi is not None and t in bi: cid = bi[t][1]
        elif bi is not None: print(f"[warn] {leaf}/{size}: text not in benchmark for id {cid}", file=sys.stderr)
        out.append((t, cid, int(s["record_index"])))
    _cache[k] = out; return out

def line_index(leaf, size):
    """{text: benchmark line index} (EMMIA positional index) via record_index if unique, else via the benchmark file order (by text)"""
    k = ("__li__", leaf, size)
    if k in _cache: return _cache[k]
    U = universe(leaf, size); ri = [r for _, _, r in U]
    if len(set(ri)) == len(ri): out = {t: r for t, _, r in U}
    else:
        bi = bench_index(leaf); out = {t: bi[t][0] for t, _, _ in U if bi and t in bi}
        if len(out) != len(U): print(f"[warn] EMMIA line map {leaf}/{size}: {len(out)}/{len(U)}", file=sys.stderr)
    _cache[k] = out; return out

def emmia_scores(con_leaf, size):
    """{(label, line_index): {key: score}}"""
    k = ("__em__", con_leaf, size)
    if k in _cache: return _cache[k]
    p = EM / PT.emmia_tag(con_leaf) / f"emmia_{size}.json"
    if not p.exists() and (EM/(PT.emmia_tag(con_leaf)+"_sub300")/f"emmia_{size}.json").exists(): p=EM/(PT.emmia_tag(con_leaf)+"_sub300")/f"emmia_{size}.json"   # 2026-09-25 SO shifted subsample
    out = {}
    if p.exists():
        ss = json.loads(p.read_text()).get("samples", [])
        n_con = sum(1 for s in ss if int(s["label"]) == 1)   # EMMIA indexes non-members after the members (index = n_con + line)
        for s in ss:
            lab = int(s["label"]); idx = int(s["index"]) - (0 if lab == 1 else n_con)
            out[(lab, idx)] = {kk: float(v) for kk, v in s["final_scores"].items()}
        nl = len(universe(con_leaf, size))
        if n_con != nl: print(f"[warn] EMMIA {con_leaf}/{size}: {n_con} members in emmia file vs {nl} in leaf", file=sys.stderr)
    _cache[k] = out; return out

def domain_instances(dom_keys, size):
    """-> (con:[(leaf, text, cid, rec)], unc:[...]) universe = instances scored by loss_zlib_lowercase (complete everywhere)."""
    con, unc = [], []
    for key in dom_keys:
        lv = PT.leaves(key, a.split, size)
        if lv is None: continue
        for (cl, cn), (ul, un) in lv:
            c = universe(cl, size); u = universe(ul, size)
            if len(c) != cn or len(u) != un: print(f"[warn] universe incomplete {cl}/{size}: {len(c)}/{cn} , {len(u)}/{un}", file=sys.stderr)
            ordr = lambda x: (x[2], x[1])
            con += [(cl,) + x for x in sorted(c, key=ordr)]; unc += [(ul,) + x for x in sorted(u, key=ordr)]
    return con, unc

def selfcrit_pair(leaf, size):
    """{record_index: score} of the DPO pair-level SelfCrit file"""
    m = "matched" if a.split == "matched" else "shifted"
    pl = f"member_posttraining_dpo_{size}_{m}" if leaf.startswith("member") else f"non-member_posttraining_dpo_{size}"
    k = ("__scp__", pl, size)
    if k in _cache: return _cache[k]
    out = {int(s["record_index"]): float(s["self_critique_score"]) for s in samples(pl, size, "selfcrit") if PT.valid(s.get("self_critique_score"))}
    _cache[k] = out; return out

def method_scores(name, stem, col, inst, side, size, con_leaf_of):
    """scores for a list of (leaf, text, cid, rec) instances; list of (score or None) [EMMIA: dict key->score]"""
    out = []
    for leaf, t, cid, rec in inst:
        if stem == "__emmia__":
            em = emmia_scores(con_leaf_of[leaf], size); li = line_index(leaf, size).get(t)
            out.append(None if li is None else em.get((1 if side == "con" else 0, li)))
        elif stem == "selfcrit" and "_dpo_" in leaf:
            out.append(selfcrit_pair(leaf, size).get(rec))
        else:
            kind, sc = scores(leaf, size, stem, col); out.append(sc.get(t if kind == "text" else cid))
    return out

def auc(y, s):
    if len(set(y)) < 2: return None
    return float(roc_auc_score(y, s))

def main():
    rng = np.random.default_rng(a.seed)
    methods = [(n, s, c, "unsup") for n, s, c in UNSUP] + [(n, s, c, "sup") for n, s, c in SUP]
    res = {}   # (name,size) -> dict(pooled=[...], macro=[...], missing=frac)
    for size in SIZES:
        doms = []
        for stg, dom, keys in PT.STRUCT:
            con, unc = domain_instances(keys, size)
            if not con or not unc: continue
            con_leaf_of = {}
            for key in keys:
                for (cl, cn), (ul, un) in (PT.leaves(key, a.split, size) or []): con_leaf_of[cl] = cl; con_leaf_of[ul] = cl
            doms.append((dom, con, unc, con_leaf_of))
        draws = []
        for r in range(a.reps):
            d = []
            for dom, con, unc, clo in doms:
                ci = rng.choice(len(con), size=min(a.n, len(con)), replace=False); ui = rng.choice(len(unc), size=min(a.n, len(unc)), replace=False)
                d.append((dom, [con[i] for i in ci], [unc[i] for i in ui], clo))
            draws.append(d)
        for name, stem, col, kind in methods:
            pooled, macro, miss, tot = [], [], 0, 0
            for d in draws:
                per_dom = []   # (dom, y, scores-or-dicts)
                for dom, cs, us, clo in d:
                    if name == "SelfCrit" and dom not in PT.SELFCRIT_DOMS: continue
                    sc = method_scores(name, stem, col, cs, "con", size, clo); su = method_scores(name, stem, col, us, "unc", size, clo)
                    y = [1] * len(sc) + [0] * len(su); s = sc + su
                    keep = [(yy, ss) for yy, ss in zip(y, s) if ss is not None]
                    miss += len(s) - len(keep); tot += len(s)
                    if keep: per_dom.append((dom, [k[0] for k in keep], [k[1] for k in keep]))
                if not per_dom: continue
                if stem == "__emmia__":
                    Y = sum((y for _, y, _ in per_dom), []); best = None
                    for kk in EMKEYS:
                        S = sum(([ss[kk] for ss in s] for _, _, s in per_dom), []); v = auc(Y, S)
                        if v is not None and (best is None or v > best): best = v
                    if best is not None: pooled.append(best)
                    ms = []
                    for _, y, s in per_dom:
                        vs = [auc(y, [ss[kk] for ss in s]) for kk in EMKEYS]; vs = [v for v in vs if v is not None]
                        if vs: ms.append(max(vs))
                    if ms: macro.append(float(np.mean(ms)))
                else:
                    Y = sum((y for _, y, _ in per_dom), []); S = sum((s for _, _, s in per_dom), [])
                    v = auc(Y, S)
                    if v is not None: pooled.append(v)
                    ms = [auc(y, s) for _, y, s in per_dom]; ms = [m for m in ms if m is not None]
                    if ms: macro.append(float(np.mean(ms)))
            res[(name, size)] = dict(pooled=pooled, macro=macro, missing=(miss / tot if tot else 1.0), kind=kind)
            print(f"{name:14s} {size:4s} pooled={np.mean(pooled) if pooled else float('nan'):.3f}±{np.std(pooled) if pooled else 0:.3f} "
                  f"macro={np.mean(macro) if macro else float('nan'):.3f} missing={100*res[(name,size)]['missing']:.1f}%", file=sys.stderr)
    # ---- outputs ----
    def cell(v, flag):
        return "---" if v is None or not len(v) else f"{np.mean(v):.2f}" + ("$^{*}$" if flag else "")
    js = {f"{n}|{s}": dict(pooled_mean=float(np.mean(v["pooled"])) if v["pooled"] else None, pooled_std=float(np.std(v["pooled"])) if v["pooled"] else None,
                          macro_mean=float(np.mean(v["macro"])) if v["macro"] else None, macro_std=float(np.std(v["macro"])) if v["macro"] else None,
                          missing_frac=v["missing"]) for (n, s), v in res.items()}
    (ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
    (ROOT / "analysis/logs/subsample_pooled.json").write_text(json.dumps(dict(n_per_side=a.n, reps=a.reps, split=a.split, seed=a.seed, cells=js), indent=1))
    names = [m[0] for m in methods]
    md = ["| Method | " + " | ".join(f"Pooled {s}" for s in SIZES) + " | Pooled Avg | " + " | ".join(f"Macro {s}" for s in SIZES) + " | Macro Avg |",
          "|---|" + "---|" * 10]
    tex = []
    for name in names:
        pv = [res[(name, s)]["pooled"] for s in SIZES]; mv = [res[(name, s)]["macro"] for s in SIZES]
        fl = [res[(name, s)]["missing"] > 0.01 for s in SIZES]
        pa = [np.mean(v) for v in pv if len(v)]; ma = [np.mean(v) for v in mv if len(v)]
        cells = [cell(v, f) for v, f in zip(pv, fl)] + [f"{np.mean(pa):.2f}" if pa else "---"] + [cell(v, f) for v, f in zip(mv, fl)] + [f"{np.mean(ma):.2f}" if ma else "---"]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
        tex.append(f"{name} & " + " & ".join(cells) + " \\\\")
    for kind, lab in (("unsup", "Method-Avg (unsup.)"),):
        row = []
        for agg in ("pooled", "macro"):
            per = []
            for s in SIZES:
                vals = [np.mean(res[(n, s)][agg]) for n, st, c, k in methods if k == kind and len(res[(n, s)][agg])]
                per.append(np.mean(vals) if vals else None)
            row += [("---" if v is None else f"{v:.2f}") for v in per] + [f"{np.mean([v for v in per if v is not None]):.2f}"]
        md.append(f"| **{lab}** | " + " | ".join(row) + " |"); tex.append(f"\\midrule \\textbf{{{lab}}} & " + " & ".join(row) + " \\\\")
    nu = len(UNSUP)
    body = tex[:nu] + ["\\midrule"] + tex[nu:nu + len(SUP)] + tex[nu + len(SUP):]
    stds = [v["pooled_std"] for v in js.values() if v["pooled_std"] is not None]
    caption = (f"Subsampling experiment (not in the main paper): every domain is subsampled to {a.n} members + {a.n} non-members "
               f"(multi-subset domains pooled over their subsets; identical instances for all methods), repeated {a.reps} times. "
               f"\\textbf{{Pooled}} = one AUC over the union of all nine domains' subsampled instances (raw scores); \\textbf{{Macro}} = mean of the nine per-domain AUCs on the same subsample "
               f"(the paper's aggregation, computed on the subsample for reference). Cells are means over the {a.reps} draws (std over draws $\\le$ {max(stds):.2f}); "
               f"Avg = mean over model sizes. $^{{*}}$ = method scored $<$99\\% of the drawn instances (partial-coverage methods: DCQ, Guided, SelfCrit on DPO/RLVR only). "
               f"EMMIA uses the best of its four base scores for each objective. Supervised rows are the in-domain setting.")
    latex = ("\\begin{table*}[t]\n\\centering\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l" + "c" * 10 + "}\n\\toprule\n"
             "& \\multicolumn{5}{c}{\\textbf{Pooled AUC (subsampled)}} & \\multicolumn{5}{c}{\\textbf{Macro AUC (subsampled)}} \\\\\n"
             "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\nMethod & 1B & 7B & 13B & 32B & Avg & 1B & 7B & 13B & 32B & Avg \\\\\n\\midrule\n"
             + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\caption{" + caption + "}\n\\label{table:subsample_pooled}\n\\end{table*}\n")
    out = ROOT / "tables/table:subsample_pooled.tex"; out.write_text(latex)
    print("\n".join(md)); print(f"\nTEX -> {out}\nJSON -> {ROOT/'analysis/logs/subsample_pooled.json'}")

if __name__ == "__main__":
    main()
