"""Threshold-transfer check across the three domain-knowledge settings. NO GPU; reuses analysis/subsample_pooled.py's loaders.

For every unsupervised method (EMMIA/SelfCrit excluded) x size, on the same 100+100-per-domain subsamples as table:subsample_pooled (R draws):
  acc_in   (setting 1 proxy): threshold tuned INSIDE the domain (2-fold split, both directions averaged) -> the domain's own dev set is not scored, so this is its unbiased proxy
  acc_all9 (setting 2)      : ONE threshold tuned on the pooled 9-domain subsample, applied to each domain (in-sample, i.e. tuned on the whole data)
  acc_lodo (setting 3)      : threshold tuned on the pooled OTHER-8-domain subsample, applied to the held-out domain
Thresholds maximise accuracy (predict member iff score >= t; score orientation = the paper's: higher = member). Accuracies are macro-averaged over the 9 domains.
Also reports, per method/size, the mean over domains of |acc_all9 - acc_lodo| and the max, i.e. how far setting 3 is from setting 2 for a training-free MIA.
Usage: ../OLMo-Detect/.venv/bin/python analysis/lodo_threshold_check.py [--n 100] [--reps 20] [--seed 0]
Outputs: analysis/logs/lodo_threshold_check.json, tables/table:threshold_transfer.tex, markdown on stdout.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
_argv = sys.argv[1:]; sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subsample_pooled as SP
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(_argv); SP.a.n, SP.a.reps, SP.a.seed, SP.a.split = a.n, a.reps, a.seed, "matched"
PT, SIZES, ROOT = SP.PT, SP.SIZES, SP.ROOT

def best_thr(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); n = len(y); P = y.sum(); N = n - P
    o = np.argsort(-s, kind="stable"); ys = y[o]; ss = s[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    acc = np.concatenate([[N / n], (tp + (N - fp)) / n]); k = int(np.argmax(acc))
    t = np.inf if k == 0 else (-np.inf if k == n else (ss[k - 1] + ss[k]) / 2)
    return float(t), float(acc[k])

def acc_at(y, s, t):
    y = np.asarray(y, int); s = np.asarray(s, float); return float(np.mean((s >= t).astype(int) == y))

def main():
    rng = np.random.default_rng(a.seed)
    methods = [(n, s, c) for n, s, c in SP.UNSUP if s != "__emmia__" and n != "SelfCrit"] + [(n, s, c) for n, s, c in SP.SUP if n != "FSD (MinK++)"]
    res = {}
    for size in SIZES:
        doms = []
        for stg, dom, keys in PT.STRUCT:
            con, unc = SP.domain_instances(keys, size)
            if not con or not unc: continue
            clo = {}
            for key in keys:
                for (cl, cn), (ul, un) in (PT.leaves(key, "matched", size) or []): clo[cl] = cl; clo[ul] = cl
            doms.append((dom, con, unc, clo))
        draws = []
        for r in range(a.reps):
            d = []
            for dom, con, unc, clo in doms:
                ci = rng.choice(len(con), size=min(a.n, len(con)), replace=False); ui = rng.choice(len(unc), size=min(a.n, len(unc)), replace=False)
                d.append((dom, [con[i] for i in ci], [unc[i] for i in ui], clo))
            draws.append(d)
        for name, stem, col in methods:
            per = {}   # dom -> dict(lists)
            for d in draws:
                pd = []
                for dom, cs, us, clo in d:
                    sc = SP.method_scores(name, stem, col, cs, "con", size, clo); su = SP.method_scores(name, stem, col, us, "unc", size, clo)
                    y = [1] * len(sc) + [0] * len(su); s = sc + su
                    keep = [(yy, ss) for yy, ss in zip(y, s) if ss is not None]
                    if len(keep) >= 20: pd.append((dom, np.array([k[0] for k in keep]), np.array([k[1] for k in keep], float)))
                if len(pd) < 3: continue
                Y = np.concatenate([y for _, y, _ in pd]); S = np.concatenate([s for _, _, s in pd]); t9, _ = best_thr(Y, S)
                for i, (dom, y, s) in enumerate(pd):
                    Yo = np.concatenate([yy for j, (_, yy, _) in enumerate(pd) if j != i]); So = np.concatenate([ss for j, (_, _, ss) in enumerate(pd) if j != i])
                    t8, _ = best_thr(Yo, So)
                    idx = rng.permutation(len(y)); h = len(y) // 2; A, B = idx[:h], idx[h:]
                    acc_in = np.mean([acc_at(y[B], s[B], best_thr(y[A], s[A])[0]), acc_at(y[A], s[A], best_thr(y[B], s[B])[0])])
                    r = per.setdefault(dom, dict(acc_in=[], acc_all9=[], acc_lodo=[], t9=[], t8=[], auc=[]))
                    r["acc_in"].append(acc_in); r["acc_all9"].append(acc_at(y, s, t9)); r["acc_lodo"].append(acc_at(y, s, t8)); r["t9"].append(t9); r["t8"].append(t8)
                    v = SP.auc(y.tolist(), s.tolist()); r["auc"].append(v if v is not None else np.nan)
            if not per: continue
            dm = {dom: {k: float(np.nanmean(v)) for k, v in r.items()} for dom, r in per.items()}
            diff = [abs(dm[d]["acc_all9"] - dm[d]["acc_lodo"]) for d in dm]
            res[(name, size)] = dict(domains=dm, acc_in=float(np.mean([dm[d]["acc_in"] for d in dm])), acc_all9=float(np.mean([dm[d]["acc_all9"] for d in dm])),
                                     acc_lodo=float(np.mean([dm[d]["acc_lodo"] for d in dm])), diff_mean=float(np.mean(diff)), diff_max=float(np.max(diff)),
                                     macro_auc=float(np.mean([dm[d]["auc"] for d in dm])), n_dom=len(dm))
            r = res[(name, size)]
            print(f"{name:8s} {size:4s} auc={r['macro_auc']:.3f} acc_in={r['acc_in']:.3f} acc_all9={r['acc_all9']:.3f} acc_lodo={r['acc_lodo']:.3f} |all9-lodo| mean={r['diff_mean']:.3f} max={r['diff_max']:.3f}", file=sys.stderr)
    (ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
    (ROOT / "analysis/logs/lodo_threshold_check.json").write_text(json.dumps({f"{n}|{s}": v for (n, s), v in res.items()}, indent=1))
    names = [m[0] for m in methods]
    md = ["| Method | " + " | ".join(f"{s} in/all9/lodo" for s in SIZES) + " | Avg in | Avg all9 | Avg lodo | mean abs(all9-lodo) | max abs(all9-lodo) |", "|---|" + "---|" * 9]
    tex = []
    for name in names:
        cells, rows = [], [res.get((name, s)) for s in SIZES]
        for r in rows: cells.append("---" if r is None else f"{r['acc_in']:.2f}/{r['acc_all9']:.2f}/{r['acc_lodo']:.2f}")
        ok = [r for r in rows if r]
        avg = [np.mean([r[k] for r in ok]) for k in ("acc_in", "acc_all9", "acc_lodo", "diff_mean")]; mx = max(r["diff_max"] for r in ok) if ok else np.nan
        md.append(f"| {name} | " + " | ".join(cells) + f" | {avg[0]:.3f} | {avg[1]:.3f} | {avg[2]:.3f} | {avg[3]:.3f} | {mx:.3f} |")
        tex.append(f"{name} & " + " & ".join(("---" if r is None else f"{r['acc_in']:.2f} & {r['acc_all9']:.2f} & {r['acc_lodo']:.2f}") for r in rows) + f" & {avg[0]:.2f} & {avg[1]:.2f} & {avg[2]:.2f} \\\\")
    allr = list(res.values())
    md.append(f"| **Method-Avg** | | | | | {np.mean([r['acc_in'] for r in allr]):.3f} | {np.mean([r['acc_all9'] for r in allr]):.3f} | {np.mean([r['acc_lodo'] for r in allr]):.3f} | {np.mean([r['diff_mean'] for r in allr]):.3f} | {max(r['diff_max'] for r in allr):.3f} |")
    caption = (f"Threshold-transfer check (not in the main paper): accuracy of unsupervised MIAs on the same {a.n}+{a.n}-per-domain subsamples as Table~\\ref{{table:subsample_pooled}} "
               f"({a.reps} draws), macro-averaged over the nine domains. \\textbf{{In}}: threshold tuned within the evaluated domain (2-fold); \\textbf{{All}}: one threshold tuned on the pooled "
               f"nine-domain data (domain-unknown setting); \\textbf{{LODO}}: threshold tuned on the pooled other eight domains and applied to the held-out domain. "
               f"Thresholds maximise accuracy on the tuning data. For training-free MIAs the All and LODO columns coincide up to sampling noise.")
    hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{s}}}}}" for s in SIZES) + " & \\multicolumn{3}{c}{\\textbf{Model-Avg}}"
    sub = " & ".join(["In & All & LODO"] * 5)
    latex = ("\\begin{table*}[t]\n\\centering\\small\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l" + "c" * 15 + "}\n\\toprule\nMethod & " + hdr + " \\\\\n"
             + " ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(5)) + "\n & " + sub + " \\\\\n\\midrule\n" + "\n".join(tex)
             + "\n\\bottomrule\n\\end{tabular}}\n\\caption{" + caption + "}\n\\label{table:threshold_transfer}\n\\end{table*}\n")
    out = ROOT / "tables/table:threshold_transfer.tex"; out.write_text(latex)
    print("\n".join(md)); print(f"\nTEX -> {out}\nJSON -> {ROOT/'analysis/logs/lodo_threshold_check.json'}")

if __name__ == "__main__":
    main()
