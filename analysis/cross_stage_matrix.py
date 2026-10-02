"""Rebuild analysis/logs/cross_stage_matrix.json (18 MIAs x 4 sizes x 13 subsets, matched split) from CURRENT results via paper_tables
(the 23-Sep matrix was a one-off). Unsupervised: paper_tables.subset_auc; Sup-CAMIA / MIA-Tuner: in-domain sup_subset_auc; FSD: base chosen
by FSD_BASE env (default 'loss' = FSD (PPL), the base that reproduces the 23-Sep matrix). Backs up the old matrix. Then run cross_stage_table.py."""
import json, os, sys, shutil, time
sys.path.insert(0, "utils"); import paper_tables as P
OLD = "analysis/logs/cross_stage_matrix.json"; M = json.load(open(OLD)); names, KEYS, STRUCT = M["names"], M["keys"], M["struct"]; SIZES = ["1b", "7b", "13b", "32b"]
unsup = {n: (s, c) for lst in vars(P).values() if isinstance(lst, list) for t in lst if isinstance(t, tuple) and len(t) == 3 and isinstance(t[0], str) for n, s, c in [t] if n in names}
SUP = {"Sup-CAMIA": ("latest", "camia_lr", "camia_lr_score"), "MIA-Tuner": ("latest", "mia_tuner", "mia_tuner_score"), "FSD": ("latest", "fsd", f"fsd_{os.environ.get('FSD_BASE','loss')}_score")}
auc = {}
for n in names:
    for s in SIZES:
        for k in KEYS:
            if n == "SelfCrit" and k not in ("dpo", "rlvr_cot-gsm8k", "rlvr_cot-math"): continue
            if n in SUP: a, ok = P.sup_subset_auc(SUP[n], k, "matched", s)
            else:
                st, col = unsup[n]; a, ok, note = P.subset_auc(n, st, col, k, "matched", s)
            if a is not None: auc[f"{n}|{s}|{k}"] = float(a)
if os.environ.get("CHECK"):
    old = M["auc"]; diffs = sorted(((abs(v - old[k]), k, round(old[k], 3), round(v, 3)) for k, v in auc.items() if k in old), reverse=True)
    print("cells new/old:", len(auc), len(old), "| missing in new:", [k for k in old if k not in auc][:8]); print("largest changes:"); [print("  ", d) for d in diffs[:15]]
    import collections; c = collections.Counter(k.split("|")[0] for d, k, *_ in diffs if d > 0.005); print("cells changed >0.005 by method:", dict(c))
else:
    shutil.copy(OLD, OLD + ".bak_" + time.strftime("%Y%m%d_%H%M")); json.dump(dict(names=names, keys=KEYS, struct=STRUCT, auc=auc), open(OLD, "w"), indent=1); print("WROTE", OLD, len(auc), "cells")
