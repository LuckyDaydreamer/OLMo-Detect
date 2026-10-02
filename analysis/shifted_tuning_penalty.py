#!/usr/bin/env python
"""How much does reusing OLMo-Detect hyperparameters cost on OLMo-Detect (Shifted)? (2026-09-26)

Shifted results reuse the matched-setting hyperparameters, because the two settings share their
non-members, so selecting new values would change the non-member scores and the matched-to-shifted
difference would no longer isolate the shift. A reviewer may object that the shifted AUCs are
therefore suboptimal. The shifted-dev sweeps in OLMo-Detect/iclr_tune record dev AUC at EVERY
candidate value, so the cost is directly measurable:

    penalty = devAUC_shifted(best value) - devAUC_shifted(matched-selected value)

Usage: python analysis/shifted_tuning_penalty.py  ->  analysis/logs/shifted_tuning_penalty.json
"""
import json, re, sys
from pathlib import Path

TUNE = Path("../OLMo-Detect/iclr_tune")
METH = Path("../OLMo-Detect/methods")
SIZES = ["1b", "7b", "13b", "32b"]
# tune-file stem -> (method source file, display name)
FAMS = {"cdd": ("cdd_method.py", "CDD"), "dcq": ("dcq_method.py", "DCQ"),
        "guided": ("guided_instruction_method.py", "Guided"),
        "neighborhood": ("neighborhood_attack_method.py", "Neigh.")}

def matched_cfg(src, size):
    """Read MATCHED_CONFIG entry for this model size from the method source (or its pre-revert backup)."""
    for cand in (METH / (src + ".bak_20260924_shiftcfg"), METH / src):
        if not cand.exists(): continue
        s = cand.read_text()
        m = re.search(r"MATCHED_CONFIG[^=]*=\s*\{(.*?)\n\}", s, re.S)
        if not m: continue
        for line in m.group(1).splitlines():
            if f"OLMo2-{size.upper()}-Instruct" in line.replace("olmo2", "OLMo2"):
                d = re.search(r"\{(.*)\}", line)
                if d: return eval("{" + d.group(1) + "}")
    return None

def match_entry(all_, cfg):
    """find the sweep entry whose candidate fields equal cfg (ignoring absent keys)"""
    for e in all_:
        keys = [k for k in e if k != "auc"]
        if all(k in cfg and abs(float(e[k]) - float(cfg[k])) < 1e-9
               if isinstance(e[k], (int, float)) else cfg.get(k) == e[k] for k in keys):
            return e
    return None

out, rows = {}, []
for fam, (src, name) in FAMS.items():
    for size in SIZES:
        p = TUNE / f"{fam}_tune_shifted_{size}.json"
        if not p.exists(): continue
        d = json.load(open(p)); all_ = d.get("all") or []
        best = d.get("best") or {}
        cfg = matched_cfg(src, size)
        if cfg is None: print(f"!! {name} {size}: no MATCHED_CONFIG"); continue
        e = match_entry(all_, cfg)
        if e is None:
            print(f"!! {name} {size}: matched value {cfg} not in sweep grid"); continue
        pen = float(best["auc"]) - float(e["auc"])
        aucs = sorted(float(x["auc"]) for x in all_)
        rows.append((name, size, float(e["auc"]), float(best["auc"]), pen, aucs[0], aucs[-1], len(all_)))
        out[f"{name}|{size}"] = dict(matched_value=cfg, matched_dev_auc=float(e["auc"]),
                                     best_value={k: v for k, v in best.items() if k != "auc"},
                                     best_dev_auc=float(best["auc"]), penalty=pen,
                                     grid_min=aucs[0], grid_max=aucs[-1], n_candidates=len(all_))

print(f"{'method':8s} {'size':5s} {'dev AUC @matched':>16s} {'@shifted-best':>14s} {'penalty':>8s} {'grid range':>16s}")
for name, size, a, b, pen, lo, hi, n in rows:
    print(f"{name:8s} {size:5s} {a:16.3f} {b:14.3f} {pen:+8.3f}   [{lo:.3f},{hi:.3f}] n={n}")
if rows:
    pens = [r[4] for r in rows]
    print(f"\npenalty: mean {sum(pens)/len(pens):+.3f}  max {max(pens):+.3f}  "
          f"cells with penalty > 0.02: {sum(p > 0.02 for p in pens)}/{len(pens)}")
json.dump(out, open("analysis/logs/shifted_tuning_penalty.json", "w"), indent=1)
print("-> analysis/logs/shifted_tuning_penalty.json")
