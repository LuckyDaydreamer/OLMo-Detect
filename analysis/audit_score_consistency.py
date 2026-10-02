"""Audit member-vs-non-member scoring consistency for every (method, split, subset, size) in results_latest.
Flags: (A) degenerate AUC (>= HI or <= LO), (B) constant / near-constant scores on either side, (C) scoring-config values that differ between the
member file and the non-member file (result-dict scalars such as alpha, n, composition, max_tokens, score_type, num_copies, plus manifest config
keys mentioning bit/quant/dtype/batch/max_tokens). Cells whose two files were produced with different settings are artefacts by construction.
Usage: ../OLMo-Detect/.venv/bin/python analysis/audit_score_consistency.py [--hi 0.97] [--lo 0.03] -> analysis/logs/audit_score_consistency.{md,json}
"""
import sys, json, argparse, ast, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as PT
ap = argparse.ArgumentParser(); ap.add_argument("--hi", type=float, default=0.97); ap.add_argument("--lo", type=float, default=0.03); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; RL, RI, SD, SIZES = PT.RL, PT.RI, PT.SD, PT.SIZES
METHODS = [(n, s, c) for n, s, c in PT.METHODS if s not in ("__emmia__", "selfcrit")] + [("Sup-CAMIA", "camia_lr", "camia_lr_score"), ("MIA-Tuner", "mia_tuner", "mia_tuner_score"), ("FSD", "fsd", "fsd_minkpp_score")]
SKIP = {"method", "dataset", "dataset_path", "dataset_data_type", "con_dev_path", "uncon_dev_path", "samples", "n_samples", "num_samples", "elapsed_seconds", "output_path", "auc", "note", "n_valid"}
def load(leaf, size, stem):
    p = RL / leaf / SD[size] / f"{stem}.json"
    if not p.exists(): return None
    return json.loads(p.read_text())
def cfg_of(d, leaf, size, stem):
    praw = RI / leaf / SD[size] / f"{stem}.json"   # config keys live only in the raw files (consolidation strips them)
    r = json.loads(praw.read_text()).get("result", {}) if praw.exists() else d.get("result", {}); c = {k: v for k, v in r.items() if k not in SKIP and isinstance(v, (int, float, str, bool)) and "path" not in k.lower() and "dataset" not in k.lower() and "file" not in k.lower() and not k.startswith("num_") and "records" not in k and k not in ("shots_used", "prefix_token_len", "n_calibration_samples", "n_direction_samples")}
    m = RI / leaf / SD[size] / f"run_manifest_{stem}.json"
    if m.exists():
        try:
            mc = json.loads(m.read_text()).get("config", {}); mc = ast.literal_eval(mc) if isinstance(mc, str) else mc
            for k, v in mc.items():
                if any(t in k.lower() for t in ("bit", "quant", "dtype", "fp16", "max_tokens")): c["manifest." + k] = str(v)
        except Exception: pass
    return c
flags = []; ncells = 0
for name, stem, col in METHODS:
    for split in ("matched", "shifted"):
        for stg, dom, keys in PT.STRUCT:
            for k in keys:
                for s in SIZES:
                    lv = PT.leaves(k, split, s)
                    if lv is None: continue
                    for (cl, cn), (ul, un) in lv:
                        dc, du = load(cl, s, stem), load(ul, s, stem)
                        if dc is None or du is None: continue
                        ncells += 1
                        vc = np.array([float(x[col]) for x in dc["result"]["samples"] if PT.valid(x.get(col))]); vu = np.array([float(x[col]) for x in du["result"]["samples"] if PT.valid(x.get(col))])
                        if len(vc) < 2 or len(vu) < 2: continue
                        auc = PT.auc(list(vc), list(vu)); issues = []
                        if auc >= a.hi or auc <= a.lo: issues.append(f"AUC={auc:.3f}")
                        for tag, v in (("members", vc), ("non-members", vu)):
                            if v.std() < 1e-6 or len(np.unique(np.round(v, 6))) <= 2: issues.append(f"{tag} constant (mean={v.mean():.4g}, distinct={len(np.unique(v))})")
                        cc, cu = cfg_of(dc, cl, s, stem), cfg_of(du, ul, s, stem)
                        diff = {kk: (cc.get(kk), cu.get(kk)) for kk in set(cc) | set(cu) if cc.get(kk) != cu.get(kk) and kk in cc and kk in cu}
                        if diff: issues.append("config member≠non-member: " + ", ".join(f"{kk}={v[0]}|{v[1]}" for kk, v in sorted(diff.items())))
                        if issues: flags.append(dict(method=name, split=split, subset=k, size=s, leaf=cl, auc=round(auc, 3), issues=issues))
(ROOT / "analysis/logs").mkdir(exist_ok=True, parents=True)
json.dump(flags, open(ROOT / "analysis/logs/audit_score_consistency.json", "w"), indent=1)
lines = [f"# Score-consistency audit ({ncells} member/non-member cells checked; {len(flags)} flagged)", ""]
from collections import Counter
cnt = Counter((f["method"], f["split"]) for f in flags)
lines += ["| method | split | #flagged cells |", "|---|---|---|"] + [f"| {m} | {sp} | {c} |" for (m, sp), c in sorted(cnt.items())] + [""]
for f in flags: lines.append(f"- {f['method']:10s} {f['split']:8s} {f['subset']:16s} {f['size']:4s} AUC={f['auc']:.3f} :: " + " ; ".join(f["issues"]))
(ROOT / "analysis/logs/audit_score_consistency.md").write_text("\n".join(lines))
print("\n".join(lines[:len(cnt) + 5]))
print(f"... full list: analysis/logs/audit_score_consistency.md ({len(flags)} flagged cells)")
