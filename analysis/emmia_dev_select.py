#!/usr/bin/env python
"""EMMIA base-score selection on the DEV split (2026-09-28).

EMMIA reports four candidate base scores (loss, zlib, mink, minkpp). paper_tables.py currently
takes the MAX over the four AUCs computed ON THE EVALUATED (test) SET, which is test-set
selection and inflates the EMMIA row. The dev runs launched 2026-09-26 (tags <tag>_dev in
results_iclr/emmia_raw) let the init be chosen on dev, like every other tuned method.

Protocol copied from analysis/shifted_retune.py::mink_pick, which is the project's established
convention: ONE value per (method, size), selected on the dev split POOLED over all domains;
the shifted split reuses the MATCHED-selected value (the paper's protocol -- see the
2026-09-26 "cost of reusing matched hyperparameters" log block). The shifted-dev selection is
also reported, for reference only.

Reports, per split and size, the macro-aggregated EMMIA row (subset -> domain -> stage -> All,
exactly as paper_tables.build does) under the old max-on-test rule and under dev selection.

Usage: ../OLMo-Detect/.venv/bin/python analysis/emmia_dev_select.py
   -> analysis/logs/emmia_dev_select.json
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
import paper_tables as PT

INITS = ("loss", "zlib", "mink", "minkpp")


def load_tag(tag, size):
    """samples of one emmia tag, or None"""
    p = PT.EM / tag / f"emmia_{size}.json"
    if not p.exists():
        alt = PT.EM / (tag + "_sub300") / f"emmia_{size}.json"
        if not alt.exists():
            return None
        p = alt
    return json.loads(p.read_text()).get("samples", [])


def split_scores(samples, k):
    """-> (member scores, non-member scores) for init k"""
    con = [float(s["final_scores"][k]) for s in samples if int(s["label"]) == 1]
    unc = [float(s["final_scores"][k]) for s in samples if int(s["label"]) == 0]
    return con, unc


def pooled_dev(split, size):
    """every dev sample for this (split,size), pooled over all domains"""
    out = []
    for stg, dom, keys in PT.STRUCT:
        for key in keys:
            lv = PT.leaves(key, split, size)
            if not lv:
                continue
            for (cl, _), (ul, _) in lv:
                ss = load_tag(PT.emmia_tag(cl) + "_dev", size)
                if ss:
                    out += ss
    return out


def dev_pick(split, size):
    """-> (best init, {init: dev auc}) selected on the pooled dev split"""
    ss = pooled_dev(split, size)
    if not ss:
        return None, {}
    devauc = {}
    for k in INITS:
        con, unc = split_scores(ss, k)
        devauc[k] = PT.auc(con, unc)
    valid = {k: v for k, v in devauc.items() if v is not None}
    if not valid:
        return None, devauc
    return max(valid, key=valid.get), devauc


def subset_test(key, split, size, init):
    """test AUC of one subset under a FIXED init (init=None -> max over inits, the old rule)"""
    lv = PT.leaves(key, split, size)
    if lv is None:
        return None
    samples = []
    for (cl, _), (ul, _) in lv:
        ss = load_tag(PT.emmia_tag(cl), size)
        if ss is None:
            return None
        samples += ss
    if not samples:
        return None
    if len({int(s["label"]) for s in samples}) < 2:
        return None
    if init is None:
        vals = [PT.auc(*split_scores(samples, k)) for k in INITS]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    return PT.auc(*split_scores(samples, init))


def macro_row(split, size, init):
    """EMMIA row aggregated subset -> domain -> stage -> All, as paper_tables.build does"""
    stages = {}
    for st in ("pre", "mid", "post"):
        doms = []
        for stg, dom, keys in PT.STRUCT:
            if stg != st:
                continue
            vals = [v for v in (subset_test(k, split, size, init) for k in keys) if v is not None]
            if vals:
                doms.append(sum(vals) / len(vals))
        stages[st] = sum(doms) / len(doms) if doms else None
    ok = [v for v in stages.values() if v is not None]
    stages["All"] = sum(ok) / len(ok) if ok else None
    return stages


def dev_pick_subset(key, split, size):
    """select the init on the DEV samples of ONE subset (same granularity as the old max)"""
    lv = PT.leaves(key, split, size)
    if lv is None:
        return None
    ss = []
    for (cl, _), (ul, _) in lv:
        s = load_tag(PT.emmia_tag(cl) + "_dev", size)
        if s:
            ss += s
    if not ss or len({int(s["label"]) for s in ss}) < 2:
        return None
    devauc = {k: PT.auc(*split_scores(ss, k)) for k in INITS}
    valid = {k: v for k, v in devauc.items() if v is not None}
    return max(valid, key=valid.get) if valid else None


def macro_row_persubset(split, size):
    """EMMIA row where the init is chosen on each subset's OWN dev samples"""
    stages = {}
    for st in ("pre", "mid", "post"):
        doms = []
        for stg, dom, keys in PT.STRUCT:
            if stg != st:
                continue
            vals = []
            for k in keys:
                init = dev_pick_subset(k, split, size)
                v = subset_test(k, split, size, init) if init else None
                if v is not None:
                    vals.append(v)
            if vals:
                doms.append(sum(vals) / len(vals))
        stages[st] = sum(doms) / len(doms) if doms else None
    ok = [v for v in stages.values() if v is not None]
    stages["All"] = sum(ok) / len(ok) if ok else None
    return stages


def best_fixed_on_test(split, size):
    """oracle: the best SINGLE init for this (split,size), scored on test.
    Separates granularity inflation (max varies per subset) from selection inflation."""
    best, bk = None, None
    for k in INITS:
        r = macro_row(split, size, k)
        if r["All"] is not None and (best is None or r["All"] > best["All"]):
            best, bk = r, k
    return bk, best


def main():
    out = {"protocol": "one init per (size); selected on pooled MATCHED dev; shifted reuses it",
           "selection": {}, "rows": {}}
    for size in PT.SIZES:
        mk, mdev = dev_pick("matched", size)
        sk, sdev = dev_pick("shifted", size)
        out["selection"][size] = {"matched_dev_pick": mk, "matched_dev_auc": mdev,
                                  "shifted_dev_pick": sk, "shifted_dev_auc": sdev}
    for split in ("matched", "shifted"):
        for size in PT.SIZES:
            init = out["selection"][size]["matched_dev_pick"]   # paper protocol: matched-selected
            ok, orow = best_fixed_on_test(split, size)
            out["rows"].setdefault(split, {})[size] = {
                "init": init,
                "old_max_on_test": macro_row(split, size, None),
                "dev_selected": macro_row(split, size, init),
                "dev_selected_per_subset": macro_row_persubset(split, size),
                "best_fixed_on_test": orow, "best_fixed_init": ok,
            }

    (ROOT / "analysis/logs").mkdir(parents=True, exist_ok=True)
    dst = ROOT / "analysis/logs/emmia_dev_select.json"
    dst.write_text(json.dumps(out, indent=1))

    print("SELECTION (pooled dev, per size)")
    print(f"{'size':<6}{'matched pick':<14}{'dev AUCs (loss/zlib/mink/minkpp)':<42}{'shifted pick'}")
    for size in PT.SIZES:
        s = out["selection"][size]
        a = "/".join("--" if s["matched_dev_auc"].get(k) is None else f"{s['matched_dev_auc'][k]:.3f}" for k in INITS)
        print(f"{size:<6}{str(s['matched_dev_pick']):<14}{a:<42}{s['shifted_dev_pick']}")
    def _f(v):
        return "--" if v is None else f"{v:.3f}"
    for split in ("matched", "shifted"):
        print(f"\nEMMIA 'All' column -- {split}   (max=old rule; pooled/per-subset=dev-selected; oracle=best fixed init on test)")
        print(f"{'size':<6}{'max@test':<11}{'dev pooled':<13}{'dev/subset':<13}{'oracle fixed':<14}{'oracle init'}")
        for size in PT.SIZES:
            r = out["rows"][split][size]
            print(f"{size:<6}{_f(r['old_max_on_test']['All']):<11}{_f(r['dev_selected']['All']):<13}"
                  f"{_f(r['dev_selected_per_subset']['All']):<13}{_f(r['best_fixed_on_test']['All']):<14}{r['best_fixed_init']}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
