#!/usr/bin/env python
"""EMMIA generation fleet.

EMMIA is a separate pipeline (scores member+non-member JOINTLY per
(domain,size); see evaluate_emmia.py). It is NOT wired into run_all, so this
driver reuses run_all.build() to enumerate EXACTLY the same test leaves and runs
methods/emmia_method.py on each (leaf, size), writing a per-leaf
results_iclr/emmia_raw/<TAG>/emmia_<size>.json (+ a persistent .npz cache).

One process per size (env/flag), both splits, so per-size SLURM jobs parallelize.
Idempotent: a leaf whose output already has non-empty "samples" is skipped.

  python run_emmia_fleet.py --sizes 1b            # both splits, size 1b
  python run_emmia_fleet.py --sizes 7b --split matched --dry-run
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import run_all  # noqa: E402  (build(), SIZES, MODEL, B)

PY = sys.executable
OUT_ROOT = ROOT / "results_iclr" / "emmia_raw"
# EMMIA hyperparameters — identical to the validated smoke run.
EMMIA_ARGS = ["--n-iterations", "10", "--mink-k", "20", "--minkpp-k", "20"]   # --max-tokens/--prefix-batch-size now CLI (22 Sep: RLVR needs 1024)


def tag_of(con_path: str, split: str) -> str:
    """Unique per-(leaf,split) slug; matches the smoke TAG scheme.
    e.g. member_pretraining_starcoder_python_test_matched.jsonl
         -> pretraining_starcoder_python_matched"""
    b = os.path.basename(con_path)
    if b.startswith("member_"):
        b = b[len("member_"):]
    if b.endswith(".jsonl"):
        b = b[: -len(".jsonl")]
    b = b.replace("_test_", "_")          # drop the _test marker (split-bearing names)
    if b.endswith("_test"):
        b = b[: -len("_test")]            # ... and split-less names (gsm8k/rlvr)
    if split not in b:                     # keep matched/shifted distinct on disk
        b = f"{b}_{split}"
    return b


def already_done(out: Path) -> bool:
    if not out.exists():
        return False
    try:
        d = json.loads(out.read_text())
        return bool(d.get("samples"))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["matched", "shifted", "both"], default="both")
    ap.add_argument("--sizes", default=None, help="e.g. '1b' or '1b 7b'; default all four")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", default=None, help="only leaves whose tag contains this substring (e.g. starcoder)")
    ap.add_argument("--max-tokens", type=int, default=512, help="EMMIA window (RLVR texts carry a 671-token shared few-shot preamble -> use 1024)")
    ap.add_argument("--prefix-batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-run even if the output already has samples")
    ap.add_argument("--dev", action="store_true",
                    help="score the DEV leaves instead of test, writing tags with a _dev suffix "
                         "(2026-09-26: needed to select EMMIA's base score on dev rather than on the evaluated set)")
    a = ap.parse_args()

    allowed = set(a.sizes.replace(",", " ").split()) if a.sizes else set(run_all.SIZES)
    splits = ["matched", "shifted"] if a.split == "both" else [a.split]

    plan = []  # (split, size, tag, con, unc)
    for split in splits:
        for j in run_all.build(split, "emmia"):
            for sz in [s for s in j["models"] if s in allowed]:
                tag = tag_of(j["con"], split)
                con, unc = j["con"], j["unc"]
                if a.dev:
                    con, unc, tag = run_all.to_dev(con), run_all.to_dev(unc), tag + "_dev"
                plan.append((split, sz, tag, con, unc))

    if a.keep:
        plan = [p for p in plan if a.keep in p[2]]

    n_skip = sum(1 for (_, sz, tag, _, _) in plan
                 if already_done(OUT_ROOT / tag / f"emmia_{sz}.json"))
    print(f"# EMMIA fleet: {len(plan)} (leaf,size) units across splits={splits} sizes={sorted(allowed)} "
          f"({n_skip} already done)")
    if a.dry_run:
        for split, sz, tag, con, unc in plan:
            done = "DONE" if already_done(OUT_ROOT / tag / f"emmia_{sz}.json") else "todo"
            print(f"  [{done}] {sz:>3} {split:7} {tag}")
            if not (ROOT / con).exists():
                print(f"      !! MISSING con: {con}")
            if not (ROOT / unc).exists():
                print(f"      !! MISSING unc: {unc}")
        return

    ok = fail = skip = 0
    for i, (split, sz, tag, con, unc) in enumerate(plan, 1):
        out = OUT_ROOT / tag / f"emmia_{sz}.json"
        cache = OUT_ROOT / "cache" / tag
        if already_done(out) and not a.force:
            skip += 1
            continue
        if not (ROOT / con).exists() or not (ROOT / unc).exists():
            print(f"[{i}/{len(plan)}] MISSING data, skip: {tag} {sz}", flush=True)
            fail += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", "methods/emmia_method.py",
               "--model", run_all.MODEL[sz],
               "--con-eval-path", str(ROOT / con),
               "--uncon-eval-path", str(ROOT / unc),
               "--output", str(out),
               "--cache-dir", str(cache), "--max-tokens", str(a.max_tokens), "--prefix-batch-size", str(a.prefix_batch_size), *EMMIA_ARGS]
        t0 = time.time()
        print(f"[{i}/{len(plan)}] RUN {sz} {split} {tag}", flush=True)
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        if rc == 0 and already_done(out):
            ok += 1
            print(f"[{i}/{len(plan)}] OK {tag} {sz} ({dt/60:.1f} min)", flush=True)
        else:
            fail += 1
            print(f"[{i}/{len(plan)}] FAILED (rc={rc}) {tag} {sz}", flush=True)
    print(f"# EMMIA fleet done: ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
