#!/usr/bin/env python
"""
Run ONE detection method across EVERY domain for a given split (matched/shifted),
calling run_detection.py per (data file, model size) with the right pairing.

Handled automatically (mirrors how the experiments were run):
  * SFT  — the 1b-32b file is scored only by 1B/32B, the 7b-13b file only by 7B/13B.
  * DPO  — per-size file scored only by its own size; for likelihood/most methods
           the chosen and rejected files are scored separately (they become the
           _chosen / _rejected result dirs); Self-Critique uses the combined file.
  * GSM8K and RLVR exist only for the matched split (no shifted variant) and are
           skipped when --split shifted.
  * Self-Critique only applies to DPO and RLVR; other domains are skipped for it.
  * recall / camia get --uncon-dev (their reference split) automatically.

Scores go to a SEPARATE dir (default results_repro/) so the released results/
is never overwritten. Dir names inside it match the released layout, so:
  python evaluate.py --method <key> --results-dir results_repro

Usage:
  python run_all.py --method minkprob --split matched
  python run_all.py --method camia   --split matched --models "olmo_models/OLMo2-1B-Instruct ..."
  python run_all.py --method loss_zlib_lowercase --split matched --dry-run
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import config
import argparse, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Benchmark root. Defaults to the reconstructed ICLR2027 benchmark; override with
# OLMO_DETECT_BENCH=/path/to/benchmark (e.g. point back at the old ./benchmark).
B = str(config.BENCHMARK)
SIZES = ["1b", "7b", "13b", "32b"]
MODEL = {s: str(config.MODELS / f"OLMo2-{s.upper()}-Instruct") for s in SIZES}
GEN_METHODS = {"neighborhood_attack", "dcq", "guided_instruction", "cdd", "selfcrit",
               "pithc_ctx"}  # pithc_ctx isn't generative but needs 4096-token context
DEV_METHODS = {"recall", "camia", "camia_faithful", "camia_lr", "fsd"}

def con(stage_dir, split, name):       # member test file (split in path + suffix)
    return f"{B}/{stage_dir}/member/{split}/test/member_{name}_{split}_test.jsonl"
def unc(stage_dir, name):              # non-member test file (shared across splits)
    return f"{B}/{stage_dir}/non-member/test/non-member_{name}_test.jsonl"
def unc_dev(stage_dir, name):
    return f"{B}/{stage_dir}/non-member/dev/non-member_{name}_dev.jsonl"

# --- fine-tuning-method helpers (opt_gap / mia_tuner / fsd) ------------------
# Continue-training methods score the same con/unc TEST files but differ in the
# side inputs they need: opt_gap -> --steps only; fsd -> --uncon-dev; mia_tuner
# -> BOTH --con-dev and --uncon-dev (supervised soft prompt).  A single test->dev
# transform derives every dev path from the matching test path (verified against
# the on-disk layout for pretraining/mid/gsm8k/dpo/sft/rlvr).
def to_dev(p: str) -> str:
    p = p.replace("/test/", "/dev/")
    if p.endswith("_test.jsonl"):
        p = p[: -len("_test.jsonl")] + "_dev.jsonl"
    return p

def _safe_filename(value: str) -> str:  # mirrors detector._safe_filename
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_") or "unknown"

def _dataset_name(path: str) -> str:    # mirrors data_utils._dataset_name_from_path
    stem = Path(path).stem
    if stem.endswith("_eval"):
        stem = stem[: -len("_eval")]
    if stem.endswith("_test"):
        stem = stem[: -len("_test")]
    return stem

def _already_done(out: str, data: str, model: str, method: str) -> bool:
    """True if THIS (method, data, model) already has a scored result on disk, so
    gap-fill re-runs skip the expensive existing trajectories.  Method-aware: only
    the specific method's json counts, so two methods can share --out safely."""
    d = ROOT / out / _dataset_name(data) / _safe_filename(Path(model).name)
    return (d / f"{_safe_filename(method)}.json").is_file()

def build(split, method):
    """Return list of jobs: dict(con, unc, dev, models)."""
    jobs = []
    ALL = SIZES
    is_self = (method == "selfcrit")

    if not is_self:
        # pre-training + mid-training (StackExchange): all four sizes
        pre = [("pretraining/dclm","pretraining_dclm"),
               ("pretraining/pes2o","pretraining_pes2o"),
               ("pretraining/openwebmath","pretraining_openwebmath"),
               ("pretraining/starcoder","pretraining_starcoder_java"),
               ("pretraining/starcoder","pretraining_starcoder_python"),
               ("midtraining/stackexchange","midtraining_stackexchange_math"),
               ("midtraining/stackexchange","midtraining_stackexchange_stackoverflow")]
        for sd, nm in pre:
            jobs.append(dict(con=con(sd,split,nm), unc=unc(sd,nm), dev=unc_dev(sd,nm), models=ALL))

        if split == "matched":   # GSM8K and RLVR have no shifted variant
            jobs.append(dict(con=f"{B}/midtraining/gsm8k/member/test/member_midtraining_gsm8k_test.jsonl",
                             unc=f"{B}/midtraining/gsm8k/non-member/test/non-member_midtraining_gsm8k_test.jsonl",
                             dev=f"{B}/midtraining/gsm8k/non-member/dev/non-member_midtraining_gsm8k_dev.jsonl",
                             models=ALL))

        # SFT: model-pair files
        for pair, models in [("1b-32b",["1b","32b"]), ("7b-13b",["7b","13b"])]:
            for sub in ["aya","wildchat"]:
                nm=f"posttraining_sft_{pair}_{sub}"
                jobs.append(dict(
                    con=f"{B}/posttraining/sft/{pair}/member/{split}/test/member_{nm}_{split}_test.jsonl",
                    unc=f"{B}/posttraining/sft/{pair}/non-member/test/non-member_{nm}_test.jsonl",
                    dev=f"{B}/posttraining/sft/{pair}/non-member/dev/non-member_{nm}_dev.jsonl",
                    models=models))

    # RLVR: matched only. For selfcrit, all three subsets; for others too.
    if split == "matched":
        for sub in ["cot-gsm8k","gsm8k_no_cot","cot-math","math_no_cot"]:   # cot-math replaces ifeval (dropped); *_no_cot = no-CoT ablation (math_no_cot added 22 Sep)
            nm=f"posttraining_rlvr_{sub}"
            jobs.append(dict(
                con=f"{B}/posttraining/rlvr/member/test/member_{nm}_test.jsonl",
                unc=f"{B}/posttraining/rlvr/non-member/test/non-member_{nm}_test.jsonl",
                dev=f"{B}/posttraining/rlvr/non-member/dev/non-member_{nm}_dev.jsonl",
                models=ALL))

    # DPO: per-size.
    for sz in SIZES:
        if is_self:   # combined file, scored by its own size
            jobs.append(dict(
                con=f"{B}/posttraining/dpo/member/{split}/test/member_posttraining_dpo_{sz}_pair_{split}_test.jsonl",
                unc=f"{B}/posttraining/dpo/non-member/test/non-member_posttraining_dpo_{sz}_pair_test.jsonl",
                dev=f"{B}/posttraining/dpo/non-member/dev/non-member_posttraining_dpo_{sz}_pair_dev.jsonl",
                models=[sz]))
        else:         # chosen and rejected files, scored separately
            for crf in ["chosen","rejected"]:
                jobs.append(dict(
                    con=f"{B}/posttraining/dpo/member/{split}/test/member_posttraining_dpo_{sz}_{crf}_{split}_test.jsonl",
                    unc=f"{B}/posttraining/dpo/non-member/test/non-member_posttraining_dpo_{sz}_{crf}_test.jsonl",
                    dev=f"{B}/posttraining/dpo/non-member/dev/non-member_posttraining_dpo_{sz}_{crf}_dev.jsonl",
                    models=[sz]))
    return jobs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True,
                    help="run_detection method name: loss_zlib_lowercase, minkprob, dcpdd, "
                         "recall, camia, pac, neighborhood_attack, dcq, guided_instruction, cdd, selfcrit.")
    ap.add_argument("--split", choices=["matched","shifted"], default="matched")
    ap.add_argument("--sizes", default=None,
                    help="Restrict to these sizes (e.g. '1b 7b' or '1b,7b'), preserving the "
                         "size-specific SFT/DPO pairing. Default: all four sizes.")
    ap.add_argument("--models", default=None, help="Override model dirs (space-separated).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Override max input tokens (default: 4096 for GEN_METHODS else 1024). "
                         "Set 4096 for gapk/tagtab/detectgpt to match the OLMo-Detect protocol.")
    ap.add_argument("--steps", type=int, default=None,
                    help="Gradient steps for opt_gap (Direct Optimization-Gap / Plateau Rate). "
                         "Required when --method opt_gap; ignored otherwise.")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="Fine-tuning LR override, forwarded to opt_gap only.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip any (data,model) that already has a scored result in --out "
                         "(idempotent gap-fill; e.g. reuse the K=100 DCLM/GSM8K/DPO trajectories).")
    ap.add_argument("--exclude", default=None,
                    help="Space/comma-separated substrings; drop any data file whose path "
                         "contains one (e.g. 'dclm gsm8k' to leave those to another job). "
                         "Case-insensitive. Avoids racing a concurrent job on shared dirs.")
    ap.add_argument("--out", default="results_repro",
                    help="Output dir (default results_repro/; kept separate from the released results/).")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and check files; run nothing.")
    a = ap.parse_args()

    override = a.models.split() if a.models else None
    allowed = set(a.sizes.replace(",", " ").split()) if a.sizes else set(SIZES)
    bad = allowed - set(SIZES)
    if bad:
        sys.exit(f"--sizes has unknown size(s): {sorted(bad)}; valid: {SIZES}")
    max_tokens = a.max_tokens if a.max_tokens is not None else (4096 if a.method in GEN_METHODS else 1024)
    if a.method in ("opt_gap", "opt_gap_control", "opt_gap_transfer", "opt_gap_did") and a.steps is None:
        sys.exit(f"--method {a.method} requires --steps (e.g. --steps 100).")
    exclude = [t.lower() for t in a.exclude.replace(",", " ").split()] if a.exclude else []
    jobs = build(a.split, a.method)

    missing = []
    cmds = []
    skipped = 0
    for j in jobs:
        job_sizes = [s for s in j["models"] if s in allowed]  # preserve per-job pairing
        if not job_sizes:
            continue
        models = override or [MODEL[s] for s in job_sizes]
        cdev, udev = to_dev(j["con"]), j["dev"]  # domain-level dev paths (shared by con/unc test)
        for mp in models:
            for data in (j["con"], j["unc"]):
                if exclude and any(x in data.lower() for x in exclude):
                    continue
                if not (ROOT/data).exists(): missing.append(data); continue
                if a.skip_existing and _already_done(a.out, data, mp, a.method):
                    skipped += 1; continue
                cmd = [sys.executable, "-u", "run_detection.py", "--model", mp, "--data", data,
                       "--method", a.method, "--batch-size", str(a.batch_size),
                       "--max-tokens", str(max_tokens), "--output-dir", a.out, "--seed", "42"]
                if a.method in ("opt_gap", "opt_gap_control", "opt_gap_transfer", "opt_gap_did"):
                    cmd += ["--steps", str(a.steps)]
                    if a.learning_rate is not None:
                        cmd += ["--learning-rate", str(a.learning_rate)]
                if a.method in DEV_METHODS:               # recall / camia / fsd -> uncon-dev
                    if not (ROOT/udev).exists(): missing.append(udev)
                    else: cmd += ["--uncon-dev-path", udev]
                if a.method == "mia_tuner":               # supervised -> BOTH dev splits
                    for dv in (cdev, udev):
                        if not (ROOT/dv).exists(): missing.append(dv)
                    if (ROOT/cdev).exists() and (ROOT/udev).exists():
                        cmd += ["--con-dev-path", cdev, "--uncon-dev-path", udev]
                cmds.append(cmd)

    print(f"# method={a.method} split={a.split}  -> {len(cmds)} run_detection calls"
          f"  ({skipped} skipped as already-done)" if a.skip_existing else
          f"# method={a.method} split={a.split}  -> {len(cmds)} run_detection calls")
    if missing:
        print(f"# WARNING: {len(set(missing))} input files not found:")
        for m in sorted(set(missing)): print("#   MISSING", m)
    if a.dry_run:
        for c in cmds: print(" ".join(c))
        return
    if missing:
        sys.exit("Refusing to run: missing input files (see above).")
    failures = []
    for c in cmds:
        data = c[c.index("--data")+1]
        print("==>", data, "|", c[c.index("--model")+1].split('/')[-1])
        try:
            subprocess.run(c, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as e:  # one bad leaf must not abort the rest
            print(f"# FAILED (rc={e.returncode}): {data}")
            failures.append(data)
    if failures:
        print(f"# {len(failures)} leaf(s) FAILED (others completed):")
        for d in failures: print("#   FAILED", d)
    print(f"All runs complete. Now: python evaluate.py --method <key> --results-dir {a.out}")
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
