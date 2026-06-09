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
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B = "benchmark"
SIZES = ["1b", "7b", "13b", "32b"]
MODEL = {s: f"olmo_models/OLMo2-{s.upper()}-Instruct" for s in SIZES}
GEN_METHODS = {"neighborhood_attack", "dcq", "guided_instruction", "cdd", "selfcrit"}
DEV_METHODS = {"recall", "camia"}

def con(stage_dir, split, name):       # contaminated test file (split in path + suffix)
    return f"{B}/{stage_dir}/contaminated/{split}/test/contaminated_{name}_test_{split}.jsonl"
def unc(stage_dir, name):              # uncontaminated test file (shared across splits)
    return f"{B}/{stage_dir}/uncontaminated/test/uncontaminated_{name}_test.jsonl"
def unc_dev(stage_dir, name):
    return f"{B}/{stage_dir}/uncontaminated/dev/uncontaminated_{name}_dev.jsonl"

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
               ("pretraining/starcoder","pretraining_starcoder_assembly"),
               ("midtraining/stackexchange","midtraining_stackexchange_math"),
               ("midtraining/stackexchange","midtraining_stackexchange_stackoverflow")]
        for sd, nm in pre:
            jobs.append(dict(con=con(sd,split,nm), unc=unc(sd,nm), dev=unc_dev(sd,nm), models=ALL))

        if split == "matched":   # GSM8K and RLVR have no shifted variant
            jobs.append(dict(con=f"{B}/midtraining/gsm8k/contaminated/test/contaminated_midtraining_gsm8k_test.jsonl",
                             unc=f"{B}/midtraining/gsm8k/uncontaminated/test/uncontaminated_midtraining_math-mix_gsm8k_test.jsonl",
                             dev=f"{B}/midtraining/gsm8k/uncontaminated/dev/uncontaminated_midtraining_math-mix_gsm8k_dev.jsonl",
                             models=ALL))

        # SFT: model-pair files
        for pair, models in [("1b-32b",["1b","32b"]), ("7b-13b",["7b","13b"])]:
            for sub in ["aya","wildchat"]:
                nm=f"posttraining_sft_{pair}_{sub}"
                jobs.append(dict(
                    con=f"{B}/posttraining/sft/{pair}/contaminated/{split}/test/contaminated_{nm}_test_{split}.jsonl",
                    unc=f"{B}/posttraining/sft/{pair}/uncontaminated/test/uncontaminated_{nm}_test.jsonl",
                    dev=f"{B}/posttraining/sft/{pair}/uncontaminated/dev/uncontaminated_{nm}_dev.jsonl",
                    models=models))

    # RLVR: matched only. For selfcrit, all three subsets; for others too.
    if split == "matched":
        for sub in ["cot-gsm8k","cot-math","ifeval"]:
            nm=f"posttraining_rlvr_{sub}"
            jobs.append(dict(
                con=f"{B}/posttraining/rlvr/contaminated/test/contaminated_{nm}_test.jsonl",
                unc=f"{B}/posttraining/rlvr/uncontaminated/test/uncontaminated_{nm}_test.jsonl",
                dev=f"{B}/posttraining/rlvr/uncontaminated/dev/uncontaminated_{nm}_dev.jsonl",
                models=ALL))

    # DPO: per-size.
    for sz in SIZES:
        if is_self:   # combined file, scored by its own size
            jobs.append(dict(
                con=f"{B}/posttraining/dpo/contaminated/{split}/test/contaminated_posttraining_dpo_{sz}_test_{split}.jsonl",
                unc=f"{B}/posttraining/dpo/uncontaminated/test/uncontaminated_posttraining_dpo_{sz}_test.jsonl",
                dev=f"{B}/posttraining/dpo/uncontaminated/dev/uncontaminated_posttraining_dpo_{sz}_dev.jsonl",
                models=[sz]))
        else:         # chosen and rejected files, scored separately
            for crf in ["chosen","rejected"]:
                jobs.append(dict(
                    con=f"{B}/posttraining/dpo/contaminated/{split}/test/contaminated_posttraining_dpo_{sz}_{split}_{crf}_test.jsonl",
                    unc=f"{B}/posttraining/dpo/uncontaminated/test/uncontaminated_posttraining_dpo_{sz}_{crf}_test.jsonl",
                    dev=f"{B}/posttraining/dpo/uncontaminated/dev/uncontaminated_posttraining_dpo_{sz}_{crf}_dev.jsonl",
                    models=[sz]))
    return jobs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True,
                    help="run_detection method name: loss_zlib_lowercase, minkprob, dcpdd, "
                         "recall, camia, pac, neighborhood_attack, dcq, guided_instruction, cdd, selfcrit.")
    ap.add_argument("--split", choices=["matched","shifted"], default="matched")
    ap.add_argument("--models", default=None, help="Override model dirs (space-separated).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default="results_repro",
                    help="Output dir (default results_repro/; kept separate from the released results/).")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and check files; run nothing.")
    a = ap.parse_args()

    override = a.models.split() if a.models else None
    max_tokens = 4096 if a.method in GEN_METHODS else 1024
    jobs = build(a.split, a.method)

    missing = []
    cmds = []
    for j in jobs:
        models = override or [MODEL[s] for s in j["models"]]
        for mp in models:
            for data in (j["con"], j["unc"]):
                if not (ROOT/data).exists(): missing.append(data); continue
                cmd = [sys.executable, "-u", "run_detection.py", "--model", mp, "--data", data,
                       "--method", a.method, "--batch-size", str(a.batch_size),
                       "--max-tokens", str(max_tokens), "--output-dir", a.out, "--seed", "42"]
                if a.method in DEV_METHODS:
                    if not (ROOT/j["dev"]).exists(): missing.append(j["dev"])
                    else: cmd += ["--uncon-dev-path", j["dev"]]
                cmds.append(cmd)

    print(f"# method={a.method} split={a.split}  -> {len(cmds)} run_detection calls")
    if missing:
        print(f"# WARNING: {len(set(missing))} input files not found:")
        for m in sorted(set(missing)): print("#   MISSING", m)
    if a.dry_run:
        for c in cmds: print(" ".join(c))
        return
    if missing:
        sys.exit("Refusing to run: missing input files (see above).")
    for c in cmds:
        print("==>", c[c.index("--data")+1], "|", c[c.index("--model")+1].split('/')[-1])
        subprocess.run(c, cwd=ROOT, check=True)
    print(f"All runs complete. Now: python evaluate.py --method <key> --results-dir {a.out}")

if __name__ == "__main__":
    main()
