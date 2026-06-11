#!/bin/bash
#SBATCH --job-name=run_all
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/run_all_%x_%j.out
# Set your cluster's GPU partition / QOS, e.g.:
##SBATCH --partition=<gpu-partition>
##SBATCH --qos=<qos>
# =============================================================================
# Run ONE detection method across EVERY domain for a split, then score it.
#
#   sbatch --job-name=<method> run_all.slurm <method> [matched|shifted]
#
# Passing --job-name puts the method in the log filename and in squeue (it is
# optional — the method is also printed at the top of the log either way).
#
# <method> is a run_detection method name:
#   loss_zlib_lowercase  minkprob  dcpdd  recall  camia  pac
#   neighborhood_attack  dcq  guided_instruction  cdd  selfcrit
# (loss_zlib_lowercase yields Perplexity/Zlib/Lowercase; minkprob yields
#  Min-K%/Min-K%++.)  Default split: matched.
#
# Per-record scores are written under results/, matching the released layout,
# so afterwards:   python evaluate.py --method <key>
# =============================================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run from the directory you submitted from. sbatch copies this script to a
# spool dir, so "$0" is NOT the repo — use SLURM_SUBMIT_DIR (falls back to the
# script dir when run as a plain shell script).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# Activate the .venv from Setup (skipped if you manage the environment yourself).
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

METHOD="${1:?usage: sbatch --job-name=<method> run_all.slurm <method> [matched|shifted]}"
SPLIT="${2:-matched}"

echo "=================================================="
echo " run_all  |  method: ${METHOD}  |  split: ${SPLIT}"
echo " repo: $(pwd)  |  job: ${SLURM_JOB_ID:-NA}"
echo "=================================================="

python -u run_all.py --method "$METHOD" --split "$SPLIT"
