#!/bin/bash
#SBATCH --job-name=olmo_detect_run_all
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/run_all_%j.out
# =============================================================================
# Run ONE detection method across EVERY domain for a split, then score it.
#
#   sbatch run_all.slurm <method> [matched|shifted]
#
# <method> is a run_detection method name:
#   loss_zlib_lowercase  minkprob  dcpdd  recall  camia  pac
#   neighborhood_attack  dcq  guided_instruction  cdd  selfcrit
# (loss_zlib_lowercase yields Perplexity/Zlib/Lowercase; minkprob yields
#  Min-K%/Min-K%++.)  Default split: matched.
#
# Per-record scores are written under results/, matching the released layout,
# so afterwards:   python reproduce_auc.py --method <key>
# =============================================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load GCCcore/11.3.0 Python/3.11.3 2>/dev/null || true
# Activate your environment first, e.g.:  conda activate olmo-detect
cd "$(dirname "$0")"
mkdir -p logs

METHOD="${1:?usage: sbatch run_all.slurm <method> [matched|shifted]}"
SPLIT="${2:-matched}"

python -u run_all.py --method "$METHOD" --split "$SPLIT"
