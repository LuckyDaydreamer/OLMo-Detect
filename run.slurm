#!/bin/bash
#SBATCH --job-name=contam_detect
#SBATCH --time=0-12:00:00
#SBATCH --mem=100G
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run_%j.out
# =============================================================================
# General reproduction script: run detection method(s) on ONE domain.
#
# A "domain" = a (contaminated, uncontaminated) pair of eval JSONL files.
# Every method is invoked through the SAME run_detection.py CLI, so this one
# script reproduces all results. Pass the data paths (and any options) as args.
#
# Per-record scores are written to:
#   <out>/<dataset_stem>/<model_name>/<method>.json
#
# ---------------------------------------------------------------------------
# USAGE
#   sbatch run.slurm --con <CON.jsonl> --uncon <UNCON.jsonl> [options]
#   bash   run.slurm --con <CON.jsonl> --uncon <UNCON.jsonl> [options]
#
# REQUIRED
#   --con   PATH        Contaminated eval JSONL for the domain.
#   --uncon PATH        Uncontaminated eval JSONL for the domain.
#
# OPTIONAL
#   --model-dir DIR     Dir holding the OLMo-2 checkpoints   (default: olmo_models)
#   --models "A B .."   Space-separated model dirs to run    (default: all 4 sizes)
#   --methods "a b .."  Space-separated methods to run       (default: all methods)
#   --con-dev PATH      Contaminated dev JSONL   (only used by calibrated methods)
#   --uncon-dev PATH    Uncontaminated dev JSONL (REQUIRED by: recall, camia)
#   --out DIR           Output dir for score files           (default: results)
#   --batch-size N      Inference batch size                 (default: 1)
#   --seed N            Global seed                          (default: 42)
#   --env PATH          virtualenv/conda activate script to source (optional)
#   -h | --help         Show this help and exit.
#
# EXAMPLE (pretraining / openwebmath)
#   sbatch run.slurm \
#     --con   benchmark/pretraining/openwebmath/contaminated/matched/test/contaminated_pretraining_openwebmath_test_matched.jsonl \
#     --uncon benchmark/pretraining/openwebmath/uncontaminated/test/uncontaminated_pretraining_openwebmath_test.jsonl
# =============================================================================

set -euo pipefail

# ------------------------------ defaults --------------------------------------
CON_PATH=""
UNCON_PATH=""
MODEL_DIR="olmo_models"
MODELS=""                       # if empty -> all 4 sizes under MODEL_DIR
METHODS=""                      # if empty -> ALL_METHODS below
CON_DEV_PATH=""
UNCON_DEV_PATH=""
OUTPUT_DIR="results"
BATCH_SIZE=1
SEED=42
ENV_ACTIVATE=""

ALL_METHODS="loss_zlib_lowercase minkprob dcpdd emmia recall camia pac neighborhood_attack dcq guided_instruction cdd selfcrit"

# Generation-based methods need a larger max-new-tokens budget; others ignore it.
GEN_METHODS="neighborhood_attack dcq guided_instruction cdd selfcrit"
# Methods that require the uncontaminated dev reference.
DEV_METHODS="recall camia"

usage() { sed -n '10,46p' "$0"; exit "${1:-0}"; }

# ------------------------------ arg parsing -----------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --con)        CON_PATH="$2"; shift 2 ;;
        --uncon)      UNCON_PATH="$2"; shift 2 ;;
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        --models)     MODELS="$2"; shift 2 ;;
        --methods)    METHODS="$2"; shift 2 ;;
        --con-dev)    CON_DEV_PATH="$2"; shift 2 ;;
        --uncon-dev)  UNCON_DEV_PATH="$2"; shift 2 ;;
        --out)        OUTPUT_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --env)        ENV_ACTIVATE="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$CON_PATH" || -z "$UNCON_PATH" ]]; then
    echo "ERROR: --con and --uncon are required." >&2
    usage 1
fi
for f in "$CON_PATH" "$UNCON_PATH"; do
    [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 1; }
done

# ------------------------------ environment -----------------------------------
if [[ -n "$ENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
fi
module load GCCcore/11.3.0 Python/3.11.3 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")"
mkdir -p "$OUTPUT_DIR" logs

# ------------------------------ resolve lists ---------------------------------
if [[ -z "$MODELS" ]]; then
    MODELS="$MODEL_DIR/OLMo2-1B-Instruct $MODEL_DIR/OLMo2-7B-Instruct $MODEL_DIR/OLMo2-13B-Instruct $MODEL_DIR/OLMo2-32B-Instruct"
fi
[[ -z "$METHODS" ]] && METHODS="$ALL_METHODS"

DATA_FILES=("$CON_PATH" "$UNCON_PATH")

echo "=================================================================="
echo " contaminated : $CON_PATH"
echo " uncontaminated: $UNCON_PATH"
echo " models        : $MODELS"
echo " methods       : $METHODS"
echo " output dir    : $OUTPUT_DIR  | batch_size=$BATCH_SIZE seed=$SEED"
echo "=================================================================="

# ------------------------------ run -------------------------------------------
for MODEL_PATH in $MODELS; do
    for METHOD in $METHODS; do

        MAXTOK=1024
        [[ " $GEN_METHODS " == *" $METHOD "* ]] && MAXTOK=4096

        DEV_ARGS=()
        if [[ " $DEV_METHODS " == *" $METHOD "* ]]; then
            if [[ -n "$UNCON_DEV_PATH" ]]; then
                DEV_ARGS=(--uncon-dev-path "$UNCON_DEV_PATH")
            else
                echo "[skip] $METHOD requires --uncon-dev but none given."
                continue
            fi
        fi

        for DATA_PATH in "${DATA_FILES[@]}"; do
            echo "==> model=$(basename "$MODEL_PATH")  method=$METHOD  data=$(basename "$DATA_PATH")"
            python -u run_detection.py \
                --model "$MODEL_PATH" \
                --data "$DATA_PATH" \
                --method "$METHOD" \
                --batch-size "$BATCH_SIZE" \
                --max-tokens "$MAXTOK" \
                --output-dir "$OUTPUT_DIR" \
                --seed "$SEED" \
                "${DEV_ARGS[@]}"
        done
    done
done

echo "Done. Per-record scores under $OUTPUT_DIR/<dataset>/<model>/<method>.json"
# Score afterward with AUC_TPR@5%FPR.py, passing the contaminated and
# uncontaminated result dirs, e.g.:
#   python "AUC_TPR@5%FPR.py" \
#     --con   "$OUTPUT_DIR/contaminated_pretraining_openwebmath_test_matched" \
#     --uncon "$OUTPUT_DIR/uncontaminated_pretraining_openwebmath_test"
