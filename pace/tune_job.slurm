#!/bin/bash
# SLURM tuning job - submitted by submit_tuning.sh, not directly.
# Env vars required: MASK_RATIO (0.0 | 0.25 | 0.5 | 0.75), PERSON (first name)
# Before submitting: run 'sinfo' to verify --partition and 'sacctmgr show assoc user=$(whoami)' for --account

#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   # 4 requested, script uses 3, 1 for the OS
#SBATCH --mem=32G           # always ask for more than you need
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00     # ice-gpu max; tune_model.py auto-recovers completed stages if resubmitted
# #SBATCH --account=<account>

set -euo pipefail

PROJECT_DIR="$HOME/the-space-cats-project"
DATASET="galaxiesml_small"
NUM_CORES=3
MASK_RATIO="${MASK_RATIO:-0.0}"
PERSON="${PERSON:-user}"
MASK_LABEL="${MASK_RATIO//./_}"

CONFIG_FILE="$PROJECT_DIR/configs/tune_default.json"
PREP_DIR="$PROJECT_DIR/data/preprocessed/mask_$MASK_LABEL/$DATASET"
OUTPUT_DIR="$PROJECT_DIR/experiments/tune_mae_small_${PERSON}_${MASK_RATIO}"
mkdir -p "$PROJECT_DIR/logs"
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=${CUDA_VISIBLE_DEVICES:-n} mask_ratio=$MASK_RATIO person=$PERSON"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null
eval "$(conda shell.bash hook)"

set +u
conda activate spacecats-cuda
set -u

#######################################################
echo "checking GPU availability"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ "${CUDA_VISIBLE_DEVICES:-}" = "NoDevFiles" ]; then
    echo "ERROR: no GPU assigned by SLURM"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: no NVIDIA GPU visible to this job"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    exit 1
fi

python - <<'PY'
import sys

try:
    import torch
except Exception as e:
    print(f"ERROR: could not import PyTorch: {e}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see a CUDA GPU", file=sys.stderr)
    sys.exit(1)

print(f"PyTorch CUDA OK: {torch.cuda.get_device_name(0)}")
PY

#######################################################

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"


if [ ! -d "$PREP_DIR" ]; then
    echo "preprocessing $DATASET mask_ratio=$MASK_RATIO ..."
    python "$PROJECT_DIR/src/preprocess_data.py" \
        --input-folder "$PROJECT_DIR/data/$DATASET" \
        --output-folder "$PROJECT_DIR/data/preprocessed/mask_$MASK_LABEL" \
        --num-cores "$NUM_CORES" \
        --mask-ratio "$MASK_RATIO"
fi

echo "starting tuning | person=$PERSON mask_ratio=$MASK_RATIO output=$OUTPUT_DIR"

python "$PROJECT_DIR/src/tune_model.py" \
    --config-file "$CONFIG_FILE" \
    --input-folder "$PREP_DIR" \
    --output-folder "$OUTPUT_DIR" \
    --gpu-memory-fraction 0.9 \
    --num-cores "$NUM_CORES" \
    --tune-optimizer-steps 750 \
    --validate-every-steps 50 \
    --tune-patience 3

echo "done | $OUTPUT_DIR"
