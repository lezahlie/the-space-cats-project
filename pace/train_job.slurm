#!/bin/bash
# SLURM tuning job - submitted by submit_tuning.sh, not directly.
# Env vars required: MASK_RATIO (0.0 | 0.25 | 0.5 | 0.75), PERSON (first name)
# Before submitting: run 'sinfo' to verify --partition and 'sacctmgr show assoc user=$(whoami)' for --account

#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6   # 6 requested, script uses 5, 1 for the OS
#SBATCH --mem=64G           # always ask for more than you need
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00     # ice-gpu max; tune_model.py auto-recovers completed stages if resubmitted
# #SBATCH --account=<account>

set -euo pipefail

PROJECT_DIR="$HOME/the-space-cats-project"
DATASET="galaxiesml_medium"
NUM_CORES=5
MASK_RATIO="${MASK_RATIO:-0.0}"
PERSON="${PERSON:-user}"
MASK_LABEL="${MASK_RATIO//./_}"

CONFIG_FILE="$PROJECT_DIR/configs/train_best_${PERSON}_mask_${MASK_LABEL}.json"
PREP_BASE_DIR="$PROJECT_DIR/data/preprocessed/mask_$MASK_LABEL"
PREP_DIR="$PREP_BASE_DIR/$DATASET"
OUTPUT_DIR="$PROJECT_DIR/experiments/train_mae_medium_${PERSON}_mask_${MASK_LABEL}"

mkdir -p "$PROJECT_DIR/logs"
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=${CUDA_VISIBLE_DEVICES:-?} mask_ratio=$MASK_RATIO person=$PERSON"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate spacecats-cuda
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi

if [ ! -d "$PREP_DIR" ]; then
    echo "preprocessing $DATASET mask_ratio=$MASK_RATIO ..."
    srun -n 1 -c 6 python "$PROJECT_DIR/src/preprocess_data.py" \
        --input-folder "$PROJECT_DIR/data/$DATASET" \
        --output-folder "$PROJECT_DIR/data/preprocessed/mask_$MASK_LABEL" \
        --num-cores "$NUM_CORES" \
        --mask-ratio "$MASK_RATIO"
fi


if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: missing training config: $CONFIG_FILE"
    echo "Run: bash pace/submit_tuning.sh $PERSON"
    exit 1
fi

echo "starting tuning | person=$PERSON mask_ratio=$MASK_RATIO output=$OUTPUT_DIR"

srun -n 1 -c 6 python "$PROJECT_DIR/src/train_model.py" \
    --config-file "$CONFIG_FILE" \
    --input-folder "$PREP_DIR" \
    --output-folder "$OUTPUT_DIR" \
    --gpu-memory-fraction 0.9 \
    --num-cores "$NUM_CORES" \
    --validate-every-steps 100 \
    --max-wallclock-hours 16 \
    --checkpoint-buffer-minutes 120

echo "done | $OUTPUT_DIR"
