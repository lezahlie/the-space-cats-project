#!/bin/bash
# SLURM tuning job - submitted by submit_tuning.sh, not directly.
# Env vars required: MASK_RATIO (0.0 | 0.25 | 0.5 | 0.75), PERSON (first name)
# Before submitting: run 'sinfo' to verify --partition and 'sacctmgr show assoc user=$(whoami)' for --account

#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 # 5 requested, script uses 4, 1 for the OS
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00 # ice-gpu max; tune_model.py auto-recovers completed stages if resubmitted
# #SBATCH --account=<account>

set -euo pipefail

PROJECT_DIR="$HOME/the-space-cats-project"
DATASET="galaxiesml_small"
NUM_CORES=4
MASK_RATIO="${MASK_RATIO:-0.0}"
PERSON="${PERSON:-user}"
MASK_LABEL="${MASK_RATIO//./_}"

PREP_DIR="$PROJECT_DIR/data/preprocessed/mask_$MASK_LABEL/$DATASET"
OUTPUT_DIR="$PROJECT_DIR/experiments/tune_mae_small_${PERSON}_${MASK_RATIO}"
mkdir -p "$PROJECT_DIR/logs"
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=${CUDA_VISIBLE_DEVICES:-?} mask_ratio=$MASK_RATIO person=$PERSON"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate spacecats-cuda
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi

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
    --input-folder "$PREP_DIR" \
    --output-folder "$OUTPUT_DIR" \
    --gpu-memory-fraction 0.9 \
    --num-cores "$NUM_CORES" \
    --tune-optimizer-steps 750 \
    --validate-every-steps 75 \
    --tune-patience 5
echo "done | $OUTPUT_DIR"
