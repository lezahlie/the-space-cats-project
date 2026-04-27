#!/bin/bash
# SLURM smoke test — trains one epoch on the tiny dataset to verify GPU, conda, and data are all working.
# Usage: sbatch pace/test_job.slurm

#SBATCH --job-name=test_train
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 # 5 requested, script uses 4, 1 for the OS
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
# #SBATCH --account=<account>

set -euo pipefail

PROJECT_DIR="$HOME/the-space-cats-project"
DATASET="galaxiesml_tiny"
NUM_CORES=4

PREP_DIR="$PROJECT_DIR/data/preprocessed/mask_0_0/$DATASET"
OUTPUT_DIR="$PROJECT_DIR/experiments/test_train"
mkdir -p "$PROJECT_DIR/logs"
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=${CUDA_VISIBLE_DEVICES:-?}"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate spacecats-cuda
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi

if [ ! -d "$PREP_DIR" ]; then
    echo "preprocessing $DATASET ..."
    python "$PROJECT_DIR/src/preprocess_data.py" \
        --input-folder "$PROJECT_DIR/data/$DATASET" \
        --output-folder "$PROJECT_DIR/data/preprocessed/mask_0_0" \
        --num-cores "$NUM_CORES" \
        --mask-ratio 0.0 \
        --debug
fi
echo "starting test train | output=$OUTPUT_DIR"
python "$PROJECT_DIR/src/train_model.py" \
    --config-file "$PROJECT_DIR/configs/train_defaults.json" \
    --input-folder "$PREP_DIR" \
    --output-folder "$OUTPUT_DIR" \
    --gpu-memory-fraction 0.9 \
    --num-cores "$NUM_CORES" \
    --tune-optimizer-steps 1000 \
    --validate-every-steps 50 \
    --max-wallclock-hours 2 \
    --checkpoint-buffer-minutes 30
    --debug
echo "done | $OUTPUT_DIR"
