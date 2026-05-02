#!/bin/bash

#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=16:00:00

set -eo pipefail

PROJECT_DIR="/storage/ice-shared/cs7643/shared-group-project-data/the-space-cats/the-space-cats-project"
NUM_CORES=4

PERSON="${PERSON:?PERSON is required}"
MASK_RATIO="${MASK_RATIO:?MASK_RATIO is required}"

RUN_NAME="train_mae_medium_${PERSON}_mask_${MASK_RATIO}"
INPUT_DIR="${PROJECT_DIR}/experiments/${RUN_NAME}/artifacts/samples"
OUTPUT_DIR="${PROJECT_DIR}/experiments/knn_results/${RUN_NAME}"
PARAMS_FILE="${PROJECT_DIR}/configs/knn_best_params.yaml"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME gpus=${CUDA_VISIBLE_DEVICES:-n} person=$PERSON mask_ratio=$MASK_RATIO"
echo "input=$INPUT_DIR"
echo "output=$OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: missing MAE sample output folder:"
    echo "  $INPUT_DIR"
    echo "Run the MAE training/export job first for person=$PERSON mask_ratio=$MASK_RATIO"
    exit 1
fi

for f in training_outputs_best.hdf5 validation_outputs_best.hdf5 testing_outputs_best.hdf5; do
    if [ ! -f "$INPUT_DIR/$f" ]; then
        echo "ERROR: missing required KNN input file:"
        echo "  $INPUT_DIR/$f"
        echo "Run or rerun MAE training/export first for person=$PERSON mask_ratio=$MASK_RATIO"
        exit 1
    fi
done

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null
eval "$(conda shell.bash hook)"

set +u
conda activate spacecats-cuda
set -u

cd "$PROJECT_DIR"

KNN_ARGS=(
    --input-folder "$INPUT_DIR"
    --output-folder "$OUTPUT_DIR"
)

if [ "$PERSON" = "leslie" ] && [ "$MASK_RATIO" = "0.0" ]; then
    echo "starting baseline KNN tuning"
else
    if [ ! -f "$PARAMS_FILE" ]; then
        echo "ERROR: missing shared KNN params:"
        echo "  $PARAMS_FILE"
        echo "Run Leslie/0.0 KNN first, or submit all jobs with pace/submit_knn.sh so dependencies are set."
        exit 1
    fi

    echo "starting KNN with shared params:"
    echo "  $PARAMS_FILE"

    KNN_ARGS+=(--params-file "$PARAMS_FILE")
fi

python "$PROJECT_DIR/src/dedupe_exports.py"

wait

srun -n 1 -c "$NUM_CORES" python -m src.analysis.knn_regressor "${KNN_ARGS[@]}"

if [ "$PERSON" = "leslie" ] && [ "$MASK_RATIO" = "0.0" ]; then
    if [ ! -f "$OUTPUT_DIR/knn_best_params.yaml" ]; then
        echo "ERROR: baseline KNN finished but did not create:"
        echo "  $OUTPUT_DIR/knn_best_params.yaml"
        exit 1
    fi

    cp "$OUTPUT_DIR/knn_best_params.yaml" "$PARAMS_FILE"
    echo "saved shared params:"
    echo "  $PARAMS_FILE"
fi

echo "done | $OUTPUT_DIR"