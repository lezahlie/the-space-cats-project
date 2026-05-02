#!/bin/bash
# Submit KNN jobs for all 4 mask ratios.
#
# Usage:
#   bash pace/submit_knn.sh          # all 4 jobs
#   bash pace/submit_knn.sh leslie   # Leslie only
#   bash pace/submit_knn.sh charlie  # Charlie only
#
# Leslie/0.0 tunes KNN and saves configs/knn_best_params.yaml.
# Everyone else uses that shared params file.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/storage/ice-shared/cs7643/shared-group-project-data/the-space-cats/the-space-cats-project"
LOG_DIR="${PROJECT_DIR}/logs"
FILTER="${1:-all}"

mkdir -p "$LOG_DIR"

submit_knn() {
    local person="$1"
    local mask_ratio="$2"
    local dependency="${3:-}"

    if [[ "$FILTER" != "all" && "$FILTER" != "$person" ]]; then
        return
    fi

    local name="knn_${person}_${mask_ratio}"
    local sbatch_args=(
        --job-name="$name"
        --output="$LOG_DIR/${name}_%j.out"
        --error="$LOG_DIR/${name}_%j.err"
        --export=ALL,PERSON="$person",MASK_RATIO="$mask_ratio"
    )

    if [[ -n "$dependency" ]]; then
        sbatch_args+=(--dependency="afterok:${dependency}")
    fi

    local job_id
    job_id=$(sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/knn_job.slurm" | awk '{print $NF}')

    echo "submitted | person=$person mask_ratio=$mask_ratio job_id=$job_id dependency=${dependency:-none}"
}

leslie_job_id=""

if [[ "$FILTER" == "all" || "$FILTER" == "leslie" ]]; then
    leslie_job_id=$(sbatch \
        --job-name="knn_leslie_0.0" \
        --output="$LOG_DIR/knn_leslie_0.0_%j.out" \
        --error="$LOG_DIR/knn_leslie_0.0_%j.err" \
        --export=ALL,PERSON="leslie",MASK_RATIO="0.0" \
        "$SCRIPT_DIR/knn_job.slurm" | awk '{print $NF}')

    echo "submitted | person=leslie mask_ratio=0.0 job_id=$leslie_job_id dependency=none"
fi

if [[ "$FILTER" == "all" ]]; then
    submit_knn charlie 0.25 "$leslie_job_id"
    submit_knn chris 0.5 "$leslie_job_id"
    submit_knn wen 0.75 "$leslie_job_id"
else
    submit_knn charlie 0.25
    submit_knn chris 0.5
    submit_knn wen 0.75
fi