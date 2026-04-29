#!/bin/bash
# Submit training jobs for all 4 mask ratios.
# Usage:
#   bash pace/submit_training.sh # all 4 jobs
#   bash pace/submit_training.sh charlie # your job only
#
# Mask ratio assignments:
#   leslie -> 0.0 | charlie -> 0.25 | chris -> 0.5 | wen -> 0.75

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/logs"
FILTER="${1:-all}"
mkdir -p "$LOG_DIR"

submit() {
    local person="$1"
    local mask_ratio="$2"
    [[ "$FILTER" != "all" && "$FILTER" != "$person" ]] && return

    local project_root="/storage/ice-shared/cs7643/shared-group-project-data/the-space-cats/the-space-cats-project"
    local tune_config="$project_root/experiments/tune_mae_small_${person}_${mask_ratio}/best_overall_config.json"
    local train_config="$project_root/configs/best_config_${person}_${mask_ratio}.json"

    if [ ! -f "$train_config" ]; then
        echo "skip missing config | person=$person mask_ratio=$mask_ratio"
        echo "  or manually copy tuned config:"
        echo "  from: $tune_config"
        echo "   to: $train_config"
    fi

    local name="train_mae_${person}_${mask_ratio}"
    local job_id

    job_id=$(sbatch \
        --job-name="$name" \
        --output="$LOG_DIR/${name}_%j.out" \
        --error="$LOG_DIR/${name}_%j.err" \
        --export=ALL,MASK_RATIO="$mask_ratio",PERSON="$person" \
        "$SCRIPT_DIR/train_job.slurm" | awk '{print $NF}')

    echo "submitted | person=$person mask_ratio=$mask_ratio job_id=$job_id"
}

submit leslie 0.0
submit charlie 0.25
submit chris 0.5
submit wen 0.75
