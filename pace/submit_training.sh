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

    local mask_label="${mask_ratio//./_}"

    local tune_config="$PROJECT_ROOT/experiments/tune_mae_${person}_${mask_ratio}/best_overall_config.json"
    local train_config="$PROJECT_ROOT/configs/train_best_${person}_mask_${mask_label}.json"

    if [ ! -f "$tune_config" ]; then
        echo "skip missing best tuning config | person=$person mask_ratio=$mask_ratio"
        echo "  missing: $tune_config"
        echo "  run/resubmit tuning first: bash pace/submit_tuning.sh $person"
        return
    fi

    cp "$tune_config" "$train_config"

    echo "copied training config:"
    echo "  from: $tune_config"
    echo "  to:   $train_config"

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
