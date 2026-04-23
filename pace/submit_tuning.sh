#!/bin/bash
# Submit tuning jobs for all 4 mask ratios.
# Usage:
#   bash pace/submit_tuning.sh # all 4 jobs
#   bash pace/submit_tuning.sh charlie # your job only
#
# Mask ratio assignments:
#   leslie -> 0.0 | charlie -> 0.25 | chris -> 0.5 | wen -> 0.75

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/logs"
FILTER="${1:-all}"
mkdir -p "$LOG_DIR"

submit() {
    local person="$1" mask_ratio="$2"
    [[ "$FILTER" != "all" && "$FILTER" != "$person" ]] && return

    local name="tune_mae_${person}_${mask_ratio}"
    local job_id
    job_id=$(sbatch \
        --job-name="$name" \
        --output="$LOG_DIR/${name}_%j.out" \
        --error="$LOG_DIR/${name}_%j.err" \
        --export=ALL,MASK_RATIO="$mask_ratio",PERSON="$person" \
        "$SCRIPT_DIR/tune_job.slurm" | awk '{print $NF}')

    echo "submitted | person=$person mask_ratio=$mask_ratio job_id=$job_id"
}

submit leslie 0.0
submit charlie 0.25
submit chris 0.5
submit wen 0.75
