#!/bin/bash
# Run once on the PACE ICE login node to create the conda environment.
# Usage: bash pace/setup_pace.sh

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || {
    echo "could not load anaconda/miniconda — run 'module avail anaconda' to find the correct module name"
    exit 1
}

if conda env list | grep -q "spacecats-cuda"; then
    echo "spacecats-cuda already exists, skipping"
else
    conda config --set channel_priority strict
    conda env create -f "$PROJECT_DIR/cuda_environment.yaml"
fi

eval "$(conda shell.bash hook)"
conda activate spacecats-cuda
conda env config vars set PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
conda deactivate
conda activate spacecats-cuda

mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/logs"

echo "setup complete | PYTHONPATH=$PYTHONPATH"
echo ""
echo "transfer data (tuning uses small, training uses medium/large):"
echo "  scp galaxiesml_small.tar.gz <gtuser>@login-ice.pace.gatech.edu:$PROJECT_DIR/data/"
echo "  scp galaxiesml_medium.tar.gz <gtuser>@login-ice.pace.gatech.edu:$PROJECT_DIR/data/"
echo ""
echo "extract data:"
echo "  cd $PROJECT_DIR/data && tar -xzf galaxiesml_small.tar.gz"
echo "  cd $PROJECT_DIR/data && tar -xzf galaxiesml_medium.tar.gz"
echo ""
echo "submit your job only: bash pace/submit_tuning.sh <your_first_name>"
echo "submit tuning jobs: bash pace/submit_tuning.sh"
