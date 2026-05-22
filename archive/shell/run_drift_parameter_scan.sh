#!/bin/bash
# Launch drift reversal parameter space analysis
# This script activates the conda environment and runs the parameter scan

echo "=========================================="
echo "Drift Reversal Parameter Space Analysis"
echo "=========================================="
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nucleosome

# Run the script
cd "$(dirname "$0")"
python src/scripts/drift_reversal_parameter_scan.py

echo ""
echo "Analysis complete!"
echo "Check output/drift_parameter_scan/ for results"
