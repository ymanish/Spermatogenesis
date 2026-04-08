#!/bin/bash
################################################################################
# Local Markov Solver Test Script
################################################################################
# This script runs a quick Markov solver test on a small subset of nucleosomes
# for local testing and validation before submitting to the cluster.
#
# Usage:
#   ./run_local_markov.sh [dataset]
#
# Arguments:
#   dataset - "bound" or "unbound" (default: "bound")
#
# Example:
#   ./run_local_markov.sh bound
#   ./run_local_markov.sh unbound
#
# Author: MY
# Date: 2025-12-11
################################################################################

set -e  # Exit on error

# Configuration
DATASET="${1:-bound}"
BASE_DIR="hamnucret_data"
OUTPUT_DIR="output/local_markov_test"

# Input file selection
if [ "$DATASET" = "bound" ]; then
    INFILE="${BASE_DIR}/exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
elif [ "$DATASET" = "unbound" ]; then
    INFILE="${BASE_DIR}/exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"
else
    echo "ERROR: Unknown dataset: $DATASET"
    echo "Usage: $0 [bound|unbound]"
    exit 1
fi

# Parameters (test values)
K_WRAP=1.0
BINDING_SITES=14
PROT_K_BIND=1.0
PROT_K_UNBIND=89.7
PROT_P_CONC=0.0
PROT_COOPERATIVITY=0.0
TAU_MAX=1000.0
TAU_STEPS=500
METHOD="ode"
BATCH_SIZE=1
N_WORKERS=20
MAX_NUCS=20  # Test on small number

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "temps"

echo "=========================================="
echo "LOCAL MARKOV SOLVER TEST"
echo "=========================================="
echo "Dataset:          $DATASET"
echo "Input:            $INFILE"
echo "Output:           $OUTPUT_DIR"
echo ""
echo "--- Parameters ---"
echo "K Wrap:           $K_WRAP"
echo "Prot Conc:        $PROT_P_CONC μM"
echo "Cooperativity:    $PROT_COOPERATIVITY kT"
echo "Tau Max:         $TAU_MAX"
echo "Tau Steps:       $TAU_STEPS"
echo "Method:           $METHOD"
echo ""
echo "--- Execution ---"
echo "Workers:          $N_WORKERS"
echo "Batch Size:       $BATCH_SIZE"
echo "Max Nucs:         $MAX_NUCS (test mode)"
echo "=========================================="
echo ""

# Check if input file exists
if [ ! -f "$INFILE" ]; then
    echo "ERROR: Input file not found: $INFILE"
    exit 1
fi

# Run Markov solver
echo "-> Running Markov solver..."
python3 -m src.markov_execution.cli \
    --infile "$INFILE" \
    --storage_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --k_wrap "$K_WRAP" \
    --binding_sites "$BINDING_SITES" \
    --prot_k_bind "$PROT_K_BIND" \
    --prot_k_unbind "$PROT_K_UNBIND" \
    --prot_p_conc "$PROT_P_CONC" \
    --prot_cooperativity "$PROT_COOPERATIVITY" \
    --tau_max "$TAU_MAX" \
    --tau_steps "$TAU_STEPS" \
    --method "$METHOD" \
    --batch_size "$BATCH_SIZE" \
    --n_workers "$N_WORKERS" \
    --max_nucs "$MAX_NUCS" \
    --save_survival \
    --save_mfpt

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Markov solver test completed"
    echo ""
    echo "Results location: $OUTPUT_DIR"
    echo ""
    echo "To view results:"
    echo "  ls -lh $OUTPUT_DIR/*/*"
    echo ""
    echo "To analyze results in Python:"
    echo "  python3"
    echo "  >>> import polars as pl"
    echo "  >>> from pathlib import Path"
    echo "  >>> # Find the output directory"
    echo "  >>> dirs = list(Path('$OUTPUT_DIR').glob('*'))"
    echo "  >>> summary = pl.read_csv(dirs[0] / '${DATASET}_001_markov_summary.tsv', separator='\t')"
    echo "  >>> print(summary)"
else
    echo "ERROR: Markov solver test failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
