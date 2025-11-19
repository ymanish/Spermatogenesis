#!/bin/bash
#
# rebuild_simulation_index.sh
#
# Rebuild the simulation index after parallel jobs complete.
# Run this on the cluster after your SLURM jobs finish.
#
# Usage:
#   bash rebuild_simulation_index.sh
#   bash rebuild_simulation_index.sh /custom/results/path
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-output/simulations}"

echo "======================================================================="
echo "Rebuilding Simulation Index"
echo "======================================================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if singularity container exists
if [ ! -f "$SCRIPT_DIR/nucleosome.sif" ]; then
    echo "ERROR: nucleosome.sif not found in $SCRIPT_DIR"
    echo "Please build the container first."
    exit 1
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Count parameter directories
PARAM_DIRS=$(find "$RESULTS_DIR" -maxdepth 1 -type d -not -path "$RESULTS_DIR" | wc -l)
echo "Found $PARAM_DIRS parameter directories"

if [ "$PARAM_DIRS" -eq 0 ]; then
    echo "WARNING: No parameter directories found. Make sure simulations ran successfully."
    exit 1
fi

echo ""
echo "Running rebuild_index.py..."
echo ""

# Run the rebuild script
if [ -n "$1" ]; then
    # Custom results directory provided
    singularity exec \
        --bind "$SCRIPT_DIR:/project" \
        "$SCRIPT_DIR/nucleosome.sif" \
        python3 /project/rebuild_index.py --results-dir "/project/$RESULTS_DIR"
else
    # Use default from config
    singularity exec \
        --bind "$SCRIPT_DIR:/project" \
        "$SCRIPT_DIR/nucleosome.sif" \
        python3 /project/rebuild_index.py
fi

echo ""
echo "======================================================================="
echo "✓ Index rebuild complete!"
echo "======================================================================="
echo ""
echo "The index file is ready at: $RESULTS_DIR/simulation_index.csv"
echo ""
echo "You can now query simulations using:"
echo "  storage = SimulationStorage(RESULTS_DIR, use_index=True)"
echo "  matches = storage.find_simulations(...)"
echo ""
