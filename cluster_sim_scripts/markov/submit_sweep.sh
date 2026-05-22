#!/bin/bash
# Submit the Markov sweep as a SLURM array job, auto-sized from the grid file.
#
# Usage:
#   ./cluster_sim_scripts/markov/submit_sweep.sh                  # use default grid
#   ./cluster_sim_scripts/markov/submit_sweep.sh path/to/grid.tsv # custom grid
#
# Override the concurrency cap by exporting CONCURRENCY before running:
#   CONCURRENCY=10 ./cluster_sim_scripts/markov/submit_sweep.sh

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID="${1:-$HERE/sweep_grid.tsv}"
CONCURRENCY="${CONCURRENCY:-20}"

if [ ! -f "$GRID" ]; then
    echo "ERROR: grid file not found: $GRID"
    echo "Run: python $HERE/generate_sweep_grid.py"
    exit 1
fi

N=$(($(wc -l < "$GRID") - 1))   # subtract header

if [ "$N" -lt 1 ]; then
    echo "ERROR: grid file has no task rows: $GRID"
    exit 1
fi

echo "Submitting array of $N tasks (concurrency cap %$CONCURRENCY) from $GRID..."
sbatch --array=1-${N}%${CONCURRENCY} "$HERE/launch_markov_sweep.job" "$GRID"
