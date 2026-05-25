#!/bin/bash
# Submit the Gillespie-event sweep as a SLURM array job, auto-sized from the grid file.
#
# sweep_grid.tsv is a committed artifact — regenerate it LOCALLY (on your
# laptop/workstation, where python3 + pyyaml are available natively) after
# editing gillespie_event_sweep.yaml, then commit + push + pull on the cluster:
#
#   # locally:
#   python cluster_sim_scripts/gillespie_event/generate_sweep_grid.py --no-validate
#   git add cluster_sim_scripts/gillespie_event/gillespie_event_sweep.yaml \
#           cluster_sim_scripts/gillespie_event/sweep_grid.tsv
#   git commit -m "..." && git push
#
#   # on the cluster:
#   git pull
#   ./cluster_sim_scripts/gillespie_event/submit_sweep.sh
#
# Usage:
#   ./cluster_sim_scripts/gillespie_event/submit_sweep.sh                  # default grid
#   ./cluster_sim_scripts/gillespie_event/submit_sweep.sh path/to/grid.tsv # custom grid
#
# Override the concurrency cap by exporting CONCURRENCY:
#   CONCURRENCY=10 ./cluster_sim_scripts/gillespie_event/submit_sweep.sh

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID="${1:-$HERE/sweep_grid.tsv}"
CONCURRENCY="${CONCURRENCY:-20}"

if [ ! -f "$GRID" ]; then
    echo "ERROR: grid file not found: $GRID"
    echo ""
    echo "sweep_grid.tsv is a committed artifact. Regenerate it locally:"
    echo "  python $HERE/generate_sweep_grid.py --no-validate"
    echo "Then: git add ... && git commit && git push, and 'git pull' on the cluster."
    exit 1
fi

# Sanity check: warn if the TSV is older than the YAML (likely stale).
YAML="$HERE/gillespie_event_sweep.yaml"
if [ -f "$YAML" ] && [ "$YAML" -nt "$GRID" ]; then
    echo "WARN: $YAML is newer than $GRID."
    echo "      sweep_grid.tsv may be stale. Regenerate locally and recommit."
    echo ""
fi

N=$(($(wc -l < "$GRID") - 1))   # subtract header

if [ "$N" -lt 1 ]; then
    echo "ERROR: grid file has no task rows: $GRID"
    exit 1
fi

echo "Submitting array of $N tasks (concurrency cap %$CONCURRENCY) from $GRID..."
sbatch --array=1-${N}%${CONCURRENCY} "$HERE/launch_gillespie_event_sweep.job" "$GRID"
