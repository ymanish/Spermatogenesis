#!/bin/bash
# Identify failed array tasks in the Gillespie-event sweep by scanning .error logs.
# A task is FAILED if its .error file contains a real failure marker
# (Traceback / Error / Exception / oom-kill) -- NOT merely because it is
# non-empty (tqdm writes its progress bar to stderr in every job).
#
# Usage: ./find_failed.sh [LOG_DIR]
set -euo pipefail

LOG_DIR="${1:-/home/pol_schiessel/maya620d/Spermatogenesis/log}"

# Failure markers. Word-anchored "Error"/"Exception" to avoid matching e.g.
# "0 errors". Add patterns here if you spot other failure modes.
PATTERN='Traceback \(most recent call last\)|[A-Za-z]*Error\b|[A-Za-z]*Exception\b|oom-kill|Killed|CANCELLED'

failed=()
for ef in "$LOG_DIR"/gillespie_event_sweep_*.error; do
    [ -e "$ef" ] || continue
    if grep -Eq "$PATTERN" "$ef"; then
        # array task id is the trailing _<id> before .error
        base=$(basename "$ef" .error)        # gillespie_event_sweep_<JOB>_<ARRAY>
        array_id=${base##*_}
        out="${ef%.error}.out"
        ds=$(grep -m1 'dataset_name:' "$out" 2>/dev/null | awk '{print $2}')
        pc=$(grep -m1 'prot_p_conc:'  "$out" 2>/dev/null | awk '{print $2}')
        co=$(grep -m1 'prot_cooperativity:' "$out" 2>/dev/null | awk '{print $2}')
        reason=$(grep -Eo "$PATTERN" "$ef" | tail -1)
        printf 'task %-4s  %-50s p=%-7s coop=%-5s  [%s]\n' \
               "$array_id" "${ds:-?}" "${pc:-?}" "${co:-?}" "$reason"
        failed+=("$array_id")
    fi
done

if [ ${#failed[@]} -eq 0 ]; then
    echo "No failed tasks found."
    exit 0
fi

# Sorted, de-duplicated, comma-joined list for resubmission.
list=$(printf '%s\n' "${failed[@]}" | sort -n | uniq | paste -sd, -)
echo ""
echo "Failed task count: ${#failed[@]}"
echo "Resubmit with:"
echo "  sbatch --array=${list}%20 cluster_sim_scripts/gillespie_event/launch_gillespie_event_sweep.job cluster_sim_scripts/gillespie_event/sweep_grid.tsv"
