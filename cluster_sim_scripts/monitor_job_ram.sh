#!/usr/bin/env bash
#
# Monitor RAM usage of SLURM jobs in real-time
# Supports array jobs and shows live memory usage via sstat (running) and sacct (completed)
#
# Usage:
#   ./monitor_job_ram.sh <JOB_ID> [ARRAY_TASK_ID]
#
# Examples:
#   ./monitor_job_ram.sh 4805598              # Monitor all array tasks
#   ./monitor_job_ram.sh 4805598 2            # Monitor specific task 4805598_2
#   ./monitor_job_ram.sh 4805598 2 --once     # Single snapshot (no loop)

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <JOB_ID> [ARRAY_TASK_ID] [--once]"
    echo ""
    echo "Examples:"
    echo "  $0 4805598              # Monitor all array tasks (live updates)"
    echo "  $0 4805598 2            # Monitor specific task 4805598_2"
    echo "  $0 4805598 --once       # Single snapshot of all tasks"
    echo "  $0 4805598 2 --once     # Single snapshot of task 2"
    exit 1
fi

JOB_ID="$1"
ARRAY_TASK=""
ONCE_MODE=false

# Parse arguments
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --once)
            ONCE_MODE=true
            ;;
        *)
            if [ -z "$ARRAY_TASK" ]; then
                ARRAY_TASK="$1"
            fi
            ;;
    esac
    shift
done

# Convert memory strings to GB for consistent display
convert_to_gb() {
    local mem="$1"
    if [ -z "$mem" ] || [ "$mem" = "0" ] || [ "$mem" = "-" ]; then
        echo "0.00"
        return
    fi
    
    # Remove trailing units and convert
    if [[ "$mem" =~ ([0-9.]+)G ]]; then
        printf "%.2f" "${BASH_REMATCH[1]}"
    elif [[ "$mem" =~ ([0-9.]+)M ]]; then
        printf "%.2f" "$(echo "${BASH_REMATCH[1]} / 1024" | bc -l)"
    elif [[ "$mem" =~ ([0-9.]+)K ]]; then
        printf "%.2f" "$(echo "${BASH_REMATCH[1]} / 1024 / 1024" | bc -l)"
    elif [[ "$mem" =~ ^[0-9.]+$ ]]; then
        # Assume bytes if no unit
        printf "%.2f" "$(echo "$mem / 1024 / 1024 / 1024" | bc -l)"
    else
        echo "0.00"
    fi
}

print_header() {
    printf "\n%-19s | %-15s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
           "Time" "JobID" "State" "Elapsed" "ReqMem" "MaxRSS" "AvgRSS"
    printf "%0.s-" {1..100}; echo
}

print_row() {
    local time="$1" jobid="$2" state="$3" elapsed="$4" reqmem="$5" maxrss="$6" avgrss="$7"
    printf "%-19s | %-15s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
           "$time" "$jobid" "$state" "$elapsed" "$reqmem" "$maxrss" "$avgrss"
}

get_job_data() {
    local job_spec="$1"
    local timestamp="$2"
    
    # Try sstat first (for RUNNING jobs with .batch suffix)
    local sstat_data
    sstat_data=$(sstat -j "${job_spec}.batch" -n -P \
                 --format=JobID,MaxRSS,AveRSS 2>/dev/null || echo "")
    
    if [ -n "$sstat_data" ]; then
        # Running job - get live data from sstat
        local jid maxrss avgrss
        IFS='|' read -r jid maxrss avgrss <<< "$sstat_data"
        
        # Get state and elapsed from sacct
        local sacct_data
        sacct_data=$(sacct -j "$job_spec" -n -P \
                     --format=State,Elapsed,ReqMem --units=G 2>/dev/null | head -n1)
        
        local state elapsed reqmem
        IFS='|' read -r state elapsed reqmem <<< "$sacct_data"
        
        # Convert memory to GB
        maxrss_gb=$(convert_to_gb "$maxrss")
        avgrss_gb=$(convert_to_gb "$avgrss")
        
        print_row "$timestamp" "$job_spec" "${state:-RUNNING}" "$elapsed" \
                  "$reqmem" "${maxrss_gb}G" "${avgrss_gb}G"
        return 0
    fi
    
    # If sstat failed, try sacct (for PENDING/COMPLETED/FAILED jobs)
    local sacct_data
    sacct_data=$(sacct -j "$job_spec" -n -P \
                 --format=State,Elapsed,ReqMem,MaxRSS --units=G 2>/dev/null | head -n1)
    
    if [ -n "$sacct_data" ]; then
        local state elapsed reqmem maxrss
        IFS='|' read -r state elapsed reqmem maxrss <<< "$sacct_data"
        
        # For completed jobs, MaxRSS is already in GB from sacct
        maxrss_gb=$(convert_to_gb "$maxrss")
        
        print_row "$timestamp" "$job_spec" "$state" "$elapsed" \
                  "$reqmem" "${maxrss_gb}G" "-"
        return 0
    fi
    
    # No data available
    print_row "$timestamp" "$job_spec" "NO_DATA" "-" "-" "-" "-"
    return 1
}

monitor_single_task() {
    local job_spec="$1"
    
    if [ "$ONCE_MODE" = true ]; then
        print_header
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        get_job_data "$job_spec" "$timestamp"
        return
    fi
    
    echo "Monitoring job ${job_spec} (Press Ctrl+C to stop)"
    print_header
    
    while true; do
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        if ! get_job_data "$job_spec" "$timestamp"; then
            sleep 5
            continue
        fi
        
        # Check if job is in terminal state
        local state
        state=$(sacct -j "$job_spec" -n -P --format=State 2>/dev/null | head -n1)
        
        case "$state" in
            COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL)
                echo ""
                echo "✓ Job ${job_spec} reached terminal state: ${state}"
                return 0
                ;;
        esac
        
        sleep 5
    done
}

monitor_all_tasks() {
    local job_id="$1"
    
    # Get list of array task IDs
    local task_ids
    task_ids=$(sacct -j "$job_id" -n -P --format=JobID 2>/dev/null | \
               grep -E "${job_id}_[0-9]+" | \
               sed "s/${job_id}_//" | \
               sed 's/\.batch$//' | \
               sort -u)
    
    if [ -z "$task_ids" ]; then
        echo "No array tasks found for job ${job_id}"
        exit 1
    fi
    
    local task_count
    task_count=$(echo "$task_ids" | wc -l)
    
    if [ "$ONCE_MODE" = true ]; then
        echo "Job ${job_id} - Array tasks: ${task_count}"
        print_header
        
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        for task_id in $task_ids; do
            get_job_data "${job_id}_${task_id}" "$timestamp"
        done
        
        echo ""
        echo "Summary:"
        sacct -j "$job_id" -n --format=JobID,State,MaxRSS --units=G | \
            grep -E "${job_id}_[0-9]+" | grep -v ".batch" | \
            awk '{print "  Task " $1 ": " $2 " (MaxRSS: " $3 ")"}'
        return
    fi
    
    echo "Monitoring job ${job_id} - ${task_count} array tasks (Press Ctrl+C to stop)"
    echo "Showing summary every 10 seconds..."
    
    while true; do
        print_header
        
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        local completed=0
        local running=0
        local pending=0
        local failed=0
        
        for task_id in $task_ids; do
            local state
            state=$(sacct -j "${job_id}_${task_id}" -n -P --format=State 2>/dev/null | head -n1)
            
            case "$state" in
                COMPLETED) ((completed++)) ;;
                RUNNING) ((running++)) ;;
                PENDING) ((pending++)) ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL) ((failed++)) ;;
            esac
            
            get_job_data "${job_id}_${task_id}" "$timestamp"
        done
        
        echo ""
        printf "Status: %d completed | %d running | %d pending | %d failed (Total: %d)\n" \
               "$completed" "$running" "$pending" "$failed" "$task_count"
        
        # Check if all tasks are done
        if [ $((completed + failed)) -eq "$task_count" ]; then
            echo ""
            echo "✓ All tasks completed!"
            echo ""
            echo "Final Summary:"
            sacct -j "$job_id" -n --format=JobID,State,Elapsed,MaxRSS --units=G | \
                grep -E "${job_id}_[0-9]+" | grep -v ".batch"
            return 0
        fi
        
        sleep 10
    done
}

# Main logic
if [ -n "$ARRAY_TASK" ]; then
    # Monitor specific task
    monitor_single_task "${JOB_ID}_${ARRAY_TASK}"
else
    # Monitor all tasks
    monitor_all_tasks "$JOB_ID"
fi
