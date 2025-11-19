#!/bin/bash
# ---------------- SLURM parameters ----------------
#SBATCH -p all.q
#SBATCH --ntasks 40
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --tmp=10G
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH -J rmst_est
#SBATCH -D /home/pol_schiessel/maya620d/Spermatogensis
#SBATCH --output=/home/pol_schiessel/maya620d/Spermatogensis/log/rmst_%A_%a.out
#SBATCH --error=/home/pol_schiessel/maya620d/Spermatogensis/log/rmst_%A_%a.error
#SBATCH --exclude=compute-0-[13-15]
#SBATCH --time=200:00:00
#SBATCH --array=1-10%10

################################################################################
# RMST-Based Replicate Estimation - HPC Cluster Launch Script
################################################################################
# Author: MY
# Date: 2025-11-14
#
# This script launches RMST-based replicate estimation jobs using the
# modular rmst_estimator package.
#
# Usage:
#   sbatch cluster_sim_scripts/launch_rmst_estimation.sh
#
# Notes:
#   - Array job: processes multiple parameter combinations in parallel
#   - Uses new modular structure: src.analysis.rmst_estimator.cli
#   - Memory-efficient with reservoir sampling
################################################################################

# ---------------- Load modules --------------------
module load apps/singularity

# ---------------- Runtime setup -------------------
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
echo "Using TMPDIR=$TMPDIR"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_NTASKS"
echo "Started: $(date)"
echo "=========================================="
echo ""

################################################################################
# CONFIGURATION
################################################################################

# Output directory
OUTDIR="/group/pol_schiessel/Manish/Spermatogensis/output/rmst_replicate_estimation"

# Analysis parameters
KWRAP=1.0
PROT_K_UNBIND=89.7
PROT_K_BIND=1.0
N_NUCS=100
N_REPS=50
N_WORKERS="$SLURM_NTASKS"
TOLERANCE=0.05
TAU_MAX=10000.0
TAU_STEPS=1000
BATCH_SIZE=10
RANDOM_SAMPLE="True"
SEED="$SLURM_ARRAY_TASK_ID"

################################################################################
# PARAMETER GRID
################################################################################
# Format: "dataset:prot_conc:cooperativity"

PARAMS=(
    "bound:0.0:0.0"
    "unbound:0.0:0.0"
    "bound:100.0:0.0"
    "unbound:100.0:0.0"
    "bound:100.0:4.5"
    "unbound:100.0:4.5"
    "bound:500.0:4.5"
    "unbound:500.0:4.5"
    "bound:1000.0:4.5"
    "unbound:1000.0:4.5"
)

# Parse parameter for this array task
PARAM_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
N_PARAMS=${#PARAMS[@]}

if [ $PARAM_INDEX -ge $N_PARAMS ]; then
    echo "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds number of parameters ($N_PARAMS)"
    exit 1
fi

CURRENT_PARAM=${PARAMS[$PARAM_INDEX]}
IFS=':' read -r DATASET PROT_CONC COOPERATIVITY <<< "$CURRENT_PARAM"

################################################################################
# PRINT CONFIGURATION
################################################################################

echo "=========================================="
echo "PARAMETER COMBINATION $((PARAM_INDEX + 1))/$N_PARAMS"
echo "=========================================="
echo "Dataset:          $DATASET"
echo "K Wrap:           $KWRAP 1/sec"
echo "Prot K Unbind:    $PROT_K_UNBIND 1/sec"
echo "Prot K Bind:      $PROT_K_BIND 1/sec"
echo "Prot Conc:        $PROT_CONC μM"
echo "Cooperativity:    $COOPERATIVITY k_B T"
echo "N Nucleosomes:    $N_NUCS"
echo "N Replicates:     $N_REPS"
echo "Tolerance:        $TOLERANCE"
echo "Seed:             $SEED"
echo "Workers:          $N_WORKERS"
echo "Tau Max:          $TAU_MAX"
echo "Tau Steps:        $TAU_STEPS"
echo "Batch Size:       $BATCH_SIZE"
echo "Random Sample:    $RANDOM_SAMPLE"
echo "=========================================="
echo ""

################################################################################
# RUN RMST ESTIMATION (New Modular CLI)
################################################################################

echo "-> Launching RMST estimation via modular CLI..."
singularity exec \
    --env TMPDIR="$TMPDIR" \
    --bind $PWD:/project,"$TMPDIR":"$TMPDIR" \
    nucleosome.sif \
    python3 -m src.analysis.rmst_estimator.cli \
    --dataset "$DATASET" \
    --k-wrap "$KWRAP" \
    --prot-k-unbind "$PROT_K_UNBIND" \
    --prot-k-bind "$PROT_K_BIND" \
    --prot-p-conc "$PROT_CONC" \
    --prot-cooperativity "$COOPERATIVITY" \
    --n-nucs "$N_NUCS" \
    --n-reps "$N_REPS" \
    --n-workers "$N_WORKERS" \
    --tolerance "$TOLERANCE" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --random-sample "$RANDOM_SAMPLE" \
    --tau-max "$TAU_MAX" \
    --tau-steps "$TAU_STEPS" \
    --plot \
    --output-dir "$OUTDIR"

EXIT_CODE=$?

################################################################################
# CLEANUP AND STATUS
################################################################################

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: Job completed"
    echo "Parameter: $CURRENT_PARAM"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Job failed with exit code $EXIT_CODE"
    echo "Parameter: $CURRENT_PARAM"
    echo "Ended: $(date)"
    echo "=========================================="
    exit 1
fi
