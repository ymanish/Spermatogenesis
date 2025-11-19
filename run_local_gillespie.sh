#!/bin/bash

################################################################################
# LOCAL GILLESPIE SIMULATION SCRIPT
################################################################################
# Simple configuration-based script for running nucleosome simulations locally
# for testing and debugging purposes. This script runs on one type of input file.
# Author: MY
# Date: 2025-11-16

################################################################################
# CONFIGURATION
################################################################################

# Input/Output
BASE_DIR="/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis"
INFILE="${BASE_DIR}/hamnucret_data/exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
STORAGE_DIR="${BASE_DIR}/output/local_tests"

# Execution parameters
BATCH_SIZE=1
N_WORKERS=20
FLUSH_EVERY=1000  # Flush trajectory data more frequently (was 10000)

# Nucleosome parameters
K_WRAP=1.0
BINDING_SITES=14
INF_PROTAMINE="--inf_protamine"  # Set to "--inf_protamine" to enable

# Protamine parameters
PROT_K_UNBIND=89.7
PROT_K_BIND=1.0 
PROT_P_CONC=1000.0
PROT_COOPERATIVITY=4.5

# Simulation parameters
MAX_NUCS=20
REPLICATES=2
TAU_STOP=1000.0
TAU_NUM=1000
MAXPOINTS_SAVED_TRAJECTORIES=100

# Trajectory and renucleation options
SAVE_TRAJECTORIES="--save_trajectories"  # Set to "--save_trajectories" to enable OR "" to disable
RENUCLEATION=""       # Set to "--renucleation" to enable OR "" to disable

################################################################################
# PARAMETER GRID (OPTIONAL)
################################################################################
# Uncomment and modify to test multiple parameter combinations
# Format: "prot_conc:cooperativity:k_wrap"

# PARAMS=(
#     "0.0:0.0:1.0"
#     "100.0:0.0:1.0"
#     "100.0:4.5:1.0"
#     "500.0:4.5:1.0"
#     "1000.0:4.5:1.0"
# )

################################################################################
# PRINT CONFIGURATION
################################################################################

echo "=========================================="
echo "GILLESPIE SIMULATION CONFIGURATION"
echo "=========================================="
echo "Input File:        $INFILE"
echo "Storage Dir:       $STORAGE_DIR"
echo "K Wrap:            $K_WRAP"
echo "Prot K Unbind:     $PROT_K_UNBIND"
echo "Prot K Bind:       $PROT_K_BIND"
echo "Prot Conc:         $PROT_P_CONC"
echo "Cooperativity:     $PROT_COOPERATIVITY"
echo "Binding Sites:     $BINDING_SITES"
echo "Replicates:        $REPLICATES"
echo "Tau Stop:          $TAU_STOP"
echo "Tau Num:           $TAU_NUM"
echo "Workers:           $N_WORKERS"
echo "Batch Size:        $BATCH_SIZE"
echo "Flush Every:       $FLUSH_EVERY"
echo "Max Nucs (test):   $MAX_NUCS"
echo "Save Trajectories: $SAVE_TRAJECTORIES"
echo "Renucleation:      $RENUCLEATION"
echo "=========================================="
echo ""

################################################################################
# RUN SIMULATION
################################################################################

# Single run mode (default)
if [ ${#PARAMS[@]} -eq 0 ]; then
    echo "-> Running single simulation..."
    echo "-> Launching main worker script....."
    singularity exec \
        --bind $PWD:/project \
        nucleosome.sif \
        python3 -m src.simulation.cli \
        --infile "$INFILE" \
        --storage_dir "$STORAGE_DIR" \
        --batch_size "$BATCH_SIZE" \
        --n_workers "$N_WORKERS" \
        --flush_every "$FLUSH_EVERY" \
        --k_wrap "$K_WRAP" \
        --binding_sites "$BINDING_SITES" \
        --prot_k_unbind "$PROT_K_UNBIND" \
        --prot_k_bind "$PROT_K_BIND" \
        --prot_p_conc "$PROT_P_CONC" \
        --prot_cooperativity "$PROT_COOPERATIVITY" \
        --replicates "$REPLICATES" \
        --tau_stop "$TAU_STOP" \
        --tau_num "$TAU_NUM" \
        --maxpoints_saved_trajectories "$MAXPOINTS_SAVED_TRAJECTORIES" \
        --max_nucs "$MAX_NUCS" \
        $INF_PROTAMINE \
        $SAVE_TRAJECTORIES \
        $RENUCLEATION 

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Simulation completed successfully!"
    else
        echo ""
        echo "✗ Simulation failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi

# Parameter grid mode
else
    echo "-> Running parameter grid..."
    TOTAL_PARAMS=${#PARAMS[@]}
    
    for i in "${!PARAMS[@]}"; do
        PARAM_NUM=$((i + 1))
        CURRENT_PARAM=${PARAMS[$i]}
        IFS=':' read -r PROT_P_CONC PROT_COOPERATIVITY K_WRAP <<< "$CURRENT_PARAM"
        
        echo ""
        echo "=========================================="
        echo "PARAMETER COMBINATION $PARAM_NUM/$TOTAL_PARAMS"
        echo "=========================================="
        echo "Prot Conc:         $PROT_P_CONC"
        echo "Cooperativity:     $PROT_COOPERATIVITY"
        echo "K Wrap:            $K_WRAP"
        echo "=========================================="
        echo ""

        singularity exec \
        --bind $PWD:/project \
        nucleosome.sif \
        python3 -m src.simulation.cli \
            --infile "$INFILE" \
            --storage_dir "$STORAGE_DIR" \
            --batch_size "$BATCH_SIZE" \
            --n_workers "$N_WORKERS" \
            --flush_every "$FLUSH_EVERY" \
            --k_wrap "$K_WRAP" \
            --binding_sites "$BINDING_SITES" \
            --prot_k_unbind "$PROT_K_UNBIND" \
            --prot_k_bind "$PROT_K_BIND" \
            --prot_p_conc "$PROT_P_CONC" \
            --prot_cooperativity "$PROT_COOPERATIVITY" \
            --replicates "$REPLICATES" \
            --tau_stop "$TAU_STOP" \
            --tau_num "$TAU_NUM" \
            --maxpoints_saved_trajectories "$MAXPOINTS_SAVED_TRAJECTORIES" \
            --max_nucs "$MAX_NUCS" \
            $INF_PROTAMINE \
            $SAVE_TRAJECTORIES \
            $RENUCLEATION
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -ne 0 ]; then
            echo ""
            echo "✗ Simulation failed for parameter combination $PARAM_NUM with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    done
    
    echo ""
    echo "✓ All parameter combinations completed successfully!"
fi

echo ""
echo "Results saved to: $STORAGE_DIR"
