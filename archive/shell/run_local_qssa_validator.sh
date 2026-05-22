#!/bin/bash

################################################################################
# LOCAL QSSA VALIDATION SCRIPT
################################################################################
# Configuration-based script for validating QSSA (Quasi-Steady-State 
# Approximation) in nucleosome-protamine simulations locally.
# 
# This script checks whether protamine binding/unbinding equilibrates faster
# than nucleosome wrapping/unwrapping, which determines if the hybrid
# rejection simulator can be used.
#
# Author: MY
# Date: 2025-11-27
################################################################################

################################################################################
# CONFIGURATION
################################################################################

# Input/Output
BASE_DIR="/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis"
INFILE="${BASE_DIR}/hamnucret_data/exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"
OUTPUT_DIR="${BASE_DIR}/output/qssa_validation/local_test"

# Nucleosome parameters
K_WRAP=1.0
BINDING_SITES=14

# Protamine parameters
PROT_K_UNBIND=89.7
PROT_K_BIND=1.0
PROT_P_CONC=100.0
PROT_COOPERATIVITY=0.0

# QSSA parameters
THRESHOLD=0.1

# Execution parameters
MAX_NUCLEOSOMES=20
VERBOSE="--verbose"  # Set to "--verbose" to enable OR "" to disable

################################################################################
# PARAMETER GRID (OPTIONAL)
################################################################################
# Uncomment and modify to test multiple parameter combinations
# Format: "prot_conc:cooperativity:k_wrap:threshold"

# PARAMS=(
#     "10.0:0.0:21.0:0.1"
#     "50.0:0.0:21.0:0.1"
#     "100.0:0.0:21.0:0.1"
#     "200.0:0.0:21.0:0.1"
#     "100.0:0.0:1.0:0.1"
#     "100.0:4.5:21.0:0.1"
# )

################################################################################
# PRINT CONFIGURATION
################################################################################

echo "=========================================="
echo "QSSA VALIDATION CONFIGURATION"
echo "=========================================="
echo "Input File:        $INFILE"
echo "Output Dir:        $OUTPUT_DIR"
echo ""
echo "Nucleosome Parameters:"
echo "  K Wrap:          $K_WRAP"
echo "  Binding Sites:   $BINDING_SITES"
echo ""
echo "Protamine Parameters:"
echo "  K Unbind:        $PROT_K_UNBIND"
echo "  K Bind:          $PROT_K_BIND"
echo "  P Conc:          $PROT_P_CONC μM"
echo "  Cooperativity:   $PROT_COOPERATIVITY"
echo ""
echo "QSSA Parameters:"
echo "  Threshold:       $THRESHOLD"
echo ""
echo "Execution:"
echo "  Max Nucleosomes: $MAX_NUCLEOSOMES"
echo "  Verbose:         $VERBOSE"
echo "=========================================="
echo ""

################################################################################
# RUN VALIDATION
################################################################################

# Single run mode (default)
if [ ${#PARAMS[@]} -eq 0 ]; then
    echo "-> Running single QSSA validation..."
    echo ""
    
    python -m src.analysis.qssa_validator.cli \
        "$INFILE" \
        --k-wrap "$K_WRAP" \
        --binding-sites "$BINDING_SITES" \
        --prot-k-unbind "$PROT_K_UNBIND" \
        --prot-k-bind "$PROT_K_BIND" \
        --prot-conc "$PROT_P_CONC" \
        --prot-cooperativity "$PROT_COOPERATIVITY" \
        --threshold "$THRESHOLD" \
        --max-nucleosomes "$MAX_NUCLEOSOMES" \
        --output-dir "$OUTPUT_DIR" \
        $VERBOSE

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ QSSA IS VALID"
        echo "=========================================="
        echo "→ Can use Hybrid Rejection Simulator"
        echo "→ Protamines equilibrate faster than nucleosomes"
        echo "→ Significant computational speedup possible"
        echo ""
        echo "Results saved to: $OUTPUT_DIR"
        echo "  - qssa_config.json: Configuration parameters"
        echo "  - qssa_validation_report.txt: Detailed report"
        echo "  - qssa_validation_data.tsv: Machine-readable data"
    elif [ $EXIT_CODE -eq 1 ]; then
        echo ""
        echo "=========================================="
        echo "✗ QSSA IS INVALID"
        echo "=========================================="
        echo "→ Must use full Gillespie Simulator"
        echo "→ Protamine dynamics too slow relative to nucleosomes"
        echo "→ Cannot integrate out protamine degrees of freedom"
        echo ""
        echo "Results saved to: $OUTPUT_DIR"
        echo "See report for detailed breakdown of failed states."
    else
        echo ""
        echo "✗ Validation failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi

# Parameter grid mode
else
    echo "-> Running QSSA validation parameter grid..."
    TOTAL_PARAMS=${#PARAMS[@]}
    
    # Arrays to store results
    declare -a GRID_RESULTS
    declare -a GRID_PARAMS
    
    for i in "${!PARAMS[@]}"; do
        PARAM_NUM=$((i + 1))
        CURRENT_PARAM=${PARAMS[$i]}
        IFS=':' read -r PROT_P_CONC PROT_COOPERATIVITY K_WRAP THRESHOLD <<< "$CURRENT_PARAM"
        
        # Create output subdirectory for this parameter combination
        PARAM_OUTPUT_DIR="${OUTPUT_DIR}/param_${PARAM_NUM}"
        
        echo ""
        echo "=========================================="
        echo "PARAMETER COMBINATION $PARAM_NUM/$TOTAL_PARAMS"
        echo "=========================================="
        echo "Prot Conc:         $PROT_P_CONC μM"
        echo "Cooperativity:     $PROT_COOPERATIVITY"
        echo "K Wrap:            $K_WRAP"
        echo "Threshold:         $THRESHOLD"
        echo "Output:            $PARAM_OUTPUT_DIR"
        echo "=========================================="
        echo ""

        python -m src.analysis.qssa_validator.cli \
            "$INFILE" \
            --k-wrap "$K_WRAP" \
            --binding-sites "$BINDING_SITES" \
            --prot-k-unbind "$PROT_K_UNBIND" \
            --prot-k-bind "$PROT_K_BIND" \
            --prot-conc "$PROT_P_CONC" \
            --prot-cooperativity "$PROT_COOPERATIVITY" \
            --threshold "$THRESHOLD" \
            --max-nucleosomes "$MAX_NUCLEOSOMES" \
            --output-dir "$PARAM_OUTPUT_DIR" \
            $VERBOSE
        
        EXIT_CODE=$?
        
        # Store result
        GRID_PARAMS[$i]="P=${PROT_P_CONC}, J=${PROT_COOPERATIVITY}, k=${K_WRAP}, ε=${THRESHOLD}"
        if [ $EXIT_CODE -eq 0 ]; then
            GRID_RESULTS[$i]="✓ VALID"
        elif [ $EXIT_CODE -eq 1 ]; then
            GRID_RESULTS[$i]="✗ INVALID"
        else
            echo ""
            echo "✗ Validation failed for parameter combination $PARAM_NUM with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    done
    
    # Print summary table
    echo ""
    echo "=========================================="
    echo "PARAMETER GRID SUMMARY"
    echo "=========================================="
    printf "%-5s %-50s %-15s\n" "No." "Parameters" "QSSA Status"
    echo "------------------------------------------------------------------"
    for i in "${!PARAMS[@]}"; do
        PARAM_NUM=$((i + 1))
        printf "%-5d %-50s %-15s\n" "$PARAM_NUM" "${GRID_PARAMS[$i]}" "${GRID_RESULTS[$i]}"
    done
    echo "=========================================="
    echo ""
    echo "✓ All parameter combinations completed successfully!"
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. Review validation report: ${OUTPUT_DIR}/qssa_validation_report.txt"
echo "  2. Analyze data: ${OUTPUT_DIR}/qssa_validation_data.tsv"
echo "  3. If QSSA is valid: Use src/scripts/local_test_hybrid.py"
echo "  4. If QSSA is invalid: Use src/simulation/cli.py (Gillespie)"
echo ""
