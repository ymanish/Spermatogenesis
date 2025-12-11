#!/usr/bin/env python3
"""
Example: QSSA Validation for Nucleosome Simulations
====================================================

This script demonstrates how to validate the Quasi-Steady-State Approximation
(QSSA) for your simulation parameters.

The QSSA is valid when protamines equilibrate much faster than nucleosome
dynamics, allowing you to use an effective nucleosome-only model instead of
the full protamine-resolved Gillespie simulation.

Usage:
    python examples/example_qssa_validation.py

Author: MY
Date: 2024-11-19
"""

import numpy as np
import itertools
from pathlib import Path
import sys

from src.config.path import HAMNUCRET_DATA_DIR, RESULTS_DIR

from src.core.build_nucleosomes import nucleosome_generator, Nucleosomes
from src.core.protamine import protamines
from src.analysis.qssa_validator import (
    validate_qssa_for_system,
    print_qssa_summary,
    generate_qssa_report,
    save_qssa_data
)


def validate_simulation_parameters(
    file_path: Path,
    k_wrap: float,
    prot_params: dict,
    binding_sites: int = 14,
    max_nucs: int = 10,
    qssa_threshold: float = 0.1,
    output_dir: Path = None
):
    """
    Validate QSSA for a given set of simulation parameters.
    
    Args:
        file_path: Path to nucleosome data file
        k_wrap: Wrapping rate constant
        prot_params: Dict with keys: k_unbind, k_bind, p_conc, cooperativity
        binding_sites: Number of binding sites per nucleosome
        max_nucs: Maximum number of nucleosomes to load
        qssa_threshold: QSSA validity threshold (default 0.1)
        output_dir: Directory for output files (optional)
    """
    print("=" * 80)
    print("QSSA VALIDATION FOR SIMULATION PARAMETERS")
    print("=" * 80)
    
    print(f"\nInput File: {file_path}")
    print(f"Wrapping rate (k_wrap): {k_wrap}")
    print(f"Protamine parameters:")
    for key, val in prot_params.items():
        print(f"  {key}: {val}")
    print(f"QSSA threshold: {qssa_threshold}")
    print(f"Loading up to {max_nucs} nucleosomes...")
    
    # Load nucleosomes
    gen = nucleosome_generator(
        file_path=str(file_path),
        k_wrap=k_wrap,
        binding_sites=binding_sites
    )
    gen = itertools.islice(gen, max_nucs)
    
    nucs_list = list(gen)
    print(f"✓ Loaded {len(nucs_list)} nucleosomes")
    
    nucs = Nucleosomes(
        k_wrap=k_wrap,
        kT=1.0,
        nucleosomes=nucs_list,
        binding_sites=binding_sites
    )
    
    # Create protamine instance
    prot = protamines(
        k_unbind=prot_params['k_unbind'],
        k_bind=prot_params['k_bind'],
        p_conc=prot_params['p_conc'],
        cooperativity=prot_params['cooperativity']
    )
    
    # Validate QSSA
    print(f"\nValidating QSSA (this may take a moment)...")
    result = validate_qssa_for_system(
        nucs=nucs,
        prot=prot,
        threshold=qssa_threshold,
        beta=1.0  # Assuming kT=1
    )
    
    # Print summary
    print_qssa_summary(result, verbose=(max_nucs <= 20))
    
    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = output_dir / "qssa_validation_report.txt"
        generate_qssa_report(result, report_path, include_details=True)
        
        # Save data
        data_path = output_dir / "qssa_validation_data.tsv"
        save_qssa_data(result, data_path)
        
        print(f"\n✓ Results saved to {output_dir}/")
    
    return result


def compare_parameter_sets():
    """
    Example: Compare QSSA validity across different parameter sets.
    """
    print("\n" + "=" * 80)
    print("COMPARING QSSA VALIDITY ACROSS PARAMETER SETS")
    print("=" * 80)
    
    # Base parameters
    file_path = Path("hamnucret_data/unboundprom/breath_energy/001.tsv")
    k_wrap = 22.0
    
    # Different protamine concentrations
    param_sets = {
        "Low conc (0.1 μM)": {
            'k_unbind': 0.23,
            'k_bind': 2113,
            'p_conc': 0.1,
            'cooperativity': 4.5
        },
        "Medium conc (1.0 μM)": {
            'k_unbind': 0.23,
            'k_bind': 2113,
            'p_conc': 1.0,
            'cooperativity': 4.5
        },
        "High conc (10.0 μM)": {
            'k_unbind': 0.23,
            'k_bind': 2113,
            'p_conc': 10.0,
            'cooperativity': 4.5
        },
        "Very high conc (100 μM)": {
            'k_unbind': 0.23,
            'k_bind': 2113,
            'p_conc': 100.0,
            'cooperativity': 4.5
        }
    }
    
    results_summary = []
    
    for label, prot_params in param_sets.items():
        print(f"\n{'='*80}")
        print(f"Testing: {label}")
        print(f"{'='*80}")
        
        result = validate_simulation_parameters(
            file_path=file_path,
            k_wrap=k_wrap,
            prot_params=prot_params,
            max_nucs=5,  # Quick test with 5 nucleosomes
            output_dir=None  # Don't save individual results
        )
        
        results_summary.append({
            'label': label,
            'p_conc': prot_params['p_conc'],
            'tau_prot': result.tau_prot,
            'max_epsilon': result.max_epsilon_overall,
            'qssa_valid': result.system_qssa_valid
        })
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("QSSA VALIDITY COMPARISON")
    print("=" * 80)
    print(f"\n{'Condition':<25} {'p_conc (μM)':<15} {'tau_prot (s)':<15} {'max(ε)':<12} {'QSSA':<10}")
    print("-" * 80)
    
    for r in results_summary:
        status = "✓ VALID" if r['qssa_valid'] else "✗ INVALID"
        print(f"{r['label']:<25} {r['p_conc']:<15.2f} {r['tau_prot']:<15.6e} {r['max_epsilon']:<12.4f} {status:<10}")
    
    print("=" * 80)
    print("\nInterpretation:")
    print("  - QSSA valid (ε << 1): Protamines equilibrate fast → use reduced model")
    print("  - QSSA invalid (ε ~ 1 or > 1): Must use full Gillespie simulation")
    print("  - Higher p_conc → faster binding → smaller tau_prot → larger ε")


if __name__ == "__main__":
    # =========================================================================
    # EXAMPLE 1: Validate single parameter set
    # =========================================================================
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Parameter Set Validation")
    print("="*80)

    file_path = HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"

    # Your typical simulation parameters
    prot_params = {
        'k_unbind': 89.7,
        'k_bind': 1,
        'p_conc': 100,  # μM
        'cooperativity': 0.0
    }
    
    result = validate_simulation_parameters(
        file_path=file_path,
        k_wrap=1.0,
        prot_params=prot_params,
        max_nucs=10,
        qssa_threshold=0.1,
        output_dir=Path(RESULTS_DIR / "qssa_validation")
    )
    
    # =========================================================================
    # EXAMPLE 2: Compare across parameter sets
    # =========================================================================
    
    # Uncomment to run comparison
    # compare_parameter_sets()
    
    # =========================================================================
    # EXAMPLE 3: Check QSSA for your actual simulation config
    # =========================================================================
    
    # Use this to validate before running expensive simulations:
    #
    # result = validate_simulation_parameters(
    #     file_path=YOUR_DATA_FILE,
    #     k_wrap=YOUR_K_WRAP,
    #     prot_params={
    #         'k_unbind': YOUR_K_UNBIND,
    #         'k_bind': YOUR_K_BIND,
    #         'p_conc': YOUR_P_CONC,
    #         'cooperativity': YOUR_COOPERATIVITY
    #     },
    #     max_nucs=20,  # Test with representative sample
    #     output_dir=Path("output/qssa_validation")
    # )
    #
    # if result.system_qssa_valid:
    #     print("\n✓ QSSA is valid! You can use the reduced model.")
    # else:
    #     print("\n✗ QSSA is NOT valid. Use full Gillespie simulation.")
