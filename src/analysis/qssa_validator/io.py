"""
QSSA I/O and Reporting
=======================

This module handles file input/output and report generation for QSSA validation:
- Console output (summary printing)
- Text report generation
- TSV data export

Author: MY
Date: 2025-11-27
"""

from pathlib import Path
from typing import Optional

from .validation import SystemQSSAResult, QSSAValidationResult


def print_qssa_summary(result: SystemQSSAResult, verbose: bool = False):
    """
    Print a formatted summary of QSSA validation results.
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        verbose: If True, print per-nucleosome details
        
    Examples:
        >>> from src.analysis.qssa_validator import validate_qssa_for_system, print_qssa_summary
        >>> result = validate_qssa_for_system(nucs, prot)
        >>> print_qssa_summary(result, verbose=True)
    """
    print("=" * 80)
    print("QSSA VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nProtamine Fast Timescale:")
    print(f"  tau_prot = {result.tau_prot:.6e} seconds")
    
    print(f"\nSystem Overview:")
    print(f"  Total nucleosomes: {result.num_nucleosomes}")
    print(f"  QSSA valid: {result.num_valid} ({result.fraction_valid*100:.1f}%)")
    print(f"  QSSA invalid: {result.num_invalid}")
    print(f"  Maximum epsilon: {result.max_epsilon_overall:.4f}")
    
    if result.system_qssa_valid:
        print(f"\n✓ SYSTEM QSSA IS VALID")
        print(f"  → Can use effective nucleosome-only model")
        print(f"  → Protamines can be integrated out")
    else:
        print(f"\n✗ SYSTEM QSSA IS INVALID")
        print(f"  → Must use full protamine-resolved Gillespie simulation")
        print(f"  → Protamine dynamics too slow relative to nucleosomes")
    
    if verbose and result.num_nucleosomes <= 20:
        print(f"\n" + "=" * 80)
        print("PER-NUCLEOSOME DETAILS")
        print("=" * 80)
        
        for nuc_result in result.nucleosome_results:
            status = "✓ VALID" if nuc_result.qssa_valid else "✗ INVALID"
            
            # Count how many states failed
            num_states = len(nuc_result.qssa_valid_per_ij)
            num_failed = sum(1 for valid in nuc_result.qssa_valid_per_ij.values() if not valid)
            num_passed = num_states - num_failed
            
            print(f"\nNucleosome {nuc_result.nuc_id} (subid={nuc_result.subid}): {status}")
            print(f"  eps_max = {nuc_result.eps_max:.4f}")
            print(f"  States: {num_passed}/{num_states} passed, {num_failed}/{num_states} failed")
            
            if nuc_result.eps_max > 0.05:  # Show problematic states
                print(f"  Problematic states (i,j) with epsilon > {nuc_result.threshold}:")
                count = 0
                for ij, eps in nuc_result.epsilons.items():
                    if not nuc_result.qssa_valid_per_ij[ij]:
                        ts = nuc_result.tau_slow[ij]
                        i, j = ij
                        # Calculate n_open for reference
                        L = 14  # Assuming binding_sites = 14
                        n_open = i + (L - 1 - j)
                        print(f"    (i={i:2d}, j={j:2d}, n_open={n_open:2d}): epsilon={eps:.4f}, tau_slow={ts:.6e}")
                        count += 1
                        if count >= 10 and num_failed > 10:
                            print(f"    ... and {num_failed - 10} more failed states (see TSV for full list)")
                            break
    
    print("=" * 80)


def generate_qssa_report(
    result: SystemQSSAResult,
    output_path: Path,
    include_details: bool = True
):
    """
    Generate a detailed text report of QSSA validation.
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        output_path: Path to save report file
        include_details: Include per-nucleosome details
        
    Examples:
        >>> from pathlib import Path
        >>> output_path = Path("output/qssa/validation_report.txt")
        >>> generate_qssa_report(result, output_path, include_details=True)
        ✓ QSSA report saved to output/qssa/validation_report.txt
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QSSA VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PROTAMINE PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Fast timescale (tau_prot): {result.tau_prot:.6e} seconds\n\n")
        
        f.write("SYSTEM SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total nucleosomes: {result.num_nucleosomes}\n")
        f.write(f"QSSA valid: {result.num_valid} ({result.fraction_valid*100:.1f}%)\n")
        f.write(f"QSSA invalid: {result.num_invalid}\n")
        f.write(f"Maximum epsilon: {result.max_epsilon_overall:.6f}\n")
        f.write(f"System QSSA valid: {'YES' if result.system_qssa_valid else 'NO'}\n\n")
        
        if result.system_qssa_valid:
            f.write("RECOMMENDATION: Use effective nucleosome-only model\n")
            f.write("Protamines equilibrate much faster than nucleosome dynamics.\n\n")
        else:
            f.write("RECOMMENDATION: Use full protamine-resolved Gillespie simulation\n")
            f.write("Timescale separation is insufficient for QSSA.\n\n")
        
        if include_details:
            f.write("=" * 80 + "\n")
            f.write("PER-NUCLEOSOME DETAILS\n")
            f.write("=" * 80 + "\n\n")
            
            for nuc_result in result.nucleosome_results:
                f.write(f"Nucleosome ID: {nuc_result.nuc_id}, SubID: {nuc_result.subid}\n")
                f.write(f"Status: {'VALID' if nuc_result.qssa_valid else 'INVALID'}\n")
                f.write(f"Maximum epsilon: {nuc_result.eps_max:.6f}\n")
                
                f.write("\nTimescale ratios by state (i,j):\n")
                f.write(f"  {'(i,j)':<12} {'n_open':<8} {'epsilon':<12} {'tau_slow (s)':<15}\n")
                f.write(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*15}\n")
                
                # Sort by (i,j) tuples
                L = 14  # Assuming binding_sites = 14
                for ij in sorted(nuc_result.epsilons.keys()):
                    eps = nuc_result.epsilons[ij]
                    ts = nuc_result.tau_slow[ij]
                    i, j = ij
                    n_open = i + (L - 1 - j)
                    f.write(f"  ({i:2d},{j:2d}){' '*6} {n_open:<8d} {eps:<12.6f} {ts:<15.6e}\n")
                f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"✓ QSSA report saved to {output_path}")


def save_qssa_data(result: SystemQSSAResult, output_path: Path):
    """
    Save QSSA validation data in TSV format for analysis.
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        output_path: Path to save TSV file
    
    TSV Columns:
        - nuc_id: Nucleosome ID
        - subid: Nucleosome sub-ID
        - i, j: State indices
        - n_open: Total open contacts
        - tau_prot: Protamine fast timescale
        - tau_slow: Nucleosome slow timescale for this state
        - epsilon: Timescale ratio (tau_prot / tau_slow)
        - threshold: QSSA threshold used
        - qssa_valid_local: Whether this (i,j) state satisfies QSSA
        - qssa_valid_global: Whether the entire nucleosome satisfies QSSA
        
    Examples:
        >>> from pathlib import Path
        >>> output_path = Path("output/qssa/validation_data.tsv")
        >>> save_qssa_data(result, output_path)
        ✓ QSSA data saved to output/qssa/validation_data.tsv
    """
    import pandas as pd
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for nuc_result in result.nucleosome_results:
        # assume all nucleosomes have same binding_sites; or pass it in explicitly
        L = 14  # or nuc.binding_sites if you store it
        for (i, j), eps in nuc_result.epsilons.items():
            ts = nuc_result.tau_slow[(i, j)]
            qssa_local = nuc_result.qssa_valid_per_ij[(i, j)]
            n_open = i + (L - 1 - j)
            records.append({
                'nuc_id': nuc_result.nuc_id,
                'subid': nuc_result.subid,
                'i': i,
                'j': j,
                'n_open': n_open,
                'tau_prot': nuc_result.tau_prot,
                'tau_slow': ts,
                'epsilon': eps,
                'threshold': nuc_result.threshold,
                'qssa_valid_local': qssa_local,
                'qssa_valid_global': nuc_result.qssa_valid
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"✓ QSSA data saved to {output_path}")
