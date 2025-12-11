"""
QSSA (Quasi-Steady-State Approximation) Validator
==================================================

This module validates whether the QSSA is applicable for a given
simulation configuration by comparing timescales:

- tau_prot: Fast timescale for protamine binding/unbinding
- tau_slow(n): Slow timescale for nucleosome wrapping/unwrapping at level n

The QSSA is valid when epsilon = tau_prot / tau_slow << 1 (typically < 0.1),
meaning protamines equilibrate much faster than nucleosome dynamics.

Key Functions:
--------------
- compute_protamine_fast_timescale(): Calculate tau_prot from protamine parameters
- validate_qssa_for_nucleosome(): Check QSSA validity for a single nucleosome
- validate_qssa_for_system(): Check QSSA validity for entire system
- generate_qssa_report(): Create detailed validation report

Author: MY
Date: 2024-11-19
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from src.core.nucleosomes import Nucleosome, Nucleosomes
from src.core.protamine import protamines


@dataclass
class QSSAValidationResult:
    """Results of QSSA validation for a single nucleosome."""
    nuc_id: str
    subid: int
    tau_prot: float  # Protamine fast timescale (seconds)
    tau_slow: Dict[int, float]  # Nucleosome slow timescales per n
    epsilons: Dict[int, float]  # Timescale ratios epsilon(n) = tau_prot / tau_slow(n)
    eps_max: float  # Maximum epsilon across all n
    qssa_valid: bool  # True if eps_max <= threshold
    threshold: float  # Threshold used for validation


@dataclass
class SystemQSSAResult:
    """Results of QSSA validation for entire system."""
    tau_prot: float
    num_nucleosomes: int
    num_valid: int
    num_invalid: int
    fraction_valid: float
    max_epsilon_overall: float
    nucleosome_results: List[QSSAValidationResult]
    system_qssa_valid: bool


def compute_protamine_fast_timescale(
    prot: protamines,
    cooperativity_factor: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Compute the fast timescale for protamine binding/unbinding.
    
    The protamine fast timescale is determined by the fastest unbinding rate,
    which occurs when a protamine is fully surrounded by other protamines
    (maximum cooperativity effect).
    
    tau_prot = 1 / (k_on + k_off_max)
    
    where:
    - k_on = k_bind * p_conc
    - k_off_max = k_unbind * exp(-2 * beta * |J|)  [both neighbors bound]
    
    Args:
        prot: protamines instance with binding parameters
        cooperativity_factor: Typically 1.0 (use beta=1/kT if needed)
        beta: Inverse temperature (1/kT), default 1.0 if kT=1
        
    Returns:
        tau_prot: Fast timescale in seconds
        
    Notes:
        - If cooperativity J > 0, binding is stabilized and unbinding is slower
        - Maximum rate occurs with both neighbors bound (s_l=1, s_r=1)
        - k_off_max = k_unbind * exp(-2*beta*J) for J > 0
    """
    k_on = prot.k_bind * prot.P_free
    
    # Maximum unbinding rate (both neighbors bound)
    J = prot.cooperativity
    k_off_max = prot.k_unbind * math.exp(-2.0 * beta * abs(J))
    
    # Fast timescale
    total_rate = k_on + k_off_max
    
    if total_rate > 0:
        tau_prot = 1.0 / total_rate
    else:
        tau_prot = math.inf
    
    return tau_prot


def validate_qssa_for_nucleosome(
    nuc: Nucleosome,
    tau_prot: float,
    threshold: float = 0.1
) -> QSSAValidationResult:
    """
    Validate QSSA for a single nucleosome.
    
    Computes epsilon(n) = tau_prot / tau_slow(n) for each level n.
    QSSA is valid if max(epsilon(n)) <= threshold.
    
    Args:
        nuc: Nucleosome instance
        tau_prot: Protamine fast timescale (from compute_protamine_fast_timescale)
        threshold: QSSA validity threshold (default 0.1)
        
    Returns:
        QSSAValidationResult with validation details
        
    Notes:
        - Typical threshold is 0.1 (timescale separation of 10x)
        - If eps_max > threshold, full protamine-resolved simulation needed
        - If eps_max <= threshold, can use effective nucleosome-only model
    """
    # Compute nucleosome slow timescales
    tau_slow = nuc.compute_tau_slow_per_n()
    
    # Compute epsilon for each n
    epsilons = {}
    for n, ts in tau_slow.items():
        if math.isfinite(ts) and ts > 0:
            epsilons[n] = tau_prot / ts
        else:
            epsilons[n] = 0.0  # If tau_slow is infinite, epsilon is 0
    
    # Maximum epsilon
    eps_max = max(epsilons.values()) if epsilons else 0.0
    
    # QSSA valid if eps_max <= threshold
    qssa_valid = (eps_max <= threshold)
    
    return QSSAValidationResult(
        nuc_id=nuc.id,
        subid=nuc.subid,
        tau_prot=tau_prot,
        tau_slow=tau_slow,
        epsilons=epsilons,
        eps_max=eps_max,
        qssa_valid=qssa_valid,
        threshold=threshold
    )


def validate_qssa_for_system(
    nucs: Nucleosomes,
    prot: protamines,
    threshold: float = 0.1,
    beta: float = 1.0
) -> SystemQSSAResult:
    """
    Validate QSSA for entire system of nucleosomes.
    
    Args:
        nucs: Nucleosomes instance
        prot: protamines instance
        threshold: QSSA validity threshold (default 0.1)
        beta: Inverse temperature (default 1.0)
        
    Returns:
        SystemQSSAResult with system-wide validation
        
    Notes:
        - Checks QSSA validity for each nucleosome independently
        - System QSSA is valid only if ALL nucleosomes pass
        - Returns detailed per-nucleosome results for analysis
    """
    # Compute protamine fast timescale (same for all nucleosomes)
    tau_prot = compute_protamine_fast_timescale(prot, beta=beta)
    
    # Validate each nucleosome
    nuc_results = []
    num_valid = 0
    num_invalid = 0
    max_eps_overall = 0.0
    
    for nuc in nucs:
        result = validate_qssa_for_nucleosome(nuc, tau_prot, threshold)
        nuc_results.append(result)
        
        if result.qssa_valid:
            num_valid += 1
        else:
            num_invalid += 1
        
        max_eps_overall = max(max_eps_overall, result.eps_max)
    
    # System QSSA is valid only if ALL nucleosomes pass
    system_valid = (num_invalid == 0)
    
    return SystemQSSAResult(
        tau_prot=tau_prot,
        num_nucleosomes=len(nucs),
        num_valid=num_valid,
        num_invalid=num_invalid,
        fraction_valid=num_valid / len(nucs) if len(nucs) > 0 else 0.0,
        max_epsilon_overall=max_eps_overall,
        nucleosome_results=nuc_results,
        system_qssa_valid=system_valid
    )


def print_qssa_summary(result: SystemQSSAResult, verbose: bool = False):
    """
    Print a formatted summary of QSSA validation results.
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        verbose: If True, print per-nucleosome details
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
            print(f"\nNucleosome {nuc_result.nuc_id} (subid={nuc_result.subid}): {status}")
            print(f"  eps_max = {nuc_result.eps_max:.4f}")
            
            if nuc_result.eps_max > 0.05:  # Show problematic levels
                print(f"  Problematic levels (epsilon > 0.05):")
                for n, eps in nuc_result.epsilons.items():
                    if eps > 0.05:
                        ts = nuc_result.tau_slow[n]
                        print(f"    n={n}: epsilon={eps:.4f}, tau_slow={ts:.6e}")
    
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
    """
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
                
                f.write("\nTimescale ratios by level (n = total open contacts):\n")
                for n in sorted(nuc_result.epsilons.keys()):
                    eps = nuc_result.epsilons[n]
                    ts = nuc_result.tau_slow[n]
                    f.write(f"  n={n:2d}: epsilon={eps:.6f}, tau_slow={ts:.6e} s\n")
                f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"✓ QSSA report saved to {output_path}")


def save_qssa_data(result: SystemQSSAResult, output_path: Path):
    """
    Save QSSA validation data in TSV format for analysis.
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        output_path: Path to save TSV file
    """
    import pandas as pd
    
    records = []
    for nuc_result in result.nucleosome_results:
        for n in sorted(nuc_result.epsilons.keys()):
            records.append({
                'nuc_id': nuc_result.nuc_id,
                'subid': nuc_result.subid,
                'n_open': n,
                'tau_prot': nuc_result.tau_prot,
                'tau_slow': nuc_result.tau_slow[n],
                'epsilon': nuc_result.epsilons[n],
                'qssa_valid': nuc_result.qssa_valid
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"✓ QSSA data saved to {output_path}")


if __name__ == "__main__":
    """Example usage of QSSA validation."""
    print("QSSA Validator module loaded.")
    print("Use validate_qssa_for_system() to check QSSA validity for your simulations.")
