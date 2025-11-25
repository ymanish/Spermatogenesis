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
    # Slow timescales per (i,j) state:
    tau_slow: Dict[Tuple[int, int], float]
    # Timescale ratios epsilon(i,j) = tau_prot / tau_slow(i,j)
    epsilons: Dict[Tuple[int, int], float]
    eps_max: float      # Maximum epsilon across all (i,j)
    qssa_valid: bool    # True if eps_max <= threshold (global for nucleosome)
    threshold: float    # Threshold used for validation
    # Local QSSA validity per state:
    qssa_valid_per_ij: Dict[Tuple[int, int], bool]  # True if epsilon(i,j) <= threshold



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
    print("kon", k_on)
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
    Validate QSSA for a single nucleosome using per-(i,j) timescales.

    For each state (i,j) in the triangular G-matrix, we compute the total
    slow rate a_slow(i,j) (wrap + unwrap) and the corresponding slow
    timescale:

        tau_slow(i,j) = 1 / a_slow(i,j)   (seconds)

    Then:

        epsilon(i,j) = tau_prot / tau_slow(i,j)

    QSSA is considered valid if max_{(i,j)} epsilon(i,j) <= threshold.

    Args:
        nuc: Nucleosome instance
        tau_prot: Protamine fast timescale (from compute_protamine_fast_timescale)
        threshold: QSSA validity threshold (default 0.1)

    Returns:
        QSSAValidationResult with validation details
    """
    # Compute nucleosome slow timescales per (i,j)
    tau_slow_ij = nuc.compute_tau_slow_per_ij()

    # Compute epsilon(i,j) for each state
    epsilons: Dict[Tuple[int, int], float] = {}
    qssa_valid_per_ij: Dict[Tuple[int, int], bool] = {}
    
    for ij, ts in tau_slow_ij.items():
        if math.isfinite(ts) and ts > 0.0:
            epsilons[ij] = tau_prot / ts
        else:
            # If tau_slow is infinite or zero, epsilon ~ 0 (protamines effectively infinitely faster)
            epsilons[ij] = 0.0
        
        # Local QSSA validity for this state
        qssa_valid_per_ij[ij] = (epsilons[ij] <= threshold)

    # Maximum epsilon across all (i,j)
    eps_max = max(epsilons.values()) if epsilons else 0.0

    # Global QSSA valid if all relevant (i,j) satisfy epsilon(i,j) <= threshold
    qssa_valid = (eps_max <= threshold)

    return QSSAValidationResult(
        nuc_id=nuc.id,
        subid=nuc.subid,
        tau_prot=tau_prot,
        tau_slow=tau_slow_ij,
        epsilons=epsilons,
        eps_max=eps_max,
        qssa_valid=qssa_valid,
        threshold=threshold,
        qssa_valid_per_ij=qssa_valid_per_ij
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
    """
    import pandas as pd
    
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


if __name__ == "__main__":
    """Example usage of QSSA validation."""
    print("QSSA Validator module loaded.")
    print("Use validate_qssa_for_system() to check QSSA validity for your simulations.")
