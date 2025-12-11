"""
QSSA Validation Logic
======================

This module contains the core validation logic for checking whether the
Quasi-Steady-State Approximation (QSSA) is valid for nucleosome-protamine
simulations.

Functions validate QSSA at two levels:
1. Single nucleosome: Check all (i,j) states
2. System: Check all nucleosomes

Author: MY
Date: 2025-11-27
"""

import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.core.nucleosomes import Nucleosome, Nucleosomes
from src.core.protamine import protamines

from .core import (
    compute_protamine_fast_timescale,
    compute_nucleosome_slow_timescale_per_ij,
    compute_timescale_ratio
)


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
        
    Examples:
        >>> from src.core.build_nucleosomes import nucleosome_generator
        >>> from src.core.protamine import protamines
        >>> 
        >>> # Load nucleosome
        >>> gen = nucleosome_generator("data.tsv", k_wrap=21.0, kT=1.0, binding_sites=14)
        >>> nuc = next(gen)
        >>> 
        >>> # Create protamine instance
        >>> prot = protamines(P_conc=100.0, cooperativity=0.0, 
        ...                   k_unbind=100.0, k_bind=1.0, binding_sites=14)
        >>> 
        >>> # Compute tau_prot
        >>> tau_prot = compute_protamine_fast_timescale(prot)
        >>> 
        >>> # Validate QSSA
        >>> result = validate_qssa_for_nucleosome(nuc, tau_prot, threshold=0.1)
        >>> print(f"QSSA valid: {result.qssa_valid}")
        >>> print(f"Max epsilon: {result.eps_max:.4f}")
    """
    # Compute nucleosome slow timescales per (i,j)
    tau_slow_ij = compute_nucleosome_slow_timescale_per_ij(nuc)

    # Compute epsilon(i,j) for each state
    epsilons: Dict[Tuple[int, int], float] = {}
    qssa_valid_per_ij: Dict[Tuple[int, int], bool] = {}
    
    for ij, ts in tau_slow_ij.items():
        epsilons[ij] = compute_timescale_ratio(tau_prot, ts)
        
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
        
    Examples:
        >>> from src.core.build_nucleosomes import build_nucleosomes_from_file
        >>> from src.core.protamine import protamines
        >>> 
        >>> # Load nucleosomes
        >>> nucs = build_nucleosomes_from_file("data.tsv", k_wrap=21.0, 
        ...                                     kT=1.0, binding_sites=14)
        >>> 
        >>> # Create protamine instance
        >>> prot = protamines(P_conc=100.0, cooperativity=0.0,
        ...                   k_unbind=100.0, k_bind=1.0, binding_sites=14)
        >>> 
        >>> # Validate system
        >>> result = validate_qssa_for_system(nucs, prot, threshold=0.1)
        >>> print(f"System QSSA valid: {result.system_qssa_valid}")
        >>> print(f"Valid nucleosomes: {result.num_valid}/{result.num_nucleosomes}")
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
