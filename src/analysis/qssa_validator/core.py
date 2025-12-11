"""
Core QSSA Timescale Computation Functions
==========================================

This module contains the fundamental mathematical functions for computing
timescales in the QSSA (Quasi-Steady-State Approximation) analysis:

1. tau_prot: Fast timescale for protamine binding/unbinding
2. tau_slow(i,j): Slow timescale for nucleosome wrapping/unwrapping

Functions are pure (no side effects) and independently testable.

Author: MY
Date: 2025-11-27
"""

import math
from typing import Dict, Tuple

from src.core.nucleosomes import Nucleosome
from src.core.protamine import protamines


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
        
    Examples:
        >>> from src.core.protamine import protamines
        >>> prot = protamines(P_conc=100.0, cooperativity=0.0, 
        ...                   k_unbind=100.0, k_bind=1.0, binding_sites=14)
        >>> tau_prot = compute_protamine_fast_timescale(prot)
        >>> print(f"tau_prot = {tau_prot:.6e} seconds")
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


def compute_nucleosome_slow_timescale_per_ij(
    nuc: Nucleosome
) -> Dict[Tuple[int, int], float]:
    """
    Compute slow timescales for each (i,j) state in nucleosome.
    
    For each state (i,j) in the triangular G-matrix, compute the total
    slow rate a_slow(i,j) (wrap + unwrap) and the corresponding slow
    timescale:
    
        tau_slow(i,j) = 1 / a_slow(i,j)   (seconds)
    
    This is a thin wrapper around the Nucleosome method for consistency.
    
    Args:
        nuc: Nucleosome instance
        
    Returns:
        Dictionary mapping (i,j) -> tau_slow(i,j) in seconds
        
    Notes:
        - Calls nuc.compute_tau_slow_per_ij() internally
        - Returns inf for states where a_slow(i,j) = 0 (no wrapping/unwrapping)
        
    Examples:
        >>> tau_slow_ij = compute_nucleosome_slow_timescale_per_ij(nuc)
        >>> print(f"State (0,13): tau_slow = {tau_slow_ij[(0,13)]:.6e} seconds")
    """
    return nuc.compute_tau_slow_per_ij()


def compute_timescale_ratio(
    tau_prot: float,
    tau_slow: float
) -> float:
    """
    Compute the timescale ratio epsilon = tau_prot / tau_slow.
    
    This is the fundamental quantity for QSSA validation. QSSA is valid when
    epsilon << 1 (typically epsilon <= 0.1).
    
    Args:
        tau_prot: Protamine fast timescale (seconds)
        tau_slow: Nucleosome slow timescale (seconds)
        
    Returns:
        epsilon: Timescale ratio
        
    Notes:
        - Returns 0.0 if tau_slow is infinite (protamines infinitely faster)
        - Returns inf if tau_slow is 0.0 (invalid state)
        
    Examples:
        >>> epsilon = compute_timescale_ratio(tau_prot=1e-3, tau_slow=1e-1)
        >>> print(f"epsilon = {epsilon:.4f}")
        epsilon = 0.0100
    """
    if math.isfinite(tau_slow) and tau_slow > 0.0:
        return tau_prot / tau_slow
    elif math.isinf(tau_slow):
        # Protamines infinitely faster than nucleosome
        return 0.0
    else:
        # Invalid: tau_slow = 0 or negative
        return math.inf
