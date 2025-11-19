"""
Core RMST Computation Functions
================================

This module contains the fundamental mathematical functions for RMST
(Restricted Mean Survival Time) calculation.

Functions are pure (no side effects) and independently testable.

Author: MY
Date: 2025-11-14
"""

import numpy as np
from typing import Dict


def compute_rmst_from_survival(
    survival_curve: np.ndarray,
    tau_grid: np.ndarray
) -> float:
    """
    Compute RMST by integrating survival curve using trapezoidal rule.
    
    RMST = ∫[0,τ_max] S(τ) dτ ≈ Σ (S_j + S_{j+1})/2 * Δτ_j
    
    Args:
        survival_curve: Array of survival probabilities S(τ) at each time point
        tau_grid: Array of time points τ
    
    Returns:
        RMST value (area under survival curve)
    
    Raises:
        ValueError: If survival_curve and tau_grid have different lengths
    
    Examples:
        >>> survival = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        >>> tau_grid = np.array([0, 2, 4, 6, 8, 10])
        >>> rmst = compute_rmst_from_survival(survival, tau_grid)
        >>> print(f"RMST: {rmst:.2f}")
        RMST: 5.00
    
    Notes:
        - Uses numpy.trapz for numerical integration
        - Trapezoidal rule has O(Δτ²) error
        - With tau_steps=1000, integration error ~0.001% of RMST
    """
    if len(survival_curve) != len(tau_grid):
        raise ValueError(
            f"Survival curve (length {len(survival_curve)}) and tau grid "
            f"(length {len(tau_grid)}) must have same length"
        )
    
    # Trapezoidal integration
    rmst = np.trapz(survival_curve, tau_grid)
    
    return rmst


def extract_survival_curve_from_timeseries(
    timeseries: Dict,
    tau_grid: np.ndarray,
    n_binding_sites: int = 14
) -> np.ndarray:
    """
    Extract survival curve from simulation timeseries data.
    
    Interpolates contact size (CS) onto regular time grid and normalizes
    to get survival function S(τ) = CS(τ) / CS_initial.
    
    Args:
        timeseries: Dict with keys:
            - 'tau': Array of time points from simulation
            - 'cs': Array of contact sizes at each time point
            - 'bprot': Array of bound protamine counts (optional)
            - 'detached': Array of detached nucleosome flags (optional)
        tau_grid: Target time grid for interpolation
        n_binding_sites: Maximum number of contacts (default: 14 for nucleosome)
    
    Returns:
        Survival curve S(τ) = CS(τ) / CS_initial, same length as tau_grid
    
    Examples:
        >>> timeseries = {
        ...     'tau': np.array([0, 5, 10, 15, 20]),
        ...     'cs': np.array([14, 12, 10, 6, 0])
        ... }
        >>> tau_grid = np.linspace(0, 20, 101)
        >>> survival = extract_survival_curve_from_timeseries(
        ...     timeseries, tau_grid, n_binding_sites=14
        ... )
        >>> print(f"Survival at t=0: {survival[0]:.2f}")
        Survival at t=0: 1.00
        >>> print(f"Survival at t=20: {survival[-1]:.2f}")
        Survival at t=20: 0.00
    
    Notes:
        - Uses linear interpolation between simulation time points
        - Left extrapolation: uses initial CS
        - Right extrapolation: uses final CS
        - Assumes CS decreases monotonically (typical for detachment)
    """
    sim_tau = np.array(timeseries['tau'])
    sim_cs = np.array(timeseries['cs'])
    
    # Initial CS (should be n_binding_sites for fully wrapped nucleosome)
    initial_cs = sim_cs[0] if len(sim_cs) > 0 else n_binding_sites
    
    # Interpolate CS onto tau_grid
    cs_interp = np.interp(
        tau_grid, 
        sim_tau, 
        sim_cs, 
        left=initial_cs,      # Extrapolate left with initial value
        right=sim_cs[-1]      # Extrapolate right with final value
    )
    
    # Normalize to survival: S(τ) = CS(τ) / CS_initial
    survival = cs_interp / initial_cs
    
    return survival
