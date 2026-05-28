"""
Solver Runner Module
====================

Functions for solving individual nucleosome Markov chains.

Author: MY
Date: 2025-12-11
"""

from typing import Dict, Optional, Tuple
import numpy as np
from src.core.nucleosomes import Nucleosome
from src.analysis.markov_solver import (
    build_full_Q_from_nucleosome,
    compute_survival,
    compute_mfpt_from_Q_TT
)


def solve_single_nucleosome(
    nuc: Nucleosome,
    tau_grid: np.ndarray,
    k_wrap: float,
    protamine_params: dict,
    kT: float = 1.0,
    binding_sites: Optional[int] = None,
    method: str = 'expm',
    sparse: bool = False,
    dimensionless: bool = True,
    compute_states: bool = False,
    start_state: Tuple[int, int] = (0, 0)
) -> dict:
    """
    Solve Markov chain for a single nucleosome.
    
    Args:
        nuc: Nucleosome instance with G_mat attribute
        tau_grid: Time grid for evaluation (dimensionless τ)
        k_wrap: Wrapping rate constant (s^-1)
        protamine_params: Dictionary with protamine parameters
        kT: Thermal energy (k_B T)
        binding_sites: Number of binding sites (default: from nucleosome)
        method: Solver method ('expm' or 'ode')
        sparse: Whether to use sparse matrices
        dimensionless: Whether Q is in dimensionless units
        compute_states: Whether to compute full state probabilities
        start_state: Initial state (l, r)
    
    Returns:
        Dictionary with results:
            - 'survival': Survival function S(t)
            - 'mfpt': Mean first passage time
            - 'id': Nucleosome ID
            - 'subid': Nucleosome subID
            - 'states': List of transient states (if requested)
            - 'state_probs': State probabilities P(t) (if requested)
    """
    # Build generator matrix
    Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
        nuc,
        k_wrap=k_wrap,
        protamine_params=protamine_params,
        kT=kT,
        binding_sites=binding_sites,
        sparse=sparse,
        dimensionless=dimensionless
    )
    
    # Compute survival function
    if compute_states:
        S, P_states = compute_survival(
            Q_TT,
            state_index,
            start_state,
            tau_grid,
            method=method,
            return_states=True
        )
    else:
        S = compute_survival(
            Q_TT,
            state_index,
            start_state,
            tau_grid,
            method=method,
            return_states=False
        )
        P_states = None
    
    # Compute MFPT
    mfpt, tau_vec, mfpt_flag = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state)

    # Package results
    results = {
        'id': nuc.id,
        'subid': nuc.subid,
        'survival': S,
        'mfpt': mfpt,
        'mfpt_vec': tau_vec,
        'mfpt_flag': mfpt_flag,
        'tau_grid': tau_grid
    }
    
    # Add optional outputs
    if compute_states and P_states is not None:
        results['state_probs'] = P_states
        results['states'] = states
    
    return results


# def compute_derived_quantities(results: dict) -> dict:
#     """
#     Compute derived quantities from solver results.
    
#     Args:
#         results: Dictionary from solve_single_nucleosome
    
#     Returns:
#         Dictionary with derived quantities:
#             - 'half_life': Time when S(t) = 0.5
#             - 'final_survival': S(t_max)
#             - 'mean_survival': Mean survival over time
#     """
#     S = results['survival']
#     tau_grid = results['tau_grid']
#
#     # Half-life (time when S = 0.5)
#     idx_half = np.argmin(np.abs(S - 0.5))
#     half_life = tau_grid[idx_half] if S[0] > 0.5 else np.nan
#
#     # Final survival probability
#     final_survival = S[-1]
#
#     # Mean survival (area under curve / t_max)
#     mean_survival = np.trapz(S, tau_grid) / tau_grid[-1] if len(tau_grid) > 1 else S[0]
#
#     return {
#         'half_life': half_life,
#         'final_survival': final_survival,
#         'mean_survival': mean_survival
#     }
