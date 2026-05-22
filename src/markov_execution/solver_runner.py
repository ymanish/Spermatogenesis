"""
Solver Runner Module
====================

Functions for solving individual nucleosome Markov chains.
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
    eads_delta: float = 0.0,
    eads_weight_mode: str = "none",
    eads_apply: bool = False,
    tnp2_config=None,
    compute_states: bool = False,
    start_state: Tuple[int, int] = (0, 0)
) -> dict:
    """
    Solve Markov chain for a single nucleosome.

    Args:
        nuc:               Nucleosome instance with G_mat attribute
        tau_grid:          Time grid for evaluation (dimensionless τ)
        k_wrap:            Wrapping rate constant (s^-1)
        protamine_params:  Dict with keys k_bind, k_unbind, p_conc, cooperativity
        kT:                Thermal energy (k_B T)
        binding_sites:     Number of binding sites (default: from nucleosome)
        method:            Solver method ('expm' or 'ode')
        sparse:            Whether to use sparse matrices
        dimensionless:     Whether Q is in dimensionless units
        eads_delta:        Opening-energy reduction magnitude (k_B T)
        eads_weight_mode:  Structural weight mode for the correction
        eads_apply:        Whether to apply the Eads correction
        compute_states:    Whether to compute full state probabilities P(t)
        start_state:       Initial (l, r) state

    Returns:
        Dict with: id, subid, survival, mfpt, mfpt_vec, tau_grid,
                   and optionally state_probs + states
    """
    Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
        nuc,
        k_wrap=k_wrap,
        protamine_params=protamine_params,
        kT=kT,
        binding_sites=binding_sites,
        sparse=sparse,
        dimensionless=dimensionless,
        eads_delta=eads_delta,
        eads_weight_mode=eads_weight_mode,
        eads_apply=eads_apply,
        tnp2_config=tnp2_config,
    )

    if compute_states:
        S, P_states = compute_survival(
            Q_TT, state_index, start_state, tau_grid,
            method=method, return_states=True
        )
    else:
        S = compute_survival(
            Q_TT, state_index, start_state, tau_grid,
            method=method, return_states=False
        )
        P_states = None

    mfpt, tau_vec = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state)

    results = {
        'id':       nuc.id,
        'subid':    nuc.subid,
        'survival': S,
        'mfpt':     mfpt,
        'mfpt_vec': tau_vec,
        'tau_grid': tau_grid
    }

    if compute_states and P_states is not None:
        results['state_probs'] = P_states
        results['states'] = states

    return results
