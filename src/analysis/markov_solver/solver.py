"""
High-level solver interface.
"""
from typing import Dict, Tuple, Optional
import numpy as np

from .generator import build_full_Q_from_nucleosome
from .mfpt import compute_mfpt_from_Q_TT
from .survival import compute_survival


def solve_Q_TT_complete(
    nucleosome,
    start_state: Tuple[int, int] = (0, 0),
    tau_max: float = 100.0,
    n_points: int = 500,
    method: str = 'expm',
    sparse: bool = False,
    k_wrap: Optional[float] = None,
    protamine_params: Optional[Dict[str, float]] = None, 
    dimensionless: bool = True
) -> Dict:
    """
    Complete Q_TT analysis: build matrix, compute MFPT and survival function.
    
    High-level convenience function that:
    1. Builds Q_TT matrix from nucleosome with protamine effects
    2. Computes MFPT by solving Q_TT.T @ tau = -1
    3. Computes survival function S(t)
    
    Args:
        nucleosome: Nucleosome instance with G_mat attribute
        start_state: Initial state (default: (0,0) = fully wrapped)
        tau_max: Maximum time for survival curve (dimensionless)
        n_points: Number of time points
        method: Survival computation method ('expm' or 'ode')
        sparse: Use sparse matrix representation
        k_wrap: Override nucleosome k_wrap
        protamine_params: Dictionary with protamine parameters:
            - 'k_bind': binding rate
            - 'k_unbind': unbinding rate
            - 'p_conc': protamine concentration
            - 'cooperativity': cooperativity parameter
        dimensionless: If True, work in dimensionless units (1/k_wrap factored out) in the Q matrices
        
    Returns:
        results: Dictionary containing:
            - 'state_index': Dictionary (l,r) -> index
            - 'mfpt': Mean first passage time (dimensionless)
            - 'mfpt_vector': MFPT from all states
            - 't_grid': Time grid
            - 'survival': Survival function S(t)
            - 'k_wrap': Wrapping rate used
            - 'protamine_params': Protamine parameters used
    Examples:
        >>> from src.analysis.markov_solver import load_nucleosomes_from_file
        >>> nucs = load_nucleosomes_from_file("data.tsv", max_nucs=1)
        >>> prot_params = {
        ...     'k_bind': 1.0,
        ...     'k_unbind': 100.0,
        ...     'p_conc': 100.0,
        ...     'cooperativity': 4.5
        ... }
        >>> results = solve_Q_TT_complete(nucs[0], protamine_params=prot_params)
        >>> print(f"MFPT = {results['mfpt']:.4f} (dimensionless)")
    """
    # Build Q_TT matrix
    Q_full, Q_TT, _, states, state_index, abs_index = build_full_Q_from_nucleosome(
        nucleosome, k_wrap=k_wrap, sparse=sparse, protamine_params=protamine_params, dimensionless=dimensionless
    )
    
    # Get parameters
    k_wrap_val = k_wrap if k_wrap is not None else nucleosome.k_wrap
    
    # Compute MFPT
    mfpt, mfpt_vector, mfpt_flag = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state)

    # Compute survival function
    tau_grid = np.linspace(0, tau_max, n_points)
    survival = compute_survival(Q_TT, state_index, start_state, tau_grid, method)

    # Package results
    results = {
        'Q_TT': Q_TT,
        'mfpt': mfpt,
        'mfpt_vector': mfpt_vector,
        'mfpt_flag': mfpt_flag,
        'tau_grid': tau_grid,
        'survival': survival,
        'k_wrap': k_wrap_val,
        'start_state': start_state,
        'protamine_params': protamine_params,
    }
    
    return results
