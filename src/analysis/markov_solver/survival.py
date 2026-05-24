"""
Survival function computation for Markov chains.

Column-sum CTMC convention:
    Q[to_idx, from_idx] = rate(from -> to)
    dp/dt = Q p for column probability vector p
"""
from typing import Dict, List, Tuple
import numpy as np

try:
    from scipy.linalg import expm
    _HAS_SCIPY = True
except ImportError:
    expm = None
    _HAS_SCIPY = False


def compute_survival(
    Q_trans: np.ndarray,
    index_map: Dict[Tuple[int, int], int],
    start_state: Tuple[int, int],
    tau_grid: np.ndarray,
    method: str = 'expm',
    return_states: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Compute survival function S(t) = P(T_absorb > t) from transient generator matrix.
    
    Args:
        Q_trans: Transient generator matrix (M x M), can be in dimensionless units or physical units, depends on construction of Q_trans
        index_map: Dictionary mapping (l,r) -> index containing only transient states
        start_state: Initial state (l,r)
        tau_grid: Array of dimensionless times τ = k_wrap * t_phys
        method: Computation method ('expm' or 'ode')
        return_states: If True, also return state probabilities P(t)
        
    Returns:
        S: Array of survival probabilities at each time in tau_grid
        P_states: (optional) Array of state probabilities (shape = (len(tau_grid), M))

    Notes:
        - Q_trans should be constructed with k_wrap as the base rate, making it
          effectively dimensionless when evaluated at dimensionless times τ.
        - 'expm': Uses scipy.linalg.expm for matrix exponential (accurate but slower)
        - 'ode': Uses scipy.integrate.solve_ivp ODE solver (faster for large systems)
        - S(t) = sum of probabilities in all transient states at time t
        
    Examples:
        >>> # Simple survival function
        >>> S = compute_survival(Q_TT, idx_map, (0,0), t_grid)
        
        >>> # With state probabilities
        >>> S, P = compute_survival(Q_TT, idx_map, (0,0), t_grid, return_states=True)
        
        >>> # Using ODE solver for large systems
        >>> S = compute_survival(Q_TT, idx_map, (0,0), t_grid, method='ode')
    """
    M = Q_trans.shape[0]
    
    # Initial condition: one-hot at start_state
    p0 = np.zeros(M)
    p0[index_map[start_state]] = 1.0
    
    # Prepare output arrays
    S = np.empty(len(tau_grid), dtype=float)

    if return_states:
        P_states = np.empty((len(tau_grid), M), dtype=float)

    if method == 'expm':
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for survival method='expm'")
        # Matrix exponential method (accurate but slower)
        # Convert sparse to dense for expm if needed
        if hasattr(Q_trans, 'toarray'):
            Q_dense = Q_trans.toarray()
        else:
            Q_dense = Q_trans

        for k, t in enumerate(tau_grid):
            p_t = expm(Q_dense * t) @ p0
            S[k] = p_t.sum()
            
            if return_states:
                P_states[k, :] = p_t
    
    elif method == 'ode':
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for survival method='ode'")
        # ODE solver method (faster for large systems)
        from scipy.integrate import solve_ivp
        
        def dpdt(t, p):
            if hasattr(Q_trans, 'dot'):
                return Q_trans.dot(p)
            else:
                return Q_trans @ p
        
        sol = solve_ivp(
            dpdt,
            (tau_grid[0], tau_grid[-1]),
            p0,
            t_eval=tau_grid,
            method='LSODA',
            rtol=1e-6,
            atol=1e-8
        )

        for k in range(len(tau_grid)):
            S[k] = sol.y[:, k].sum()
            
            if return_states:
                P_states[k, :] = sol.y[:, k]
    
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'expm' or 'ode'.")
    
    if return_states:
        return S, P_states
    else:
        return S

# ============================================================================
