"""
Projection utilities for (l,r) -> n space.
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy.linalg import expm


def project_to_open_sites(
    transient_states: List[Tuple[int, int]],
    probs: np.ndarray,
    N_MAX: int
) -> np.ndarray:
    """
    Project probability distribution from (l,r) space to n = l+r space.
    
    Args:
        transient_states: List of (l,r) states
        probs: Array of probabilities p_{(l,r)}
        N_MAX: Maximum number of contacts
        
    Returns:
        P_n: Array of length N_MAX with P_n[n] = probability of n open sites
    """
    P_n = np.zeros(N_MAX, dtype=float)
    
    for p, (l, r) in zip(probs, transient_states):
        n = l + r
        P_n[n] += p
    
    return P_n


def compute_open_sites_evolution(
    Q_trans: np.ndarray,
    index_map: Dict[Tuple[int, int], int],
    transient_states: List[Tuple[int, int]],
    start_state: Tuple[int, int],
    t_grid: np.ndarray,
    N_MAX: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time evolution of open site distribution P_n(t).
    
    Args:
        Q_trans: Transient generator matrix
        index_map: Dictionary mapping (l,r) -> index
        transient_states: List of transient (l,r) states
        start_state: Initial state (l,r)
        t_grid: Array of dimensionless times
        N_MAX: Maximum number of contacts
        
    Returns:
        S: Survival function S(t)
        P_n_t: Array of shape (len(t_grid), N_MAX) with probabilities
    """
    from .survival import compute_survival
    
    # Get state probabilities over time
    S, P_states = compute_survival(Q_trans, index_map, transient_states,
                                               start_state, t_grid)
    
    # Project to open sites at each time
    P_n_t = np.empty((len(t_grid), N_MAX), dtype=float)
    for k in range(len(t_grid)):
        P_n_t[k, :] = project_to_open_sites(transient_states, P_states[k, :], N_MAX)
    
    return S, P_n_t
