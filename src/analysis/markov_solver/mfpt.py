"""
MFPT computation for Markov chains.
"""
from typing import Dict, Tuple
import numpy as np

try:
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def compute_mfpt_from_Q_TT(
    Q_TT: np.ndarray,
    state_index: Dict[Tuple[int, int], int],
    start_state: Tuple[int, int] = (0, 0)
) -> Tuple[float, np.ndarray]:
    """
    Compute MFPT from Q_TT generator matrix by solving Q_TT.T @ tau = -1.
    
    Args:
        Q_TT: Generator matrix (M x M) it has dimensionless units (k_wrap factored out)
        state_index: Dictionary mapping (l,r) -> index
        start_state: Initial state (default: (0,0) = fully wrapped)
        
    Returns:
        mfpt: Mean first passage time from start_state
        tau_vec: MFPT vector for all transient states
        
    Notes:
        - MFPT is in dimensionless time (units of 1/k_wrap)
        - To convert to physical time: t_phys = mfpt / k_wrap_phys
    """
    M = Q_TT.shape[0]
    ones = np.ones(M)
    
    # Solve Q_TT.T @ tau = -1
    tau_vec = np.linalg.solve(Q_TT.T, -ones)
    
    # Extract MFPT for start_state
    start_idx = state_index[start_state]
    mfpt = float(tau_vec[start_idx])
    
    return mfpt, tau_vec

