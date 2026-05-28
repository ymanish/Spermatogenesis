"""
MFPT computation for Markov chains.

Column-sum CTMC convention:
    Q[to_idx, from_idx] = rate(from -> to)
    dp/dt = Q p for column probability vector p
"""
from typing import Dict, Tuple
import numpy as np

try:
    import scipy.linalg as sla
    _HAS_SCIPY = True
except ImportError:
    sla = None
    _HAS_SCIPY = False


# Output flags for compute_mfpt_from_Q_TT
FLAG_OK          = "ok"           # solve completed, result is physical
FLAG_UNDERFLOWED = "underflowed"  # solve completed but result is non-physical
                                  # (mfpt <= 0 or non-finite); the true MFPT
                                  # exceeds float64 resolution.  Returned as inf.
FLAG_SINGULAR    = "singular"     # LU factorization failed (LinAlgError) or
                                  # state with zero outgoing rate (generator bug).


def compute_mfpt_from_Q_TT(
    Q_TT: np.ndarray,
    state_index: Dict[Tuple[int, int], int],
    start_state: Tuple[int, int] = (0, 0)
) -> Tuple[float, np.ndarray, str]:
    """
    Compute MFPT from Q_TT generator matrix by solving Q_TT.T @ tau = -1.

    Args:
        Q_TT: Generator matrix (M x M) in dimensionless units (k_wrap factored out).
        state_index: Dictionary mapping (l, r) -> index.
        start_state: Initial state (default: (0, 0) = fully wrapped).

    Returns:
        mfpt: Mean first passage time from start_state, in dimensionless units.
              ``inf`` when the answer underflows float64 (see ``flag``).
        tau_vec: MFPT vector for all transient states (or all-``inf`` when bad).
        flag: One of ``FLAG_OK``, ``FLAG_UNDERFLOWED``, ``FLAG_SINGULAR``.

    Notes:
        - MFPT is in dimensionless time (units of 1/k_wrap). To convert to
          physical time: t_phys = mfpt / k_wrap_phys.
        - The solver row-equilibrates A = Q_TT.T by |diag(A)| (= total outgoing
          rate per state) and then runs LU + one step of iterative refinement.
        - When MFPT exceeds ~1/eps ~ 4.5e15 dimensionless units, the slowest
          eigenvalue of A drops below the float64 noise floor on the matrix
          entries themselves; the solve produces non-physical values
          (negative or non-finite). Those are detected here, replaced by
          ``inf``, and tagged with ``FLAG_UNDERFLOWED`` so downstream code
          can distinguish numerical underflow from a true disconnection.
    """
    M = Q_TT.shape[0]
    A = np.asarray(Q_TT.T, dtype=float)
    b = -np.ones(M)

    # Row equilibration: rescale each row by its diagonal magnitude
    # (= total outgoing rate of state j).  Solution tau is unchanged.
    d_row = np.abs(np.diag(A))
    if not (d_row > 0).all():
        # A transient state with zero outgoing rate is a generator bug,
        # not a solver problem.  Refuse rather than silently divide by zero.
        return np.inf, np.full(M, np.inf), FLAG_SINGULAR

    A_s = A / d_row[:, None]
    b_s = b / d_row

    try:
        if _HAS_SCIPY:
            lu, piv = sla.lu_factor(A_s)
            tau_vec = sla.lu_solve((lu, piv), b_s)
            # One step of iterative refinement on the equilibrated system
            r = b_s - A_s @ tau_vec
            tau_vec = tau_vec + sla.lu_solve((lu, piv), r)
        else:
            tau_vec = np.linalg.solve(A_s, b_s)
    except np.linalg.LinAlgError:
        return np.inf, np.full(M, np.inf), FLAG_SINGULAR

    start_idx = state_index[start_state]
    mfpt = float(tau_vec[start_idx])

    # Option A: detect float64 underflow on the slowest mode and floor to inf.
    # A physical MFPT is strictly positive and finite.  Anything else here
    # means the matrix's smallest eigenvalue dropped below the noise floor.
    if (not np.isfinite(mfpt)) or mfpt <= 0:
        return np.inf, np.full(M, np.inf), FLAG_UNDERFLOWED

    return mfpt, tau_vec, FLAG_OK
