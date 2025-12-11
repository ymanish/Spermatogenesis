"""
Generator matrix construction for Markov chain in (l,r) space with protamine effects.
"""
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.sparse import lil_matrix, csr_matrix
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    lil_matrix = None
    csr_matrix = None


def build_full_Q_from_nucleosome(
    nucleosome,
    k_wrap: Optional[float] = None,
    protamine_params: Optional[Dict[str, float]] = None,
    kT: Optional[float] = None,
    binding_sites: Optional[int] = None,
    sparse: bool = False, 
    dimensionless: bool = True
) -> Tuple[np.ndarray,
           np.ndarray,
           np.ndarray,
           List[Tuple[int, int]],
           Dict[Tuple[int, int], int],
           int]:
    """
    Build the FULL generator Q for a nucleosome with protamine effects (fast limit).

    Rate rules for the *coarse-grained* process (fast protamines):
        - Opening (unwrapping) uses only bare nucleosome free-energy differences:
              k_open = k_wrap * exp(-(F_new - F_old) / kT)
              
        - Closing (rewrapping) is *gated* by protamine occupancy:
              k_close_eff(l,r) = k_wrap * P_free(n_open; beta_mu, beta_J)
              
          where P_free is the equilibrium probability that the boundary site is free.

    Args:
        nucleosome: Nucleosome instance with G_mat attribute
        k_wrap: Wrapping rate (default: nucleosome.k_wrap)
        protamine_params: Dictionary with:
            - 'k_bind': binding rate
            - 'k_unbind': unbinding rate
            - 'p_conc': protamine concentration
            - 'cooperativity': cooperativity parameter
        kT: Thermal energy (default: nucleosome.kT)
        binding_sites: Number of binding sites (default: nucleosome.binding_sites)
        sparse: Use sparse matrix representation
        dimensionless: If True, return Q matrices in dimensionless units (1/k_wrap factored out)

    Returns:
        Q_full: (M+1) x (M+1) full generator matrix
        Q_TT: M x M transient block
        Q_AT: 1 x M absorbing block
        states: List of (l,r) transient states
        state_index: Dict mapping (l,r) -> index
        abs_index: Index of absorbing state
    """
    from src.core.ising_model import p_free
    
    # Extract parameters
    G_mat = nucleosome.G_mat
    k_close_bare = k_wrap if k_wrap is not None else nucleosome.k_wrap
    kT_val = kT if kT is not None else nucleosome.kT
    N = binding_sites if binding_sites is not None else nucleosome.binding_sites

    # Protamine parameters in beta units
    betamu = np.log(protamine_params['p_conc'] * protamine_params['k_bind'] / protamine_params['k_unbind'])
    betaJ = protamine_params['cooperativity'] / kT_val

    # 1. Enumerate transient states: l + r < N
    states: List[Tuple[int, int]] = []
    for l in range(N):
        for r in range(N - l):
            states.append((l, r))

    M = len(states)
    state_index: Dict[Tuple[int, int], int] = {
        (l, r): idx for idx, (l, r) in enumerate(states)
    }

    # 2. Absorbing state index
    abs_index = M
    dim_full = M + 1

    # 3. Helper for bare nucleosome free energy
    def F_nuc(l: int, r: int) -> float:
        """Bare nucleosome free energy for state (l, r)."""
        if l < 0 or r < 0 or l >= N or r >= N:
            return 0.0
        if l + r >= N:
            return 0.0

        i = l
        j = (N - 1) - r
        if 0 <= i < N and 0 <= j < N and i <= j:
            return G_mat[i, j]
        else:
            return 0.0

    # 4. Initialize generator
    use_sparse = sparse and _HAS_SCIPY
    if use_sparse:
        Q_full = lil_matrix((dim_full, dim_full), dtype=float)
    else:
        Q_full = np.zeros((dim_full, dim_full), dtype=float)

    # 5. Fill columns for transient states
    for j, (l, r) in enumerate(states):
        #### each loop fills column j meaning, all the outgoing transition rates from state j .
        total_out = 0.0
        F_curr = F_nuc(l, r)

        # ---- CLOSE LEFT: (l, r) -> (l-1, r) ----
        if l > 0:
            l2, r2 = l - 1, r
            p_free_left = p_free(l, betamu, betaJ)
            rate = k_close_bare * p_free_left
            if rate > 0.0:
                i = state_index[(l2, r2)]
                Q_full[i, j] += rate
                total_out += rate

        # ---- CLOSE RIGHT: (l, r) -> (l, r-1) ----
        if r > 0:
            l2, r2 = l, r - 1
            p_free_right = p_free(r, betamu, betaJ)
            rate = k_close_bare * p_free_right
            if rate > 0.0:
                i = state_index[(l2, r2)]
                Q_full[i, j] += rate
                total_out += rate

        # ---- OPEN LEFT: (l, r) -> (l+1, r) ----
        lL, rL = l + 1, r
        if lL + rL < N:
            F_new = F_nuc(lL, rL)
            dF = F_new - F_curr
            rate = k_close_bare * np.exp(-dF / kT_val)
            if rate > 0.0:
                i = state_index[(lL, rL)]
                Q_full[i, j] += rate
                total_out += rate
        elif lL + rL >= N:
            F_new = 0.0
            dF = F_new - F_curr
            rate = k_close_bare * np.exp(-dF / kT_val)
            if rate > 0.0:
                Q_full[abs_index, j] += rate
                total_out += rate

        # ---- OPEN RIGHT: (l, r) -> (l, r+1) ----
        lR, rR = l, r + 1
        if lR + rR < N:
            F_new = F_nuc(lR, rR)
            dF = F_new - F_curr
            rate = k_close_bare * np.exp(-dF / kT_val)
            if rate > 0.0:
                i = state_index[(lR, rR)]
                Q_full[i, j] += rate
                total_out += rate
        elif lR + rR >= N:
            F_new = 0.0
            dF = F_new - F_curr
            rate = k_close_bare * np.exp(-dF / kT_val)
            if rate > 0.0:
                Q_full[abs_index, j] += rate
                total_out += rate

        # Diagonal
        Q_full[j, j] -= total_out

    # 6. Convert to CSR if sparse
    if use_sparse:
        Q_full = Q_full.tocsr()

    # 7. Extract blocks
    Q_TT = Q_full[0:M, 0:M]
    Q_AT = Q_full[abs_index:abs_index+1, 0:M]

    if dimensionless:
        Q_full *= (1/k_wrap)
        Q_TT *= (1/k_wrap)
        Q_AT *= (1/k_wrap)
        return Q_full, Q_TT, Q_AT, states, state_index, abs_index

    return Q_full, Q_TT, Q_AT, states, state_index, abs_index
