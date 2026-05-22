"""
Generator matrix construction for Markov chain in (l,r) space with protamine effects.

Column-sum CTMC convention:
    Q[to_idx, from_idx] = rate(from -> to)
    dp/dt = Q p for column probability vector p
"""
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.sparse import issparse, lil_matrix, csr_matrix
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    issparse = None
    lil_matrix = None
    csr_matrix = None


def matrix_density(Q) -> float:
    if _HAS_SCIPY and issparse(Q):
        nnz = Q.nnz
        total = Q.shape[0] * Q.shape[1]
    else:
        nnz = np.count_nonzero(Q)
        total = Q.size
    return nnz / total


def add_transition(Q, source_idx: int, dest_idx: int, rate: float) -> float:
    """Add one column-sum generator transition and return the added rate."""
    Q[dest_idx, source_idx] += rate
    return rate


def _as_1d_array(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()


def _assert_generator_invariants(Q_full, abs_index: int) -> None:
    column_sums = _as_1d_array(Q_full.sum(axis=0))
    absorbing_column = _as_1d_array(Q_full[:, abs_index])
    diagonal = _as_1d_array(Q_full.diagonal())

    assert np.allclose(column_sums, 0.0)
    assert np.allclose(absorbing_column, 0.0)
    assert np.all(diagonal <= 0.0)


def build_full_Q_from_nucleosome(
    nucleosome,
    k_wrap: Optional[float] = None,
    protamine_params: Optional[Dict[str, float]] = None,
    kT: Optional[float] = None,
    binding_sites: Optional[int] = None,
    sparse: bool = False,
    dimensionless: bool = True,
    eads_delta: float = 0.0,
    eads_weight_mode: str = "none",
    eads_apply: bool = False,
    tnp2_config=None,
) -> Tuple[np.ndarray,
           np.ndarray,
           np.ndarray,
           List[Tuple[int, int]],
           Dict[Tuple[int, int], int],
           int]:
    """
    Build the FULL generator Q for a nucleosome with protamine effects (fast limit).

    Rate rules for the *coarse-grained* process (fast protamines):
        - Opening (unwrapping) uses bare nucleosome free-energy differences with
          an optional weighted adsorption-energy correction:
              k_open = k_wrap * exp(-((F_new - F_old) - eads_delta * w(u)) / kT)
              
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
        eads_delta: Opening-energy reduction magnitude in kBT units.
        eads_weight_mode: Structural weight mode: "none", "uniform", "outer8",
            or "inner6".
        eads_apply: Whether to apply the Eads correction.

    Returns:
        Q_full: (M+1) x (M+1) full generator matrix
        Q_TT: M x M transient block
        Q_AT: 1 x M absorbing block
        states: List of (l,r) transient states
        state_index: Dict mapping (l,r) -> index
        abs_index: Index of absorbing state
    """
    from src.core.ising_model import p_free, p_free_site_dependent
    
    # Extract parameters
    G_mat = nucleosome.G_mat
    k_close_bare = k_wrap if k_wrap is not None else nucleosome.k_wrap
    kT_val = kT if kT is not None else nucleosome.kT
    N = binding_sites if binding_sites is not None else nucleosome.binding_sites
    eads_delta = float(eads_delta)
    eads_weight_mode = str(eads_weight_mode)
    if protamine_params is None:
        protamine_params = {
            'k_bind': 1.0,
            'k_unbind': 89.7,
            'p_conc': 0.0,
            'cooperativity': 0.0,
        }

    # Protamine parameters in beta units
    if protamine_params['p_conc'] <= 0.0:
        # print("  Protamine concentration <= 0. Setting betamu = -inf (no protamines).")
        betamu = -np.inf
        betaJ = 0.0
    else:
        betamu = np.log(protamine_params['p_conc'] * protamine_params['k_bind'] / protamine_params['k_unbind'])
        betaJ = protamine_params['cooperativity'] / kT_val

    # TNP2 modifies only protamine cooperativity in the closing gate.
    j_eff_left = None
    j_eff_right = None
    seq = getattr(nucleosome, 'sequence', None)
    if (
        tnp2_config is not None
        and getattr(tnp2_config, 'enabled', False)
        and seq
        and N == 14
        and np.isfinite(betamu)
        and betaJ != 0.0
    ):
        from .tnp2 import compute_oriented_jeff_profiles
        profiles = compute_oriented_jeff_profiles(
            seq_147=seq,
            eps_cpg=tnp2_config.eps_cpg,
            mu_t0=tnp2_config.mu_t0,
            j_bare=betaJ,
            beta=1.0,
            n_sites=N,
        )
        j_eff_left = profiles["left"][0]
        j_eff_right = profiles["right"][0]

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

    def eads_weight(contact: int) -> float:
        """Return w(u) for one-indexed contact u."""
        if not eads_apply or eads_delta == 0.0:
            return 0.0
        if eads_weight_mode == "none":
            return 0.0
        if eads_weight_mode == "uniform":
            return 1.0
        if eads_weight_mode == "outer8":
            return 1.0 if 1 <= contact <= 4 or N - 3 <= contact <= N else 0.0
        if eads_weight_mode == "inner6":
            return 1.0 if 5 <= contact <= N - 4 else 0.0
        raise ValueError(
            "eads_weight_mode must be one of 'none', 'uniform', 'outer8', or 'inner6', "
            f"got {eads_weight_mode!r}"
        )

    def corrected_opening_dF(F_new: float, F_curr: float, contact: int) -> float:
        return F_new - F_curr - eads_delta * eads_weight(contact)

    # 4. Initialize generator
    use_sparse = sparse and _HAS_SCIPY
    if use_sparse:
        Q_full = lil_matrix((dim_full, dim_full), dtype=float)
    else:
        Q_full = np.zeros((dim_full, dim_full), dtype=float)

    # 5. Fill columns for transient states
    for source_idx, (l, r) in enumerate(states):
        total_out = 0.0
        F_curr = F_nuc(l, r)

        # ---- CLOSE LEFT: (l, r) -> (l-1, r) ----
        if l > 0:
            l2, r2 = l - 1, r
            if j_eff_left is None:
                p_free_left = p_free(l, betamu, betaJ)
            else:
                p_free_left = p_free_site_dependent(l, betamu, j_eff_left[0:l - 1])
            rate = k_close_bare * p_free_left
            # rate = max(rate, 1e-300)
            if rate > 0.0:
                dest_idx = state_index[(l2, r2)]
                total_out += add_transition(Q_full, source_idx, dest_idx, rate)

        # ---- CLOSE RIGHT: (l, r) -> (l, r-1) ----
        if r > 0:
            l2, r2 = l, r - 1
            if j_eff_right is None:
                p_free_right = p_free(r, betamu, betaJ)
            else:
                p_free_right = p_free_site_dependent(r, betamu, j_eff_right[0:r - 1])
            rate = k_close_bare * p_free_right
            # rate = max(rate, 1e-300)
            if rate > 0.0:
                dest_idx = state_index[(l2, r2)]
                total_out += add_transition(Q_full, source_idx, dest_idx, rate)

        # ---- OPEN LEFT: (l, r) -> (l+1, r) ----
        lL, rL = l + 1, r
        left_contact = l + 1
        if lL + rL < N:
            F_new = F_nuc(lL, rL)
            dF = corrected_opening_dF(F_new, F_curr, left_contact)
            rate = k_close_bare * np.exp(-dF / kT_val)
            # rate = max(rate, 1e-300)
            if rate > 0.0:
                dest_idx = state_index[(lL, rL)]
                total_out += add_transition(Q_full, source_idx, dest_idx, rate)
        elif lL + rL >= N:
            F_new = 0.0
            dF = corrected_opening_dF(F_new, F_curr, left_contact)
            rate = k_close_bare * np.exp(-dF / kT_val)
            # rate = max(rate, 1e-300)
            if rate > 0.0:
                total_out += add_transition(Q_full, source_idx, abs_index, rate)

        # ---- OPEN RIGHT: (l, r) -> (l, r+1) ----
        lR, rR = l, r + 1
        right_contact = N - r
        if lR + rR < N:
            F_new = F_nuc(lR, rR)
            dF = corrected_opening_dF(F_new, F_curr, right_contact)
            rate = k_close_bare * np.exp(-dF / kT_val)
            # rate = max(rate, 1e-300)
            if rate > 0.0:
                dest_idx = state_index[(lR, rR)]
                total_out += add_transition(Q_full, source_idx, dest_idx, rate)

        elif lR + rR >= N:
            F_new = 0.0
            dF = corrected_opening_dF(F_new, F_curr, right_contact)
            rate = k_close_bare * np.exp(-dF / kT_val)
            # rate = max(rate, 1e-300)
            if rate > 0.0:
                total_out += add_transition(Q_full, source_idx, abs_index, rate)

        # Diagonal
        Q_full[source_idx, source_idx] -= total_out

    # 6. Convert to CSR if sparse
    if use_sparse:
        Q_full = Q_full.tocsr()

    scale = 1.0
    if dimensionless:
        scale = 1.0 / k_close_bare   # <-- use the actual k used in rates

    # Apply scaling once
    Q_full = Q_full * scale
    _assert_generator_invariants(Q_full, abs_index)

    # Re-slice AFTER scaling (so Q_TT/Q_AT are consistent and not double-scaled)
    Q_TT = Q_full[0:M, 0:M]
    Q_AT = Q_full[abs_index:abs_index+1, 0:M]

    return Q_full, Q_TT, Q_AT, states, state_index, abs_index
