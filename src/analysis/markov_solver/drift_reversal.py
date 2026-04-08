"""
Drift reversal analysis for nucleosome eviction kinetics.

This module implements a 1D reduced coordinate (birth-death process) analysis
to understand nucleosome eviction via nucleation theory and drift reversal.

Key concepts:
- Reduced coordinate: n = l + r (total open sites)
- Drift: v(n) = k+(n) - k-(n)
- Critical nucleus: n* where drift changes sign
- Quasi-potential: Φ(n) measuring barrier height
- MFPT from nucleation theory

References:
    Birth-death MFPT formulas: Van Kampen "Stochastic Processes in Physics"
    Nucleation theory: Kramers, Hanggi et al.
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm


def committor_birth_death(k_plus: np.ndarray, k_minus: np.ndarray) -> np.ndarray:
    """
    Exact committor q(n) for a 1D birth–death CTMC with absorbing boundaries at n=0 and n=N.

    Parameters
    ----------
    k_plus : array-like, shape (N,)
        k_plus[n] = rate for n -> n+1, for n=0..N-1.
    k_minus : array-like, shape (N+1,) or (N+1,)
        k_minus[n] = rate for n -> n-1, for n=1..N.
        k_minus[0] unused (can be 0).

    Returns
    -------
    q : np.ndarray, shape (N+1,)
        q[0]=0, q[N]=1, and q[n] is the probability to hit N before 0 starting from n.

    Notes
    -----
    - Requires k_plus[n] > 0 for n=0..N-1 and k_minus[n] > 0 for n=1..N
      for a well-posed interior problem. If some are zero, the chain may be
      partially absorbing / disconnected; q will still be computed but interpret carefully.
    """
    k_plus = np.asarray(k_plus, dtype=float)
    k_minus = np.asarray(k_minus, dtype=float)

    N = k_plus.shape[0]  # because k_plus is defined for 0..N-1
    if k_minus.shape[0] != N + 1:
        raise ValueError(f"k_minus must have length N+1={N+1}, got {k_minus.shape[0]}")

    # Interior unknowns: q[1..N-1]
    M = N - 1
    A = np.zeros((M, M), dtype=float)
    b = np.zeros(M, dtype=float)

    # For n = 1..N-1, map to row index i = n-1
    for n in range(1, N):
        i = n - 1
        kp = k_plus[n]      # n -> n+1
        km = k_minus[n]     # n -> n-1

        # Equation: kp(q_{n+1}-q_n) + km(q_{n-1}-q_n) = 0
        # => -(kp+km) q_n + kp q_{n+1} + km q_{n-1} = 0

        A[i, i] = -(kp + km)

        # q_{n-1} term
        if n - 1 >= 1:
            A[i, i - 1] = km
        else:
            # n-1 = 0 is boundary with q0=0; contributes km*q0 to RHS (but q0=0)
            pass

        # q_{n+1} term
        if n + 1 <= N - 1:
            A[i, i + 1] = kp
        else:
            # n+1 = N is boundary with qN=1; move kp*qN to RHS
            b[i] -= kp * 1.0

    q_interior = np.linalg.solve(A, b)

    q = np.zeros(N + 1, dtype=float)
    q[0] = 0.0
    q[N] = 1.0
    q[1:N] = q_interior

    # Numerical safety: clip tiny overshoots
    q = np.clip(q, 0.0, 1.0)
    return q


@dataclass
class DriftReversalResults:
    """Container for drift reversal analysis results."""
    
    # Coordinate and rates
    n_values: np.ndarray  # n = 0, 1, ..., N
    k_plus: np.ndarray    # Opening rates k+(n)
    k_minus: np.ndarray   # Closing rates k-(n)
    drift: np.ndarray     # v(n) = k+(n) - k-(n)
    
    # Critical nucleus
    n_star: Optional[int]  # Critical nucleus (first n where drift changes sign)
    n_star_refined: Optional[float]  # Interpolated crossing point
    
    # Quasi-potential and barrier
    phi: np.ndarray       # Quasi-potential Φ(n)
    delta_phi: float      # Barrier height Δφ = Φ(n*) - Φ(0)
    
    # MFPT
    mfpt_1d: float        # Exact 1D MFPT from (0) to (N)
    mfpt_nucleation: Optional[float]  # Nucleation approximation (if available)
    
    # Committor (probability to reach N before 0)
    committor: np.ndarray  # q(n): probability to detach (reach N) before fully wrapping (reach 0)
    
    # Per-shell distributions (for diagnostics)
    shell_data: Dict[int, Dict]  # n -> {states: [(l,r)], weights: [...], F_values: [...]}
    
    # Parameters
    k_wrap: float
    protamine_params: Dict[str, float]
    kT: float
    N: int  # Total binding sites


class DriftReversalAnalyzer:
    """
    Analyze nucleosome eviction via 1D reduced coordinate n = l + r.
    
    This implements the "drift reversal" framework:
    1. Coarse-grain (l,r) -> n by averaging within shells
    2. Compute effective 1D rates k+(n), k-(n)
    3. Identify critical nucleus n* where drift changes sign
    4. Calculate barrier Δφ and predict MFPT
    """
    
    def __init__(
        self,
        nucleosome,
        k_wrap: Optional[float] = None,
        protamine_params: Optional[Dict[str, float]] = None,
        kT: Optional[float] = None,
        binding_sites: Optional[int] = None,
    ):
        """
        Initialize drift reversal analyzer.
        
        Args:
            nucleosome: Nucleosome instance with G_mat attribute
            k_wrap: Wrapping rate (default: nucleosome.k_wrap)
            protamine_params: Dict with k_bind, k_unbind, p_conc, cooperativity
            kT: Thermal energy (default: nucleosome.kT)
            binding_sites: Number of binding sites (default: nucleosome.binding_sites)
        """
        self.nucleosome = nucleosome
        self.G_mat = nucleosome.G_mat
        self.k_wrap = k_wrap if k_wrap is not None else nucleosome.k_wrap
        self.kT = kT if kT is not None else nucleosome.kT
        self.N = binding_sites if binding_sites is not None else nucleosome.binding_sites
        
        # Protamine parameters
        if protamine_params is None:
            protamine_params = {
                'k_bind': 1.0,
                'k_unbind': 1.0,
                'p_conc': 0.0,
                'cooperativity': 0.0
            }
        self.protamine_params = protamine_params
        
        # Compute beta*mu and beta*J for Ising model
        if protamine_params['p_conc'] <= 0.0:
            self.betamu = -np.inf
            self.betaJ = 0.0
        else:
            self.betamu = np.log(
                protamine_params['p_conc'] 
                * protamine_params['k_bind'] 
                / protamine_params['k_unbind']
            )
            self.betaJ = protamine_params['cooperativity'] / self.kT
    
    def shell_states(self, n: int) -> List[Tuple[int, int]]:
        """
        Return all microstates (l,r) with l + r = n.
        
        Args:
            n: Total number of open sites
            
        Returns:
            List of (l, r) tuples
        """
        states = []
        for l in range(n + 1):
            r = n - l
            if l >= 0 and r >= 0 and l < self.N and r < self.N:
                states.append((l, r))
        return states
    
    def F_nuc(self, l: int, r: int) -> float:
        """
        Bare nucleosome free energy for state (l, r).
        
        Args:
            l: Left open sites
            r: Right open sites
            
        Returns:
            Free energy F(l, r)
        """
        if l < 0 or r < 0 or l >= self.N or r >= self.N:
            return 0.0
        if l + r >= self.N:
            return 0.0
        
        i = l
        j = (self.N - 1) - r
        if 0 <= i < self.N and 0 <= j < self.N and i <= j:
            return self.G_mat[i, j]
        else:
            return 0.0
    
    def pi_lr_given_n(self, states: List[Tuple[int, int]]) -> np.ndarray:
        """
        Quasi-equilibrium distribution over states with same n.
        
        π(l,r|n) ∝ exp(-F(l,r)/kT)
        
        Args:
            states: List of (l, r) tuples with same n
            
        Returns:
            Array of normalized weights
        """
        if not states:
            return np.array([])
        
        F_values = np.array([self.F_nuc(l, r) for l, r in states])
        log_weights = -F_values / self.kT
        
        # Normalize in log space for numerical stability
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()
        
        return weights
    
    def k_plus_lr(self, l: int, r: int) -> Tuple[float, float]:
        """
        Opening rates from microstate (l,r).
        
        Left opening: (l,r) -> (l+1,r)
        Right opening: (l,r) -> (l,r+1)
        
        Rate = k_wrap * exp(-ΔF/kT)
        
        Args:
            l: Left open sites
            r: Right open sites
            
        Returns:
            (k_plus_L, k_plus_R): Left and right opening rates
        """
        F_curr = self.F_nuc(l, r)
        
        # Left opening
        l_new, r_new = l + 1, r
        if l_new + r_new >= self.N:
            F_new = 0.0  # Detached state
        else:
            F_new = self.F_nuc(l_new, r_new)
        dF = F_new - F_curr
        k_plus_L = self.k_wrap * np.exp(-dF / self.kT)
        
        # Right opening
        l_new, r_new = l, r + 1
        if l_new + r_new >= self.N:
            F_new = 0.0
        else:
            F_new = self.F_nuc(l_new, r_new)
        dF = F_new - F_curr
        k_plus_R = self.k_wrap * np.exp(-dF / self.kT)
        
        return k_plus_L, k_plus_R
    
    def k_minus_lr(self, l: int, r: int) -> Tuple[float, float]:
        """
        Closing rates from microstate (l,r), gated by protamine occupancy.
        
        Left closing: (l,r) -> (l-1,r), rate = k_wrap * p_free(l)
        Right closing: (l,r) -> (l,r-1), rate = k_wrap * p_free(r)
        
        Args:
            l: Left open sites
            r: Right open sites
            
        Returns:
            (k_minus_L, k_minus_R): Left and right closing rates
        """
        from src.core.ising_model import p_free
        
        # Left closing
        if l > 0:
            p_free_left = p_free(l, self.betamu, self.betaJ)
            k_minus_L = self.k_wrap * p_free_left
        else:
            k_minus_L = 0.0
        
        # Right closing
        if r > 0:
            p_free_right = p_free(r, self.betamu, self.betaJ)
            k_minus_R = self.k_wrap * p_free_right
        else:
            k_minus_R = 0.0
        
        return k_minus_L, k_minus_R
    
    def compute_effective_rates(self, n: int) -> Tuple[float, float, Dict]:
        """
        Compute effective 1D rates k+(n) and k-(n) by averaging over shell.
        
        k+(n) = Σ π(l,r|n) * [k+_L(l,r) + k+_R(l,r)]
        k-(n) = Σ π(l,r|n) * [k-_L(l,r) + k-_R(l,r)]
        
        Args:
            n: Total open sites
            
        Returns:
            (k_plus_n, k_minus_n, shell_info): Effective rates and diagnostic info
        """
        states = self.shell_states(n)
        if not states:
            return 0.0, 0.0, {}
        
        weights = self.pi_lr_given_n(states)
        
        k_plus_n = 0.0
        k_minus_n = 0.0
        
        for (l, r), w in zip(states, weights):
            k_pL, k_pR = self.k_plus_lr(l, r)
            k_mL, k_mR = self.k_minus_lr(l, r)
            
            k_plus_n += w * (k_pL + k_pR)
            k_minus_n += w * (k_mL + k_mR)
        
        # Diagnostic info
        F_values = np.array([self.F_nuc(l, r) for l, r in states])
        shell_info = {
            'states': states,
            'weights': weights,
            'F_values': F_values,
            'n_states': len(states)
        }
        
        return k_plus_n, k_minus_n, shell_info
    
    def compute_all_rates(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict]]:
        """
        Compute k+(n) and k-(n) for all n = 0, 1, ..., N.
        
        Returns:
            (k_plus, k_minus, shell_data): Arrays of rates and per-shell diagnostics
        """
        n_values = np.arange(self.N + 1)
        k_plus = np.zeros(self.N + 1)
        k_minus = np.zeros(self.N + 1)
        shell_data = {}
        
        for n in n_values:
            k_p, k_m, info = self.compute_effective_rates(n)
            k_plus[n] = k_p
            k_minus[n] = k_m
            shell_data[n] = info
        
        return k_plus, k_minus, shell_data
    
    def compute_drift(self, k_plus: np.ndarray, k_minus: np.ndarray) -> np.ndarray:
        """
        Compute drift v(n) = k+(n) - k-(n).
        
        Args:
            k_plus: Opening rates
            k_minus: Closing rates
            
        Returns:
            Drift array v(n)
        """
        return k_plus - k_minus
    
    def find_critical_nucleus(
        self, 
        drift: np.ndarray,
        min_n: int = 1
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Find critical nucleus n* where drift changes sign.
        
        n* is the smallest n >= min_n such that v(n-1) < 0 and v(n) >= 0.
        
        Args:
            drift: Drift array v(n)
            min_n: Minimum n to consider (default: 1)
            
        Returns:
            (n_star, n_star_refined): 
                - n_star: Integer crossing point (None if no crossing)
                - n_star_refined: Linear interpolation of crossing (None if no crossing)
        """
        for n in range(min_n, len(drift)):
            if drift[n-1] < 0 and drift[n] >= 0:
                # Found crossing
                n_star = n
                
                # Refine by linear interpolation
                v1, v2 = drift[n-1], drift[n]
                if abs(v2 - v1) > 1e-12:
                    frac = -v1 / (v2 - v1)
                    n_star_refined = (n - 1) + frac
                else:
                    n_star_refined = float(n)
                
                return n_star, n_star_refined
        
        # No crossing found
        return None, None
    
    def compute_quasi_potential(
        self, 
        k_plus: np.ndarray, 
        k_minus: np.ndarray
    ) -> np.ndarray:
        """
        Compute quasi-potential Φ(n).
        
        Φ(n) = Σ_{j=1}^{n} ln[k-(j) / k+(j-1)]
        
        This measures the "free energy barrier" in the reduced coordinate.
        
        Args:
            k_plus: Opening rates
            k_minus: Closing rates
            
        Returns:
            Quasi-potential array Φ(n)
        """
        N = len(k_plus) - 1
        phi = np.zeros(N + 1)
        
        for n in range(1, N + 1):
            # Avoid log(0) or division by zero
            if k_plus[n-1] > 0 and k_minus[n] > 0:
                phi[n] = phi[n-1] + np.log(k_minus[n] / k_plus[n-1])
            elif k_plus[n-1] > 0 and k_minus[n] == 0:
                # No closing rate -> downhill (negative contribution)
                phi[n] = phi[n-1] - 10.0  # Large negative step
            else:
                # No opening rate -> infinite barrier
                phi[n] = phi[n-1] + 10.0  # Large positive step
        
        return phi
    
    def compute_1d_mfpt(
        self, 
        k_plus: np.ndarray, 
        k_minus: np.ndarray
    ) -> float:
        """
        Compute exact 1D MFPT from n=0 to n=N (absorption).
        
        Uses the birth-death MFPT formula:
        T(0) = Σ_{m=0}^{N-1} Σ_{k=0}^{m} [1/k+(k)] * Π_{j=k+1}^{m} [k-(j)/k+(j-1)]
        
        Implemented in log-space for numerical stability.
        
        Args:
            k_plus: Opening rates
            k_minus: Closing rates
            
        Returns:
            MFPT (in dimensionless time units, multiply by 1/k_wrap for physical time)
        """
        N = len(k_plus) - 1
        
        # Avoid log(0)
        k_plus_safe = np.maximum(k_plus, 1e-300)
        k_minus_safe = np.maximum(k_minus, 1e-300)
        
        # Precompute log(k-/k+)
        log_ratio = np.zeros(N + 1)
        for j in range(1, N + 1):
            log_ratio[j] = np.log(k_minus_safe[j] / k_plus_safe[j-1])
        
        mfpt = 0.0
        
        for m in range(N):
            for k in range(m + 1):
                # Term: (1/k+(k)) * Π_{j=k+1}^{m} [k-(j)/k+(j-1)]
                
                # Log of the term
                log_term = -np.log(k_plus_safe[k])
                for j in range(k + 1, m + 1):
                    log_term += log_ratio[j]
                
                # Add to MFPT
                if np.isfinite(log_term):
                    mfpt += np.exp(log_term)
        
        return mfpt
    
    def compute_nucleation_mfpt(
        self,
        phi: np.ndarray,
        drift: np.ndarray,
        n_star: Optional[int]
    ) -> Optional[float]:
        """
        Approximate MFPT using nucleation theory.
        
        T(0) ≈ (2π / |v'(0)| v'(n*)|)^{1/2} * exp(ΔΦ)
        
        where ΔΦ = Φ(n*) - Φ(0) and v'(n) is drift derivative.
        
        Args:
            phi: Quasi-potential
            drift: Drift array
            n_star: Critical nucleus
            
        Returns:
            MFPT approximation (None if n_star is None or derivatives can't be computed)
        """
        if n_star is None or n_star < 2 or n_star >= len(drift) - 1:
            return None
        
        # Barrier height
        delta_phi = phi[n_star] - phi[0]
        
        # Drift derivatives (finite difference)
        # v'(0) ≈ (v(1) - v(0))
        v_prime_0 = drift[1] - drift[0]
        
        # v'(n*) ≈ (v(n*+1) - v(n*-1)) / 2
        v_prime_nstar = (drift[n_star + 1] - drift[n_star - 1]) / 2.0
        
        if abs(v_prime_0) < 1e-12 or abs(v_prime_nstar) < 1e-12:
            return None
        
        # Nucleation formula
        prefactor = np.sqrt(2 * np.pi / (abs(v_prime_0) * abs(v_prime_nstar)))
        mfpt_nuc = prefactor * np.exp(delta_phi)
        
        return mfpt_nuc
    
    def build_1d_generator(
        self,
        k_plus: np.ndarray,
        k_minus: np.ndarray,
        sparse: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build 1D generator matrix for birth-death process.
        
        States: n = 0, 1, ..., N-1 (transient) + N (absorbing)
        
        Args:
            k_plus: Opening rates
            k_minus: Closing rates
            sparse: Use sparse matrix (default: True)
            
        Returns:
            (Q_1d, transient_states): Generator and list of transient state indices
        """
        N = len(k_plus) - 1
        transient_states = list(range(N))  # n = 0, 1, ..., N-1
        dim = N + 1  # Include absorbing state
        
        if sparse:
            # Build sparse matrix
            Q_1d = csr_matrix((dim, dim), dtype=float)
            Q_data = {}
            
            for n in range(N):
                # Birth: n -> n+1
                if n < N - 1:
                    Q_data[(n + 1, n)] = k_plus[n]
                else:
                    # Transition to absorbing state
                    Q_data[(N, n)] = k_plus[n]
                
                # Death: n -> n-1
                if n > 0:
                    Q_data[(n - 1, n)] = k_minus[n]
            
            # Diagonal entries
            for n in range(N):
                total_out = 0.0
                if n < N - 1:
                    total_out += k_plus[n]
                else:
                    total_out += k_plus[n]  # To absorbing
                if n > 0:
                    total_out += k_minus[n]
                Q_data[(n, n)] = -total_out
            
            # Convert to sparse matrix
            rows = [r for r, c in Q_data.keys()]
            cols = [c for r, c in Q_data.keys()]
            data = list(Q_data.values())
            Q_1d = csr_matrix((data, (rows, cols)), shape=(dim, dim))
        else:
            # Build dense matrix
            Q_1d = np.zeros((dim, dim))
            
            for n in range(N):
                # Birth: n -> n+1
                if n < N - 1:
                    Q_1d[n + 1, n] = k_plus[n]
                else:
                    Q_1d[N, n] = k_plus[n]
                
                # Death: n -> n-1
                if n > 0:
                    Q_1d[n - 1, n] = k_minus[n]
                
                # Diagonal
                Q_1d[n, n] = -(k_plus[n] + (k_minus[n] if n > 0 else 0))
        
        return Q_1d, transient_states
    
    def compute_1d_survival(
        self,
        k_plus: np.ndarray,
        k_minus: np.ndarray,
        t_grid: np.ndarray,
        initial_n: int = 0
    ) -> np.ndarray:
        """
        Compute survival probability S(t) for 1D process starting from n=initial_n.
        
        S(t) = P(not absorbed by time t) = Σ_{n=0}^{N-1} P_n(t)
        
        Args:
            k_plus: Opening rates
            k_minus: Closing rates
            t_grid: Time points to evaluate
            initial_n: Initial state (default: 0)
            
        Returns:
            Survival probability array S(t)
        """
        N = len(k_plus) - 1
        Q_1d, transient_states = self.build_1d_generator(k_plus, k_minus, sparse=True)
        
        # Extract transient block Q_TT
        Q_TT = Q_1d[:N, :N]
        
        # Initial distribution (concentrated at initial_n)
        p0 = np.zeros(N)
        p0[initial_n] = 1.0
        
        # Compute survival S(t) = 1^T exp(Q_TT t) p0
        survival = np.zeros(len(t_grid))
        
        for i, t in enumerate(t_grid):
            if t == 0:
                survival[i] = 1.0
            else:
                # exp(Q_TT t) p0
                pt = expm_multiply(Q_TT.multiply(t), p0)
                survival[i] = pt.sum()
        
        return survival
    
    def analyze(self) -> DriftReversalResults:
        """
        Perform complete drift reversal analysis.
        
        Returns:
            DriftReversalResults object with all computed quantities
        """
        # 1. Compute effective rates
        k_plus, k_minus, shell_data = self.compute_all_rates()
        
        # 2. Compute drift
        drift = self.compute_drift(k_plus, k_minus)
        
        # 3. Find critical nucleus
        n_star, n_star_refined = self.find_critical_nucleus(drift)
        
        # 4. Compute quasi-potential
        phi = self.compute_quasi_potential(k_plus, k_minus)
        
        # 5. Compute barrier height
        if n_star is not None:
            delta_phi = phi[n_star] - phi[0]
        else:
            delta_phi = np.nan
        
        # 6. Compute exact 1D MFPT
        mfpt_1d = self.compute_1d_mfpt(k_plus, k_minus)
        
        # 7. Nucleation approximation
        mfpt_nuc = self.compute_nucleation_mfpt(phi, drift, n_star)
        
        # 8. Compute committor (probability to detach)
        # Note: committor_birth_death expects k_plus[n] for n=0..N-1 (length N)
        # and k_minus[n] for n=0..N (length N+1, with k_minus[0] unused)
        committor = committor_birth_death(k_plus[:-1], k_minus)
        
        # 9. Package results
        results = DriftReversalResults(
            n_values=np.arange(self.N + 1),
            k_plus=k_plus,
            k_minus=k_minus,
            drift=drift,
            n_star=n_star,
            n_star_refined=n_star_refined,
            phi=phi,
            delta_phi=delta_phi,
            mfpt_1d=mfpt_1d,
            mfpt_nucleation=mfpt_nuc,
            committor=committor,
            shell_data=shell_data,
            k_wrap=self.k_wrap,
            protamine_params=self.protamine_params,
            kT=self.kT,
            N=self.N
        )
        
        return results


def compare_to_ssa(
    drift_results: DriftReversalResults,
    ssa_trajectories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    t_grid: np.ndarray
) -> Dict:
    """
    Compare 1D drift reversal predictions to SSA trajectories.
    
    Args:
        drift_results: DriftReversalResults from analyze()
        ssa_trajectories: List of (t, l, r) trajectories from SSA
        t_grid: Common time grid for comparison
        
    Returns:
        Dictionary with comparison metrics:
            - 'ssa_mfpt': Mean first passage time from SSA
            - '1d_mfpt': MFPT from 1D model
            - 'ssa_survival': Survival curve from SSA
            - '1d_survival': Survival curve from 1D model
            - 'n_trajectories': Histogram of n(t) from SSA
    """
    N = drift_results.N
    
    # Extract SSA detachment times and compute n(t)
    detachment_times = []
    n_trajectories = []
    
    for t, l, r in ssa_trajectories:
        n = l + r
        n_trajectories.append((t, n))
        
        # Find first time n reaches N
        detached_idx = np.where(n >= N)[0]
        if len(detached_idx) > 0:
            t_detach = t[detached_idx[0]]
            detachment_times.append(t_detach)
    
    # SSA MFPT
    if detachment_times:
        ssa_mfpt = np.mean(detachment_times)
    else:
        ssa_mfpt = np.inf
    
    # SSA survival curve
    ssa_survival = np.zeros(len(t_grid))
    n_traj = len(ssa_trajectories)
    
    for t_point_idx, t_point in enumerate(t_grid):
        n_survived = sum(1 for t_det in detachment_times if t_det > t_point)
        ssa_survival[t_point_idx] = n_survived / n_traj
    
    # 1D survival curve
    analyzer = DriftReversalAnalyzer(
        nucleosome=None,  # Not needed, rates already computed
        k_wrap=drift_results.k_wrap,
        protamine_params=drift_results.protamine_params
    )
    survival_1d = analyzer.compute_1d_survival(
        drift_results.k_plus,
        drift_results.k_minus,
        t_grid
    )
    
    return {
        'ssa_mfpt': ssa_mfpt,
        '1d_mfpt': drift_results.mfpt_1d / drift_results.k_wrap,  # Convert to physical time
        'ssa_survival': ssa_survival,
        '1d_survival': survival_1d,
        't_grid': t_grid,
        'n_trajectories': n_trajectories,
        'n_detached': len(detachment_times),
        'n_total': n_traj
    }
