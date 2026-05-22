"""
Hybrid Rejection + Inner Fast Tau-Leap Simulator
=================================================

Hybrid simulator for nucleosome-protamine dynamics with rejection sampling:
- SLOW reactions (nucleosome wrap/unwrap): SSA with rejection based on protamine occupancy
- FAST reactions (protamine binding/unbinding): Inner tau-leaping loop

Algorithm Overview:
------------------
1. Sample Δt_slow from intrinsic nucleosome rates (ignoring protamines)
2. Choose candidate slow reaction R_cand via SSA
3. Run inner tau-leap on protamines for time Δt_slow (with fixed nucleosome state)
4. At time t + Δt_slow, attempt R_cand:
   - Unwrapping: Always accepted
   - Rewrapping: Accepted only if site is unbound (state[k] == 1)
5. Advance time by Δt_slow regardless of acceptance
6. Repeat

Key difference from regular tau-leap: Protamines evolve in an inner loop
while nucleosome state is held fixed, then nucleosome attempts one move with
possible rejection based on final protamine configuration.

Author: MY
Date: 2024-11-25
"""

import numpy as np
import math
from typing import Iterator, List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.config.custom_type import SimulationState, Rates, ReactionType
from src.core.nucleosomes import Nucleosomes
from src.core.protamine import protamines


@dataclass
class SlowReaction:
    """Represents a candidate slow nucleosome reaction."""
    rxn_type: ReactionType  # UNWRAPPING or REWRAPPING
    site: int               # Which site (edge of wrapped block)
    rate: float            # Intrinsic rate (ignoring protamines)
    i_new: int             # New i index if accepted
    j_new: int             # New j index if accepted


class HybridRejectionSimulator:
    """
    Hybrid Rejection + Inner Fast Tau-Leap Simulator.
    
    Uses SSA for slow nucleosome reactions with rejection sampling,
    and tau-leaping for fast protamine dynamics in inner loop.
    
    This is more physically accurate than pure tau-leap because:
    - Nucleosome moves are exact (one at a time)
    - Protamine blocking of rewrapping is explicit
    - Time is always advanced, rejected attempts still consume time
    """
    
    def __init__(
        self,
        nuc_inst: Nucleosomes,
        prot_inst: protamines,
        tau_points: np.ndarray,
        epsilon: float = 0.1,
        inf_protamine: bool = False,
        seed: Optional[int] = None,
        tau_min: Optional[float] = None
    ):
        """
        Initialize hybrid rejection simulator.
        
        Args:
            nuc_inst: Nucleosomes instance
            prot_inst: protamines instance
            tau_points: Dimensionless time sampling points
            epsilon: Accuracy parameter for inner tau-leap (default 0.1)
            inf_protamine: Whether protamine pool is infinite
            seed: Random seed
            tau_min: Minimum dwell time for detachment
        """
        self.nuc = nuc_inst
        self.prot = prot_inst
        self.tau_points = np.asarray(tau_points, dtype=float)
        self.epsilon = epsilon
        self.inf_protamine = inf_protamine
        self.tau_min = tau_min
        
        # Reference rate for time conversion
        self.k_wrap = float(self.nuc.k_wrap)
        self.num_nuc = self.nuc.num_nucleosomes
        
        # Tracking for dwell-based detachment
        self._unwrapped_since = np.full(self.num_nuc, np.nan)
        
        # Statistics tracking
        self.stats = {
            'total_attempts': 0,
            'unwrap_attempts': 0,
            'rewrap_attempts': 0,
            'rewrap_accepted': 0,
            'rewrap_rejected': 0
        }
        
        # Initialize time
        self.tau = 0.0
        self.t = 0.0
        
        if seed is not None:
            np.random.seed(seed)
    
    def _uniform_pos_arg(self):
        """Sample uniform random number in (0, 1)."""
        return np.random.uniform(0.0, 1.0)
    
    def _compute_slow_intrinsic_reactions(self, nuc_idx: int) -> Tuple[List[SlowReaction], float]:
        """
        Compute INTRINSIC slow reaction rates (ignoring protamines).
        
        This builds all possible nucleosome moves from current (i,j) state,
        using only the G-matrix energies and k_wrap.
        
        Returns:
            (reactions, total_rate) where reactions is list of SlowReaction objects
        """
        if self.nuc[nuc_idx].detached == 1:
            return ([], 0.0)
        
        nuc = self.nuc[nuc_idx]
        reactions = []
        total_rate = 0.0
        
        # Current (i, j) indices from the wrapped block
        # i = left edge (first wrapped site)
        # j = right edge (last wrapped site)
        # Wrapped region: sites [i, i+1, ..., j]
        
        # Determine current (i,j) from state array
        wrapped_sites = np.where(nuc.state == 0)[0]
        if len(wrapped_sites) == 0:
            # Fully unwrapped - could allow nucleation here if desired
            return ([], 0.0)
        
        i_current = int(wrapped_sites[0])
        j_current = int(wrapped_sites[-1])
        L = nuc.binding_sites
        
        # --- UNWRAPPING MOVES ---
        # Unwrapping increases number of open contacts
        
        # Unwrap left: (i,j) → (i+1,j) if i < j
        if i_current < j_current:
            site = i_current  # leftmost wrapped site
            i_new = i_current + 1
            j_new = j_current
            
            # Rate = k_wrap * exp(-ΔG/kT)
            dG = nuc.G_mat[i_new, j_new] - nuc.G_mat[i_current, j_current]
            rate = self.k_wrap * np.exp(-dG / nuc.kT)
            
            reactions.append(SlowReaction(
                rxn_type=ReactionType.UNWRAPPING,
                site=site,
                rate=rate,
                i_new=i_new,
                j_new=j_new
            ))
            total_rate += rate
        
        # Unwrap right: (i,j) → (i,j-1) if i < j
        if i_current < j_current:
            site = j_current  # rightmost wrapped site
            i_new = i_current
            j_new = j_current - 1
            
            # Rate = k_wrap * exp(-ΔG/kT)
            dG = nuc.G_mat[i_new, j_new] - nuc.G_mat[i_current, j_current]
            rate = self.k_wrap * np.exp(-dG / nuc.kT)
            
            reactions.append(SlowReaction(
                rxn_type=ReactionType.UNWRAPPING,
                site=site,
                rate=rate,
                i_new=i_new,
                j_new=j_new
            ))
            total_rate += rate
        
        # --- REWRAPPING MOVES ---
        # Rewrapping decreases number of open contacts
        # INTRINSIC rate is k_wrap (ignoring protamines)
        
        # Rewrap left: (i,j) → (i-1,j) if i > 0
        if i_current > 0:
            site = i_current - 1  # site to be wrapped
            i_new = i_current - 1
            j_new = j_current
            
            # Intrinsic rate = k_wrap (will be rejected if protamine bound)
            rate = self.k_wrap
            
            reactions.append(SlowReaction(
                rxn_type=ReactionType.REWRAPPING,
                site=site,
                rate=rate,
                i_new=i_new,
                j_new=j_new
            ))
            total_rate += rate
        
        # Rewrap right: (i,j) → (i,j+1) if j < L-1
        if j_current < L - 1:
            site = j_current + 1  # site to be wrapped
            i_new = i_current
            j_new = j_current + 1
            
            # Intrinsic rate = k_wrap (will be rejected if protamine bound)
            rate = self.k_wrap
            
            reactions.append(SlowReaction(
                rxn_type=ReactionType.REWRAPPING,
                site=site,
                rate=rate,
                i_new=i_new,
                j_new=j_new
            ))
            total_rate += rate
        
        return (reactions, total_rate)
    
    def _sample_delta_t_slow(self, total_slow_rate: float) -> float:
        """
        Sample time to next slow attempt from exponential distribution.
        
        Args:
            total_slow_rate: Sum of intrinsic slow rates (dimension: 1/seconds)
            
        Returns:
            Δt_slow (dimensionless time, in units of 1/k_wrap)
        """
        if total_slow_rate <= 0:
            return float('inf')
        
        u = self._uniform_pos_arg()
        # dtau = log(1/u) * (k_wrap / total_rate)
        dtau = np.log(1.0 / u) * (self.k_wrap / total_slow_rate)
        return dtau
    
    def _select_candidate_reaction(self, reactions: List[SlowReaction]) -> Optional[SlowReaction]:
        """
        Select one candidate slow reaction via SSA.
        
        Args:
            reactions: List of possible slow reactions
            
        Returns:
            Selected SlowReaction, or None if list is empty
        """
        if len(reactions) == 0:
            return None
        
        rates = np.array([r.rate for r in reactions])
        total = rates.sum()
        
        if total <= 0:
            return None
        
        probs = rates / total
        idx = np.random.choice(len(reactions), p=probs)
        return reactions[idx]
    
    def _compute_fast_rates(self, nuc_idx: int) -> List[Tuple[int, ReactionType, float]]:
        """
        Compute FAST reaction rates (protamine binding/unbinding).
        
        Returns:
            List of (site, reaction_type, rate) tuples
        """
        nucleo = self.nuc[nuc_idx].state
        fast_rates = []
        
        # Binding (fast)
        open_indexes = np.ravel(np.where(nucleo == 1))
        if len(open_indexes) > 0:
            bind_dict = self.prot.protein_binding(open_indexes)
            for site, rate in bind_dict.items():
                fast_rates.append((site, ReactionType.BINDING, rate))
        
        # Unbinding (fast)
        bound_indexes = np.ravel(np.where(nucleo == 2))
        if len(bound_indexes) > 0:
            unbind_dict = self.prot.protein_unbinding_coop(nucleo, bound_indexes)
            for site, rate in unbind_dict.items():
                fast_rates.append((site, ReactionType.UNBINDING, rate))
        
        return fast_rates
    
    def _compute_tau_fast(self, fast_rates: List[Tuple[int, ReactionType, float]]) -> float:
        """
        Compute τ_fast from binary leap condition.
        
        For each fast reaction: P(flip) = 1 - exp(-rate * tau) <= epsilon
        
        Args:
            fast_rates: List of (site, type, rate) for protamine reactions (1/seconds)
            
        Returns:
            tau_fast: Maximum allowed time step (dimensionless, in units of 1/k_wrap)
        """
        if len(fast_rates) == 0:
            return float('inf')
        
        c = -np.log(1.0 - self.epsilon)  # -ln(1 - ε)
        tau_fast = float('inf')
        
        for site, rxn_type, rate in fast_rates:
            if rate > 0:
                # tau_k in seconds, convert to dimensionless
                tau_k_seconds = c / rate
                tau_k_dimensionless = tau_k_seconds * self.k_wrap
                if tau_k_dimensionless < tau_fast:
                    tau_fast = tau_k_dimensionless
        
        return tau_fast
    
    def _inner_fast_tau_leap(self, nuc_idx: int, delta_t_slow: float):
        """
        Inner loop: evolve fast protamine dynamics for time Δt_slow.
        
        Uses tau-leaping on protamines while keeping nucleosome state fixed.
        
        Args:
            nuc_idx: Nucleosome index
            delta_t_slow: Time interval to simulate (dimensionless)
        """
        t_fast = 0.0  # Local time counter for inner loop
        
        while t_fast < delta_t_slow:
            # 3a. Compute fast propensities at current state
            fast_rates = self._compute_fast_rates(nuc_idx)
            
            if len(fast_rates) == 0:
                # No fast reactions possible, jump to end
                break
            
            # 3b. Compute tau_fast from leap condition
            tau_fast = self._compute_tau_fast(fast_rates)
            
            # 3c. Don't step beyond Δt_slow
            dt = min(tau_fast, delta_t_slow - t_fast)
            
            if dt <= 0:
                break
            
            # 3d. Apply binary tau-leap to protamines
            self._protamine_binary_leap(nuc_idx, fast_rates, dt)
            
            t_fast += dt
    
    def _protamine_binary_leap(self, nuc_idx: int, fast_rates: List[Tuple[int, ReactionType, float]], tau: float):
        """
        Perform binary τ-leap on protamine subsystem.
        
        For each site: P(flip) = 1 - exp(-rate * tau)
        
        Args:
            nuc_idx: Nucleosome index
            fast_rates: Pre-computed rates (dimension: 1/seconds)
            tau: Time step for leap (dimensionless, in units of 1/k_wrap)
        """
        nucleo = self.nuc[nuc_idx].state
        
        # Convert tau from dimensionless to seconds
        tau_seconds = tau / self.k_wrap
        
        for site, rxn_type, rate in fast_rates:
            if rate <= 0:
                continue
            
            # P(flip) = 1 - exp(-rate * tau)
            p_flip = 1.0 - np.exp(-rate * tau_seconds)
            
            if self._uniform_pos_arg() < p_flip:
                # Perform flip
                if rxn_type == ReactionType.BINDING:
                    if nucleo[site] == 1:  # Verify still open
                        nucleo[site] = 2
                        self._update_protamine_count(+1)
                
                elif rxn_type == ReactionType.UNBINDING:
                    if nucleo[site] == 2:  # Verify still bound
                        nucleo[site] = 1
                        self._update_protamine_count(-1)
    
    def _attempt_slow_reaction(self, nuc_idx: int, reaction: SlowReaction) -> bool:
        """
        Attempt to perform candidate slow reaction with rejection.
        
        Unwrapping: Always accepted
        Rewrapping: Accepted only if site is unbound (state[k] == 1)
        
        Args:
            nuc_idx: Nucleosome index
            reaction: Candidate slow reaction
            
        Returns:
            True if accepted, False if rejected
        """
        nuc = self.nuc[nuc_idx]
        site = reaction.site
        
        self.stats['total_attempts'] += 1
        
        if reaction.rxn_type == ReactionType.UNWRAPPING:
            # Unwrapping is always accepted
            self.stats['unwrap_attempts'] += 1
            
            nuc.state[site] = 1  # Wrapped → Open
            nuc.n_closed -= 1
            
            # Track when fully unwrapped
            if nuc.n_closed == 0:
                if math.isnan(self._unwrapped_since[nuc_idx]):
                    self._unwrapped_since[nuc_idx] = self.tau
            
            return True
        
        elif reaction.rxn_type == ReactionType.REWRAPPING:
            # Rewrapping requires site to be unbound
            self.stats['rewrap_attempts'] += 1
            
            if nuc.state[site] == 1:  # Open and unbound → accept
                nuc.state[site] = 0  # Open → Wrapped
                nuc.n_closed += 1
                
                # Clear unwrapped tracking
                if nuc.n_closed == 1:
                    self._unwrapped_since[nuc_idx] = np.nan
                
                self.stats['rewrap_accepted'] += 1
                return True
            
            else:  # Site is bound (state[site] == 2) → reject
                # Nucleosome state unchanged, time still advances
                self.stats['rewrap_rejected'] += 1
                return False
        
        return False
    
    def _update_protamine_count(self, delta: int):
        """Update protamine counts."""
        if self.inf_protamine:
            self.prot.N_bound += delta
        else:
            volume = 1
            self.prot.P_free -= delta / volume
            self.prot.N_bound += delta / volume
    
    def _update_detachment_flags(self):
        """Check and update detachment flags."""
        if self.tau_min is None:
            for i in range(self.num_nuc):
                if self.nuc[i].detached == 0 and self.nuc[i].n_closed == 0:
                    self.nuc[i].detached = 1
                    self.nuc[i].detach_time = self.t
            return
        
        for i in range(self.num_nuc):
            if self.nuc[i].detached == 0 and self.nuc[i].n_closed == 0:
                t0 = self._unwrapped_since[i]
                if not math.isnan(t0) and ((self.tau - t0) >= self.tau_min):
                    self.nuc[i].detached = 1
                    self.nuc[i].detach_time = self.t
    
    def _get_state(self) -> SimulationState:
        """Get current simulation state."""
        cs = np.array([self.nuc[i].n_closed for i in range(self.num_nuc)])
        cs_total = cs.sum()
        
        detached = np.array([self.nuc[i].detached for i in range(self.num_nuc)])
        detached_tot = detached.sum()
        
        return SimulationState(
            tau=self.tau,
            time=self.t,
            cs_total=cs_total,
            detached_total=detached_tot,
            bprot=self.prot.N_bound
        )
    
    def run(self) -> Iterator[SimulationState]:
        """
        Run hybrid rejection + inner fast tau-leap simulation with
        PRE-EVENT recording on the sampling grid.

        For each slow step of duration delta_tau_slow:
        - All grid points up to and including tau_event see the state
          BEFORE that slow+fast step.
        - The inner fast tau-leap and the slow reaction attempt happen
          AFTER recording those pre-event states.
        """

        grid_tau = self.tau_points.astype(float)
        i_record = 0
        num_points = len(grid_tau)

        # Initialize global time
        self.tau = 0.0
        self.t = 0.0

        # Main loop over sampling grid
        while i_record < num_points:

            ### Loop over nucleosomes (sequential hybrid steps).
            ### Right now I have only one nucleosome per simulation, so it is okay ###
            #### However, this hybrid logic needs to be revised for multiple nucleosomes ####
            for nuc_idx in range(self.num_nuc):

                # Skip already detached nucleosomes
                if self.nuc[nuc_idx].detached == 1:
                    continue

                # === STEP 1: Compute intrinsic slow rates (ignoring protamines) ===
                reactions, total_slow = self._compute_slow_intrinsic_reactions(nuc_idx)
                if total_slow <= 0:
                    continue

                # === STEP 2: Sample Δtau_slow and choose candidate reaction ===
                delta_tau_slow = self._sample_delta_t_slow(total_slow)
                if not np.isfinite(delta_tau_slow) or delta_tau_slow <= 0:
                    continue

                # Time at the start of this slow step
                tau_start = self.tau

                # Do not go beyond final grid time
                remaining = grid_tau[-1] - tau_start
                if remaining <= 0:
                    # No time left to simulate, break to fill remaining grid points
                    break
                delta_tau_slow = min(delta_tau_slow, remaining)

                if delta_tau_slow <= 0:
                    continue

                # Candidate slow reaction chosen based on state at tau_start
                R_cand = self._select_candidate_reaction(reactions)
                if R_cand is None:
                    continue

                # Time of the slow "event" for this step
                tau_event = tau_start + delta_tau_slow

                # === STEP 3: PRE-EVENT SAMPLING ON THE GRID ===
                # We record all grid points <= tau_event using the CURRENT state
                # (i.e., before any fast tau-leap or slow reaction is applied).
                while i_record < num_points and grid_tau[i_record] <= tau_event:
                    self.tau = float(grid_tau[i_record])
                    self.t = self.tau / self.k_wrap
                    if self.tau_min is not None:
                        self._update_detachment_flags()
                    yield self._get_state()
                    i_record += 1

                # If we've exhausted the grid, we can still update internal state
                # but there is nothing left to record.
                if i_record >= num_points:
                    # Advance time to tau_event, apply fast+slow once, then break
                    self.tau = tau_event
                    self.t = self.tau / self.k_wrap

                    # Inner fast tau-leap over [tau_start, tau_event]
                    self._inner_fast_tau_leap(nuc_idx, delta_tau_slow)

                    # Attempt slow reaction at tau_event
                    self._attempt_slow_reaction(nuc_idx, R_cand)

                    # Update detachment flags at final time
                    if self.tau_min is not None:
                        self._update_detachment_flags()
                    break
            
                # === STEP 4: ADVANCE GLOBAL TIME TO THE EVENT ===
                # (No further recording here; we've already emitted all
                #  pre-event states up to tau_event.)
                self.tau = tau_event
                self.t = self.tau / self.k_wrap

                # === STEP 5: INNER FAST TAU-LEAP OVER delta_tau_slow ===
                # Protamines evolve over this interval while nucleosome state is held fixed.
                self._inner_fast_tau_leap(nuc_idx, delta_tau_slow)

                # === STEP 6: ATTEMPT SLOW REACTION WITH REJECTION ===
                self._attempt_slow_reaction(nuc_idx, R_cand)

                # === STEP 7: UPDATE DETACHMENT FLAGS AT tau_event ===
                if self.tau_min is not None:
                    self._update_detachment_flags()

                # (No grid recording here; any future grid points belong to the
                #  next step and will see this post-event state as their "pre-event"
                #  baseline.)

                # If we've filled all grid points, break out
                if i_record >= num_points:
                    break

            # Check if all nucleosomes are detached
            if all(self.nuc[i].detached == 1 for i in range(self.num_nuc)):
                # Fill remaining grid points with frozen detached state
                while i_record < num_points:
                    self.tau = float(grid_tau[i_record])
                    self.t = self.tau / self.k_wrap
                    if self.tau_min is not None:
                        self._update_detachment_flags()
                    yield self._get_state()
                    i_record += 1
                break

        # Safety: fill any remaining grid points (if the loop exited early)
        while i_record < num_points:
            self.tau = float(grid_tau[i_record])
            self.t = self.tau / self.k_wrap
            if self.tau_min is not None:
                self._update_detachment_flags()
            yield self._get_state()
            i_record += 1

    
    def print_stats(self):
        """Print simulation statistics."""
        print("=" * 60)
        print("HYBRID REJECTION SIMULATOR STATISTICS")
        print("=" * 60)
        print(f"Total slow attempts:     {self.stats['total_attempts']}")
        print(f"  Unwrap attempts:       {self.stats['unwrap_attempts']} (always accepted)")
        print(f"  Rewrap attempts:       {self.stats['rewrap_attempts']}")
        print(f"    Accepted:            {self.stats['rewrap_accepted']}")
        print(f"    Rejected:            {self.stats['rewrap_rejected']}")
        
        if self.stats['rewrap_attempts'] > 0:
            acceptance = 100.0 * self.stats['rewrap_accepted'] / self.stats['rewrap_attempts']
            print(f"  Rewrap acceptance rate: {acceptance:.1f}%")
        print("=" * 60)


if __name__ == "__main__":
    """Example usage of hybrid rejection simulator."""
    print("Hybrid Rejection + Inner Fast Tau-Leap Simulator module loaded.")
    print("Use HybridRejectionSimulator for physically accurate protamine blocking.")
