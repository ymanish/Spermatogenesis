"""
Hybrid SSA + Tau-Leaping Simulator
===================================

Hybrid simulator for nucleosome-protamine dynamics:
- SLOW reactions (nucleosome wrap/unwrap): Exact SSA
- FAST reactions (protamine binding/unbinding): Binary τ-leaping

This approach is efficient when protamines equilibrate much faster than
nucleosomes, allowing us to leap over many fast binding/unbinding events
while keeping nucleosome transitions exact.

Algorithm Overview:
------------------
1. Compute slow propensities (nucleosome wrap/unwrap) → Δt_slow via SSA
2. Compute fast rates (protamine binding/unbinding) → τ_fast from ε-condition
3. Choose τ = min(Δt_slow, τ_fast, remaining_time)
4. Perform binary τ-leap on protamines over τ
5. If τ == Δt_slow: Fire exactly ONE slow nucleosome reaction
6. Update time and repeat

Key: Protamine rates are computed ONCE at start of step and held constant
during the τ-leap (no recomputation inside the leap).

Author: MY
Date: 2024-11-24
"""

import numpy as np
import math
from typing import Iterator, List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.config.custom_type import SimulationState, Rates, ReactionType
from src.core.nucleosomes import Nucleosomes
from src.core.protamine import protamines


class TauLeapingSimulator:
    """
    Tau-leaping simulator using existing Nucleosomes and protamines classes.
    
    Like GillespieSimulator but uses tau-leaping instead of exact SSA:
    - Computes tau from epsilon-leaping condition
    - Fires Poisson(a_i * tau) reactions per channel
    - Much faster when rates are high
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
        Initialize τ-leaping simulator.
        
        Args:
            nuc_inst: Nucleosomes instance (same as Gillespie)
            prot_inst: protamines instance (same as Gillespie)
            tau_points: Dimensionless time sampling points
            epsilon: Accuracy parameter (default 0.1, smaller = more accurate)
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
        
        # Initialize time
        self.tau = 0.0
        self.t = 0.0
        
        if seed is not None:
            np.random.seed(seed)
    
    def _compute_slow_rates(self, nuc_idx: int) -> Tuple[Dict[ReactionType, Dict[int, float]], float]:
        """
        Compute SLOW reaction rates (nucleosome wrap/unwrap only).
        
        Returns:
            (rates_dict, total_rate) where rates_dict has keys:
            ReactionType.UNWRAPPING, ReactionType.REWRAPPING mapping to per-site dicts
        """
        if self.nuc[nuc_idx].detached == 1:
            return ({}, 0.0)
        
        slow_rates = {}
        total = 0.0
        
        # Unwrapping (slow)
        unwrap_dict = self.nuc[nuc_idx].unwrapping()
        if len(unwrap_dict) > 0:
            slow_rates[ReactionType.UNWRAPPING] = unwrap_dict
            total += sum(unwrap_dict.values())
        
        # Rewrapping (slow)
        rewrap_dict = self.nuc[nuc_idx].rewrapping()
        if len(rewrap_dict) > 0:
            slow_rates[ReactionType.REWRAPPING] = rewrap_dict
            total += sum(rewrap_dict.values())
        
        return (slow_rates, total)
    
    def _compute_fast_rates(self, nuc_idx: int) -> List[Tuple[int, ReactionType, float]]:
        """
        Compute FAST reaction rates (protamine binding/unbinding).
        
        Returns:
            List of (site, reaction_type, rate) tuples for each protamine site
            reaction_type is ReactionType.BINDING or ReactionType.UNBINDING
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
    
    def _uniform_pos_arg(self):
        return np.random.uniform(0.0, 1.0)
    
    def _sample_delta_t_slow(self, total_slow_rate: float) -> float:
        """
        Sample time to next slow event from exponential distribution.
        
        CRITICAL: total_slow_rate has dimension [1/time], so we need to
        convert to dimensionless time using k_wrap, exactly like Gillespie.
        
        Args:
            total_slow_rate: Sum of all slow reaction rates (dimension: 1/seconds)
            
        Returns:
            Delta_t_slow (dimensionless time, in units of 1/k_wrap)
        """
        if total_slow_rate <= 0:
            return float('inf')

        u = self._uniform_pos_arg()
        # Same as Gillespie: dtau = log(1/u) * (k_wrap / total_rate)
        dtau = np.log(1 / u) * (self.k_wrap / total_slow_rate)
        return dtau

    def _compute_tau_fast(self, fast_rates: List[Tuple[int, str, float]]) -> float:
        """
        Compute τ_fast from binary leap condition.
        
        For each fast reaction: P(flip) = 1 - exp(-rate * tau) <= epsilon
        Therefore: tau <= -ln(1 - epsilon) / rate
        
        CRITICAL: Rates have dimension [1/time], so tau_k = c / rate has
        dimension [time]. We convert to dimensionless by multiplying k_wrap.
        
        Args:
            fast_rates: List of (site, type, rate) for protamine reactions (dimension: 1/seconds)
            
        Returns:
            tau_fast: Maximum allowed time step (dimensionless, in units of 1/k_wrap)
        """
        if len(fast_rates) == 0:
            return float('inf')
        
        c = -np.log(1.0 - self.epsilon)  # -ln(1 - ε)
        tau_fast = float('inf')
        
        for site, rxn_type, rate in fast_rates:
            if rate > 0:
                # tau_k has dimension [seconds], convert to dimensionless
                tau_k_seconds = c / rate
                tau_k_dimensionless = tau_k_seconds * self.k_wrap
                if tau_k_dimensionless < tau_fast:
                    tau_fast = tau_k_dimensionless
        
        return tau_fast
    
    def _protamine_binary_leap(self, nuc_idx: int, fast_rates: List[Tuple[int, str, float]], tau: float):
        """
        Perform binary τ-leap on protamine subsystem.
        
        CRITICAL: Uses rates computed at START of step, held constant during leap.
        For each site: P(flip) = 1 - exp(-rate * tau)
        
        Args:
            nuc_idx: Nucleosome index
            fast_rates: Pre-computed rates from START of step (dimension: 1/seconds)
            tau: Time step for leap (dimensionless, in units of 1/k_wrap)
        """
        nucleo = self.nuc[nuc_idx].state
        
        # Convert tau from dimensionless to seconds
        tau_seconds = tau / self.k_wrap
        
        for site, rxn_type, rate in fast_rates:
            if rate <= 0:
                continue
            
            # rate has dimension [1/seconds], tau_seconds has dimension [seconds]
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
    
    def _fire_one_slow_reaction(self, nuc_idx: int, slow_rates: Dict[ReactionType, Dict[int, float]]):
        """
        Fire exactly ONE slow reaction via SSA selection.
        
        Args:
            nuc_idx: Nucleosome index
            slow_rates: Dict of slow reaction rates (ReactionType -> site -> rate)
        """
        # Flatten all slow reactions
        all_reactions = []
        for rxn_type, site_dict in slow_rates.items():
            for site, rate in site_dict.items():
                all_reactions.append((rxn_type, site, rate))
        
        if len(all_reactions) == 0:
            return
        
        # SSA selection
        rates = np.array([r for _, _, r in all_reactions])
        total = rates.sum()
        
        if total <= 0:
            return
        
        probs = rates / total
        choice_idx = np.random.choice(len(all_reactions), p=probs)
        
        rxn_type, site, rate = all_reactions[choice_idx]
        
        # Perform the reaction using ReactionType
        if rxn_type == ReactionType.UNWRAPPING:
            self.nuc[nuc_idx].state[site] = 1  # Wrapped -> Open
            self.nuc[nuc_idx].n_closed -= 1
            
            if self.nuc[nuc_idx].n_closed == 0:
                if math.isnan(self._unwrapped_since[nuc_idx]):
                    self._unwrapped_since[nuc_idx] = self.tau
        
        elif rxn_type == ReactionType.REWRAPPING:
            self.nuc[nuc_idx].state[site] = 0  # Open -> Wrapped
            self.nuc[nuc_idx].n_closed += 1
            
            if self.nuc[nuc_idx].n_closed == 1:
                self._unwrapped_since[nuc_idx] = np.nan
    
    def _update_protamine_count(self, delta: int):
        """Update protamine counts."""
        if self.inf_protamine:
            self.prot.N_bound += delta
        else:
            volume = 1
            self.prot.P_free -= delta / volume
            self.prot.N_bound += delta / volume
    
    def _update_detachment_flags(self):
        """Check and update detachment flags (same as Gillespie)."""
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
        """Get current simulation state (same format as Gillespie)."""
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
        Run hybrid SSA + τ-leaping simulation.
        
        Hybrid algorithm:
        1. Compute Δt_slow (SSA for nucleosome wrap/unwrap)
        2. Compute τ_fast (ε-condition for protamine binding/unbinding)
        3. Choose τ = min(Δt_slow, τ_fast, remaining)
        4. Binary τ-leap protamines over τ (rates held constant)
        5. If τ == Δt_slow: Fire ONE slow nucleosome reaction
        6. Advance time and repeat
        
        Yields:
            SimulationState at each sampling point
        """
        grid_tau = self.tau_points.astype(float)
        i_record = 0
        num_points = len(grid_tau)
        
        # Initialize
        self.tau = 0.0
        self.t = 0.0
        
        while i_record < num_points:
            # For each nucleosome, perform one hybrid step
            # (For simplicity, process nucleosomes sequentially)
            # In QSSA regime, protamines should be much faster
            
            for nuc_idx in range(self.num_nuc):
                if self.nuc[nuc_idx].detached == 1:
                    continue
                
                # === STEP 1: Compute slow rates (nucleosome wrap/unwrap) ===
                slow_rates, total_slow = self._compute_slow_rates(nuc_idx)
                delta_tau_slow = self._sample_delta_t_slow(total_slow)
                
                # === STEP 2: Compute fast rates (protamine bind/unbind) ===
                # CRITICAL: Rates computed ONCE at start of step
                fast_rates = self._compute_fast_rates(nuc_idx)
                tau_fast = self._compute_tau_fast(fast_rates)
                
                # === STEP 3: Choose τ ===
                remaining = grid_tau[-1] - self.tau
                tau_step = min(delta_tau_slow, tau_fast, remaining)
                print(f"nuc_idx={nuc_idx}, delta_tau_slow={delta_tau_slow:.4f}, tau_fast={tau_fast:.4f}, tau_step={tau_step:.4f}")

                if tau_step <= 0 or tau_step == float('inf'):
                    continue
                
                # === STEP 4: Binary τ-leap on protamines ===
                # Use rates from STEP 2, held constant during leap
                self._protamine_binary_leap(nuc_idx, fast_rates, tau_step)
                
                # === STEP 5: If τ == Δτ_slow, fire ONE slow reaction ===
                # Use small tolerance for float comparison
                if delta_tau_slow < float('inf') and abs(tau_step - delta_tau_slow) < 1e-12:
                    self._fire_one_slow_reaction(nuc_idx, slow_rates)
                
                # === STEP 6: Advance time ===
                self.tau += tau_step
                self.t = self.tau / self.k_wrap
                
                # Update detachment
                self._update_detachment_flags()
            
            # === Yield at sampling points ===
            while i_record < num_points and grid_tau[i_record] <= self.tau:
                self.tau = float(grid_tau[i_record])
                self.t = self.tau / self.k_wrap
                if self.tau_min is not None:
                    self._update_detachment_flags()
                yield self._get_state()
                i_record += 1
            
            # Check if all nucleosomes detached
            if all(self.nuc[i].detached == 1 for i in range(self.num_nuc)):
                while i_record < num_points:
                    self.tau = float(grid_tau[i_record])
                    self.t = self.tau / self.k_wrap
                    yield self._get_state()
                    i_record += 1
                break
            
            # Safety: if no progress, advance to next grid point
            if i_record < num_points and self.tau < grid_tau[i_record]:
                # Force advance to avoid infinite loop
                self.tau = grid_tau[i_record]
                self.t = self.tau / self.k_wrap
        
        # Fill any remaining points
        while i_record < num_points:
            self.tau = float(grid_tau[i_record])
            self.t = self.tau / self.k_wrap
            if self.tau_min is not None:
                self._update_detachment_flags()
            yield self._get_state()
            i_record += 1


if __name__ == "__main__":
    """Example usage of τ-leaping simulator."""
    print("Tau-leaping simulator module loaded.")
    print("Use TauLeapingSimulator - same interface as GillespieSimulator but with tau-leaping.")
