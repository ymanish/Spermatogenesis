"""Event-driven Gillespie simulator.

Independent reimplementation of the rate / reaction logic from
src/core/gillespie_simulator.py, with two differences:
  1. No fixed tau sampling grid. Replicate runs until detachment or
     tau > tau_max (right-censoring at the boundary).
  2. Trajectory records only on n_closed-change events (UNWRAPPING /
     REWRAPPING), plus endpoints.

Returns a single ReplicateResult dataclass instead of yielding states.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import numpy as np

from src.config.custom_type import (
    Rates,
    ReactionType,
    REACTION_TARGET_STATE,
)
from src.core.nucleosomes import Nucleosome
from src.core.protamine import protamines


@dataclass
class ReplicateResult:
    detach_tau:       float
    censored:         bool
    final_tau:        float
    mean_n_open:      float
    mean_bprot:       float
    n_events_by_type: Dict[ReactionType, int]
    traj_tau:         np.ndarray
    traj_n_closed:    np.ndarray


class GillespieEventSimulator:

    def __init__(
        self,
        nuc: Nucleosome,
        prot: protamines,
        *,
        tau_max: float,
        inf_protamine: bool = True,
        seed: Optional[int] = None,
    ):
        if tau_max <= 0:
            raise ValueError(f"tau_max must be > 0, got {tau_max}")
        self.nuc = nuc
        self.prot = prot
        self.tau_max = float(tau_max)
        self.inf_protamine = inf_protamine
        self.k_wrap = float(self.nuc.k_wrap)

        if seed is not None:
            self._rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        else:
            self._rng = np.random.default_rng()

    # ── Rate machinery (same physics as src/core/gillespie_simulator.py) ──

    def _calculate_rates(self) -> Rates:
        state = self.nuc.state
        total_rates = {k: 0.0 for k in ReactionType}
        persite_rates = {k: {} for k in ReactionType}

        if self.nuc.detached == 0:
            unwrap = self.nuc.unwrapping()
            if unwrap:
                total_rates[ReactionType.UNWRAPPING] = sum(unwrap.values())
                persite_rates[ReactionType.UNWRAPPING] = unwrap

            rewrap = self.nuc.rewrapping()
            if rewrap:
                total_rates[ReactionType.REWRAPPING] = sum(rewrap.values())
                persite_rates[ReactionType.REWRAPPING] = rewrap

        bound_idx = np.ravel(np.where(state == 2))
        if len(bound_idx) > 0:
            unbind = self.prot.protein_unbinding_coop(state, bound_idx)
            total_rates[ReactionType.UNBINDING] = sum(unbind.values())
            persite_rates[ReactionType.UNBINDING] = unbind

        open_idx = np.ravel(np.where(state == 1))
        if len(open_idx) > 0:
            bind = self.prot.protein_binding(open_idx)
            total_rates[ReactionType.BINDING] = sum(bind.values())
            persite_rates[ReactionType.BINDING] = bind

        return Rates(persite_rates, total_rates)

    def _choose_reaction(self, rates: Rates) -> ReactionType:
        rt_list = list(ReactionType)
        type_rates = np.array([rates.total[rt] for rt in rt_list], dtype=float)
        total = type_rates.sum()
        if total <= 0:
            raise RuntimeError("No reactions available (total rate = 0)")
        idx = self._rng.choice(len(rt_list), p=type_rates / total)
        return rt_list[idx]

    def _perform_reaction(
        self,
        reaction: ReactionType,
        persite: Dict[ReactionType, Dict[int, float]],
    ) -> None:
        rates_dict = persite[reaction]
        if not rates_dict:
            return
        sites = list(rates_dict.keys())
        weights = np.array(list(rates_dict.values()), dtype=float)
        weights /= weights.sum()
        chosen_site = self._rng.choice(sites, p=weights)
        self.nuc.state[int(chosen_site)] = REACTION_TARGET_STATE[reaction].value

    def _update_species_count(self, reaction: ReactionType) -> None:
        if reaction == ReactionType.UNWRAPPING:
            self.nuc.n_closed -= 1
        elif reaction == ReactionType.REWRAPPING:
            self.nuc.n_closed += 1
        elif reaction == ReactionType.BINDING:
            if not self.inf_protamine:
                self.prot.P_free -= 1
            self.prot.N_bound += 1
        elif reaction == ReactionType.UNBINDING:
            if not self.inf_protamine:
                self.prot.P_free += 1
            self.prot.N_bound -= 1

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> ReplicateResult:
        tau = 0.0
        binding_sites = int(self.nuc.binding_sites)

        # Time-weighted accumulators
        sum_n_open = 0.0
        sum_bprot = 0.0

        # Event counter
        n_events_by_type: Dict[ReactionType, int] = {rt: 0 for rt in ReactionType}

        # Trajectory: always emit initial state
        traj_tau_list = [0.0]
        traj_n_closed_list = [binding_sites]

        detach_tau = math.nan
        censored = False

        while True:
            rates = self._calculate_rates()
            total_rate = sum(rates.total.values())

            if total_rate <= 0:
                # Means n_closed > 0 but every reaction has zero rate — a
                # physics bug. Censoring should only happen by tau_max.
                if self.nuc.detached == 0:
                    raise RuntimeError(
                        "total_rate == 0 while still attached "
                        f"(n_closed={self.nuc.n_closed}); upstream model is stuck"
                    )
                # Already detached: shouldn't reach here since we break on
                # detachment below.
                break

            u = self._rng.random()
            # Guard against u == 0 (would give dtau = inf with log(1/0))
            while u <= 0.0:
                u = self._rng.random()
            dtau = math.log(1.0 / u) / total_rate

            # Current state values (constant on [tau, tau+dtau])
            n_open_now = binding_sites - int(self.nuc.n_closed)
            bprot_now = int(self.prot.N_bound)

            tau_event = tau + dtau

            if tau_event > self.tau_max:
                # Censor: credit partial interval [tau, tau_max], emit terminal
                dt_partial = self.tau_max - tau
                sum_n_open += n_open_now * dt_partial
                sum_bprot += bprot_now * dt_partial
                tau = self.tau_max
                censored = True
                traj_tau_list.append(tau)
                traj_n_closed_list.append(int(self.nuc.n_closed))
                break

            # Credit full interval [tau, tau_event]
            sum_n_open += n_open_now * dtau
            sum_bprot += bprot_now * dtau
            tau = tau_event

            reaction = self._choose_reaction(rates)
            self._perform_reaction(reaction, rates.persite)
            self._update_species_count(reaction)
            n_events_by_type[reaction] += 1

            # Record on n_closed-change events
            if reaction in (ReactionType.UNWRAPPING, ReactionType.REWRAPPING):
                traj_tau_list.append(tau)
                traj_n_closed_list.append(int(self.nuc.n_closed))

            # Detachment check
            if self.nuc.detached == 0 and self.nuc.n_closed == 0:
                self.nuc.detached = 1
                self.nuc.detach_time = tau
                detach_tau = tau
                # Ensure terminal row is present (the UNWRAPPING that brought
                # n_closed to 0 was just recorded above, so tau matches).
                break

        final_tau = tau
        if final_tau > 0:
            mean_n_open = sum_n_open / final_tau
            mean_bprot = sum_bprot / final_tau
        else:
            mean_n_open = 0.0
            mean_bprot = 0.0

        return ReplicateResult(
            detach_tau=detach_tau,
            censored=censored,
            final_tau=final_tau,
            mean_n_open=mean_n_open,
            mean_bprot=mean_bprot,
            n_events_by_type=n_events_by_type,
            traj_tau=np.asarray(traj_tau_list, dtype=np.float64),
            traj_n_closed=np.asarray(traj_n_closed_list, dtype=np.uint8),
        )
