#/src/core/gillespie_simulator.py
import numpy as np
import math

from typing import Iterator, Dict
from src.config.custom_type import SimulationState, Rates, ReactionType, REACTION_TARGET_STATE

from src.core.protamine import protamines
from src.core.nucleosomes import Nucleosome


class GillespieSimulator:

    def __init__(self, nuc_inst: Nucleosome, prot_inst: protamines,
                 t_points: np.ndarray | None, max_steps: int | None,
                 inf_protamine: bool = False,
                 seed: int | None = 25,
                 tau_min: float | None = None,
                 tau_points: np.ndarray | None = None):

        if (t_points is None) and (tau_points is None) and (max_steps is None):
            raise ValueError("Provide tau_points (preferred), or t_points, or max_steps.")

        self.nuc = nuc_inst
        self.prot = prot_inst

        self.k_wrap = float(self.nuc.k_wrap)
        self.tau_points = None if tau_points is None else np.asarray(tau_points, dtype=float)
        self.tau = 0.0

        self.t_points = t_points
        self.t = 0.0
        self.max_steps = max_steps
        self.inf_protamine = inf_protamine
        self.tau_min = tau_min
        self._unwrapped_since = np.nan  # tau when fully-unwrapped state began

        if seed is not None:
            np.random.seed(seed)

    def calculate_rates(self) -> Rates:
        state = self.nuc.state

        total_rates = {k: 0 for k in ReactionType}
        persite_rates = {k: {} for k in ReactionType}

        if self.nuc.detached == 0:
            unwrapping_sites_rate = self.nuc.unwrapping()
            if unwrapping_sites_rate:
                total_rates[ReactionType.UNWRAPPING] = sum(unwrapping_sites_rate.values())
                persite_rates[ReactionType.UNWRAPPING] = unwrapping_sites_rate

            rewrapping_sites_rate = self.nuc.rewrapping()
            if rewrapping_sites_rate:
                total_rates[ReactionType.REWRAPPING] = sum(rewrapping_sites_rate.values())
                persite_rates[ReactionType.REWRAPPING] = rewrapping_sites_rate

        protein_bound_indexes = np.ravel(np.where(state == 2))
        if len(protein_bound_indexes) > 0:
            unbound_sites_rate = self.prot.protein_unbinding_coop(state, protein_bound_indexes)
            total_rates[ReactionType.UNBINDING] = sum(unbound_sites_rate.values())
            persite_rates[ReactionType.UNBINDING] = unbound_sites_rate

        open_indexes = np.ravel(np.where(state == 1))
        if len(open_indexes) > 0:
            bound_sites_rate = self.prot.protein_binding(open_indexes)
            total_rates[ReactionType.BINDING] = sum(bound_sites_rate.values())
            persite_rates[ReactionType.BINDING] = bound_sites_rate

        return Rates(persite_rates, total_rates)

    def _choose_reaction(self, rates: Rates) -> ReactionType:
        rt_list = list(ReactionType)
        type_rates = np.array([rates.total[rt] for rt in rt_list], dtype=float)
        total = type_rates.sum()
        if total <= 0:
            raise RuntimeError("No reactions available (total rate = 0)")
        choice_idx = np.random.choice(len(rt_list), p=type_rates / total)
        return rt_list[choice_idx]

    def _uniform_pos_arg(self):
        return np.random.uniform(0.0, 1.0)

    def perform_reaction(self, reaction: ReactionType,
                         persite: Dict[ReactionType, Dict[int, float]]) -> None:
        rates_dict = persite[reaction]
        if not rates_dict:
            return
        sites, weights = zip(*rates_dict.items())
        w = np.array(weights, dtype=float)
        w /= w.sum()
        chosen_site = np.random.choice(sites, p=w)
        self.nuc.state[chosen_site] = REACTION_TARGET_STATE[reaction].value

    def _update_detachment_flags(self):
        if self.tau_min is None:
            if self.nuc.detached == 0 and self.nuc.n_closed == 0:
                self.nuc.detached = 1
                self.nuc.detach_time = self.t
            return

        if self.nuc.detached == 0 and self.nuc.n_closed == 0:
            t0 = self._unwrapped_since
            if not math.isnan(t0) and (self.tau - t0) >= self.tau_min:
                self.nuc.detached = 1
                self.nuc.detach_time = self.t

    def _update_species_count(self, reaction: ReactionType):
        if reaction == ReactionType.UNWRAPPING:
            self.nuc.n_closed -= 1
            if self.nuc.n_closed == 0 and math.isnan(self._unwrapped_since):
                self._unwrapped_since = self.tau

        elif reaction == ReactionType.REWRAPPING:
            self.nuc.n_closed += 1
            if self.nuc.n_closed == 1:
                self._unwrapped_since = np.nan

        elif reaction == ReactionType.BINDING:
            if self.inf_protamine:
                self.prot.N_bound += 1
            else:
                self.prot.P_free -= 1
                self.prot.N_bound += 1

        elif reaction == ReactionType.UNBINDING:
            if self.inf_protamine:
                self.prot.N_bound -= 1
            else:
                self.prot.P_free += 1
                self.prot.N_bound -= 1

    def _get_state(self) -> SimulationState:
        return SimulationState(
            tau=self.tau,
            time=self.t,
            cs_total=self.nuc.n_closed,
            detached_total=self.nuc.detached,
            bprot=self.prot.N_bound
        )

    def run(self) -> Iterator[SimulationState]:
        """Run the Gillespie simulation; advance natively in tau.
        Sample on self.tau_points if provided, else convert self.t_points to tau."""

        if self.tau_points is not None:
            grid_tau = self.tau_points.astype(float)
        else:
            if self.t_points is None:
                raise RuntimeError("No sampling grid: provide tau_points or t_points.")
            grid_tau = np.asarray(self.t_points, dtype=float) * self.k_wrap

        i_record = 0
        num_points = len(grid_tau)

        self.tau = 0.0
        self.t = 0.0

        while i_record < num_points:
            rates = self.calculate_rates()
            total_rate = sum(rates.total.values())

            if total_rate <= 0:
                print(f'No reactions possible, exiting simulation: total_rate = {total_rate}')
                while i_record < num_points:
                    self.tau = grid_tau[i_record]
                    self.t = float(self.tau / self.k_wrap)
                    if self.tau_min is not None:
                        self._update_detachment_flags()
                    yield self._get_state()
                    i_record += 1
                break

            # Draw dimensionless time increment
            dtau = np.log(1.0 / self._uniform_pos_arg()) * (self.k_wrap / total_rate)
            tau_event = self.tau + dtau

            # Yield at crossed grid points
            while i_record < num_points and grid_tau[i_record] < tau_event:
                self.tau = float(grid_tau[i_record])
                self.t = self.tau / self.k_wrap
                if self.tau_min is not None:
                    self._update_detachment_flags()
                yield self._get_state()
                i_record += 1

            if i_record >= num_points:
                break

            # Advance to event, choose and perform reaction
            self.tau = tau_event
            self.t = self.tau / self.k_wrap

            reaction = self._choose_reaction(rates)
            self.perform_reaction(reaction, persite=rates.persite)
            self._update_species_count(reaction)
            self._update_detachment_flags()

            # If nucleosome detached, fill remaining time points with frozen state
            if self.nuc.detached == 1:
                while i_record < num_points:
                    self.tau = float(grid_tau[i_record])
                    self.t = self.tau / self.k_wrap
                    yield self._get_state()
                    i_record += 1
                break

        # Fill any remaining grid points
        while i_record < num_points:
            self.tau = float(grid_tau[i_record])
            self.t = self.tau / self.k_wrap
            if self.tau_min is not None:
                self._update_detachment_flags()
            yield self._get_state()
            i_record += 1


if __name__ == "__main__":
    import sys
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from src.core.build_nucleosomes import nucleosome_generator_sprm
    from src.core.protamine import protamines
    from src.config.path import SPRM_DATA_DIR


    # ── Parameters ──────────────────────────────────────────────────────────
    SPRM_DIR    = SPRM_DATA_DIR / "ret_single_nuc"
    N_NUCS      = 3           # nucleosomes to test
    N_REPS      = 5           # replicates per nucleosome
    K_WRAP      = 1.0
    K_UNBIND    = 89.7
    K_BIND      = 1.0
    P_CONC      = 10.0
    COOPERATIVITY = 0.0
    TAU_MAX     = 5000.0
    TAU_STEPS   = 2000
    INF_PROT    = True
    BINDING_SITES = 14
    # ────────────────────────────────────────────────────────────────────────

    tau_points = np.linspace(0, TAU_MAX, TAU_STEPS)

    gen = itertools.islice(
        nucleosome_generator_sprm(SPRM_DIR, k_wrap=K_WRAP, kT=1.0,
                                  binding_sites=BINDING_SITES),
        N_NUCS
    )

    all_ok = True
    fig, axes = plt.subplots(N_NUCS, 1, figsize=(10, 3 * N_NUCS), sharex=True)
    if N_NUCS == 1:
        axes = [axes]

    for nuc_idx, nuc in enumerate(gen):
        print(f"\n── Nucleosome {nuc_idx}: id={nuc.id!r}, subid={nuc.subid} ──")
        rep_trajectories = []

        for rep in range(N_REPS):
            import copy
            nuc_copy = copy.deepcopy(nuc)
            prot = protamines(k_unbind=K_UNBIND, k_bind=K_BIND,
                              p_conc=P_CONC, cooperativity=COOPERATIVITY)
            sim = GillespieSimulator(
                nuc_inst=nuc_copy, prot_inst=prot,
                t_points=None, max_steps=None,
                inf_protamine=INF_PROT,
                seed=rep * 1000 + nuc_idx,
                tau_points=tau_points
            )

            taus, cs = [], []
            prev_detached = 0
            ok = True

            for state in sim.run():
                taus.append(state.tau)
                cs.append(state.cs_total)

                # ── Correctness checks ────────────────────────────────
                if not (0 <= state.cs_total <= BINDING_SITES):
                    print(f"  [FAIL] rep={rep}: cs_total={state.cs_total} out of range at tau={state.tau:.2f}")
                    ok = False

                if state.bprot < 0:
                    print(f"  [FAIL] rep={rep}: negative bprot={state.bprot} at tau={state.tau:.2f}")
                    ok = False

                if state.detached_total not in (0, 1):
                    print(f"  [FAIL] rep={rep}: detached_total={state.detached_total} invalid at tau={state.tau:.2f}")
                    ok = False

                if prev_detached == 1 and state.detached_total == 0:
                    print(f"  [FAIL] rep={rep}: detachment reversed at tau={state.tau:.2f}")
                    ok = False
                prev_detached = state.detached_total
                # ─────────────────────────────────────────────────────

            # Check trajectory length
            if len(taus) != TAU_STEPS:
                print(f"  [FAIL] rep={rep}: expected {TAU_STEPS} points, got {len(taus)}")
                ok = False

            # Check tau grid matches
            if not np.allclose(taus, tau_points):
                print(f"  [FAIL] rep={rep}: tau grid mismatch")
                ok = False

            detach_time = sim.nuc.detach_time
            status = "detached" if sim.nuc.detached else "attached"
            print(f"  rep={rep}: final cs={cs[-1]}, {status}"
                  + (f", detach_tau={detach_time*K_WRAP:.1f}" if sim.nuc.detached else "")
                  + ("  [OK]" if ok else "  [FAIL]"))
            all_ok = all_ok and ok
            rep_trajectories.append(np.array(cs))

        # Plot mean ± std across replicates
        ax = axes[nuc_idx]
        stack = np.vstack(rep_trajectories)
        mean_cs = stack.mean(axis=0)
        std_cs  = stack.std(axis=0)
        ax.plot(tau_points, mean_cs, lw=1.5, label="mean cs")
        ax.fill_between(tau_points, mean_cs - std_cs, mean_cs + std_cs, alpha=0.25)
        for traj in rep_trajectories:
            ax.plot(tau_points, traj, lw=0.5, alpha=0.4, color="grey")
        ax.set_ylabel("cs_total")
        ax.set_ylim(-0.5, BINDING_SITES + 0.5)
        ax.set_title(f"{nuc.id}  (subid={nuc.subid})")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("tau (dimensionless)")
    fig.suptitle(f"Gillespie sanity check — {SPRM_DIR}", fontsize=11)
    plt.tight_layout()
    # plt.savefig("gillespie_sanity_check.png", dpi=150)
    plt.show()

    print(f"\n{'All checks PASSED' if all_ok else 'Some checks FAILED'}")
    sys.exit(0 if all_ok else 1)


