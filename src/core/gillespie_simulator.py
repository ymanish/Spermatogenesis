#/src/core/gillespie_simulator.py
import numpy as np
import math
import csv

from dataclasses import dataclass
from typing import List, Iterator, Tuple, Dict
from src.config.custom_type import SimulationState, Rates, ReactionChoice, ReactionType, REACTION_TARGET_STATE

from src.core.protamine import protamines
from src.utils.logger_util import get_logger

from src.core.nucleosomes import Nucleosomes


class GillespieSimulator():

    def __init__(self, nuc_inst: Nucleosomes, prot_inst: protamines, 
                 t_points: np.ndarray | None, max_steps: int | None,
                   inf_protamine: bool = False, 
                   seed: int | None = 25, 
                   tau_min: float | None = None):  

        if t_points is None and max_steps is None:
            raise ValueError("Either t_points or max_steps must be provided for Gillespie simulation.")
        
        self.nuc = nuc_inst
        self.prot = prot_inst
        self.t_points = t_points
        self.t = 0.0
        self.num_nuc = self.nuc.num_nucleosomes
        self.max_steps = max_steps
        self.inf_protamine = inf_protamine
        self.tau_min = tau_min
        self._unwrapped_since = np.full(self.num_nuc, np.nan)


        if seed is not None:
            np.random.seed(seed)

    def calculate_rates(self, nuc_idx: int) -> Rates:
        nucleo = self.nuc[nuc_idx].state

        assert nucleo.ndim == 1, f"nucleo must be a 1D numpy array, got {nucleo.ndim}D, process one nucleosome at a time."

        total_rates = {k: 0 for k in ReactionType}
        persite_rates = {k: {} for k in ReactionType}

        # histone_occupied_indx = np.ravel(np.where(nucleo == 0))
        # if len(histone_occupied_indx) == 0:
        #     pass
        # else:
        #     unwrapping_sites_rate = self.nuc[nuc_idx].unwrapping()
        #     total_rates[ReactionType.UNWRAPPING] = sum(unwrapping_sites_rate.values())
        #     persite_rates[ReactionType.UNWRAPPING] = unwrapping_sites_rate

        #     rewrapping_sites_rate = self.nuc[nuc_idx].rewrapping()
        #     total_rates[ReactionType.REWRAPPING] = sum(rewrapping_sites_rate.values())
        #     persite_rates[ReactionType.REWRAPPING] = rewrapping_sites_rate

        if self.nuc[nuc_idx].detached == 1:
            pass
        else:
            unwrapping_sites_rate = self.nuc[nuc_idx].unwrapping()
            if len(unwrapping_sites_rate) > 0:
                # print(f"Found {len(unwrapping_sites_rate)} unwrapping sites in nucleosome {nuc_idx}.")
                total_rates[ReactionType.UNWRAPPING] = sum(unwrapping_sites_rate.values())
                persite_rates[ReactionType.UNWRAPPING] = unwrapping_sites_rate

            rewrapping_sites_rate = self.nuc[nuc_idx].rewrapping()
            if len(rewrapping_sites_rate) > 0:
                # print(f"Found {len(rewrapping_sites_rate)} rewrapping sites in nucleosome {nuc_idx}.")
                total_rates[ReactionType.REWRAPPING] = sum(rewrapping_sites_rate.values())
                persite_rates[ReactionType.REWRAPPING] = rewrapping_sites_rate

        protein_bound_indexes = np.ravel(np.where(nucleo == 2))
        if len(protein_bound_indexes) > 0:
            # print(f"Found {len(protein_bound_indexes)} protamine bound sites in nucleosome {nuc_idx}.")

            unbound_sites_rate = self.prot.protein_unbinding_coop(nucleo, protein_bound_indexes)
            total_rates[ReactionType.UNBINDING] = sum(unbound_sites_rate.values())
            persite_rates[ReactionType.UNBINDING] = unbound_sites_rate
        
        open_indexes = np.ravel(np.where(nucleo == 1))
        if len(open_indexes) == 0:
            total_rates[ReactionType.BINDING] = 0
            persite_rates[ReactionType.BINDING] = {}
        else:
            bound_sites_rate = self.prot.protein_binding(open_indexes)
            total_rates[ReactionType.BINDING] = sum(bound_sites_rate.values())
            persite_rates[ReactionType.BINDING] = bound_sites_rate

        return Rates(persite_rates, total_rates)

    def _choose_reaction(self, all_rates:List[Rates]) -> ReactionChoice:
        """Given list of (total_rate, per_site_dict), pick which nuc and which reaction."""
        # 1) pick nucleosome index by relative total_rate
        # 2) pick reaction within that nucleosome
         # Select nucleosome and reaction to perform
        nuc_weights = np.array([sum(r.total.values()) for r in all_rates], dtype=float)
        totalN = nuc_weights.sum()
        if totalN <= 0:
            raise RuntimeError("No reactions available (total rate = 0)")
        p_nuc = nuc_weights / totalN
        nuc_idx = np.random.choice(self.num_nuc, p=p_nuc)

        rt_list    = list(ReactionType)
        rates_dict = all_rates[nuc_idx].total

        type_rates = np.array([rates_dict[rt] for rt in rt_list], dtype=float)
        sub_total  = type_rates.sum()
        if sub_total <= 0:
            raise RuntimeError(f"No reactions for nucleosome {nuc_idx}")
        
        type_probs = type_rates / sub_total
        choice_idx = np.random.choice(len(rt_list), p=type_probs)
        chosen_rt  = rt_list[choice_idx]

        return ReactionChoice(nuc_idx=nuc_idx, 
                              reaction=chosen_rt)

    def _uniform_pos_arg(self):
        return np.random.uniform(0.0, 1.0)

    def perform_reaction(self, choice: ReactionChoice, 
                         persite:Dict[ReactionType, Dict[int, float]])->None:
        # nucleosome is a list of sites, with some value (say, 0 for closed, 1 for open/unbound, 2 for open/bound)
        nuc_idx = choice.nuc_idx
        react_id = choice.reaction  # Get the index of the reaction type

        rates_dict = persite[react_id]
        if not rates_dict:
            return
        
        sites, weights = zip(*rates_dict.items())
        w = np.array(weights, dtype=float)
        w /= w.sum()

        chosen_site = np.random.choice(sites, p=w)

        new_digit = REACTION_TARGET_STATE[react_id].value

        self.nuc[nuc_idx].state[chosen_site] = new_digit
        return



    def _update_detachment_flags(self):
        if self.tau_min is None:
            return
        for i in range(self.num_nuc):
            if self.nuc[i].detached == 0 and self.nuc[i].n_closed == 0:
                t0 = self._unwrapped_since[i]
                if not math.isnan(t0) and ((self.t - t0) >= self.tau_min):  # t0 not NaN
                    self.nuc[i].detached = 1
                    self.nuc[i].detach_time = self.t

    def _update_species_count(self, reaction:ReactionChoice):
        if reaction is None:
            return
        nuc_idx = reaction.nuc_idx

        if reaction.reaction == ReactionType.UNWRAPPING:
            # Open site
            self.nuc[nuc_idx].n_closed -= 1
            # if self.nuc[nuc_idx].n_closed == 0 and self.nuc[nuc_idx].detached == 0:
            #     self.nuc[nuc_idx].detached = 1

            ###NEW PART
            if self.nuc[nuc_idx].n_closed == 0:
                if math.isnan(self._unwrapped_since[nuc_idx]):
                    self._unwrapped_since[nuc_idx] = self.t   # start dwell clock


        if reaction.reaction == ReactionType.REWRAPPING:
            # Close site
            self.nuc[nuc_idx].n_closed += 1

            ###NEW PART
            if self.nuc[nuc_idx].n_closed == 1:
                # left the fully-unwrapped state -> reset dwell clock
                self._unwrapped_since[nuc_idx] = np.nan

        if self.inf_protamine:
            if reaction.reaction == ReactionType.BINDING:
                # Protamine binds
                self.prot.N_bound += 1
            if reaction.reaction == ReactionType.UNBINDING:
                # Protamine unbinds
                self.prot.N_bound -= 1
        
        else:
            volume = 1 ### micrometer^3, assume constant volume
            if reaction.reaction == ReactionType.BINDING:
                # Protamine binds
                self.prot.P_free -= 1/volume
                self.prot.N_bound += 1/volume
            if reaction.reaction == ReactionType.UNBINDING:
                # Protamine unbinds
                self.prot.P_free += 1/volume
                self.prot.N_bound -= 1/volume


    def _accumulate_end_blocking(self, dt: float) -> None:
        """
        Accumulate per-nucleosome total and blocked residence times over interval dt.
        - A nucleosome is considered 'blocked' if its leftmost or rightmost wrapped
          site (state==0) has an adjacent protamine (state==2).
        - Robust to missing attributes: will create t_total / t_block if absent.
        """
        for i in range(self.num_nuc):
            nuc = self.nuc[i]
            try:
                state = nuc.state
                # fallback for binding_sites
                binding_sites = getattr(nuc, "binding_sites", len(state))
                blocked = False
                wrapped_idx = np.where(state == 0)[0]
                if wrapped_idx.size > 0:
                    L = int(wrapped_idx[0])
                    R = int(wrapped_idx[-1])
                    left_block = (L > 0) and (state[L - 1] == 2)
                    right_block = (R < (binding_sites - 1)) and (state[R + 1] == 2)
                    blocked = left_block or right_block

                # ensure accumulators exist
                if not hasattr(nuc, "t_total"):
                    setattr(nuc, "t_total", 0.0)
                if not hasattr(nuc, "t_block"):
                    setattr(nuc, "t_block", 0.0)

                nuc.t_total += float(dt)
                if blocked:
                    nuc.t_block += float(dt)
            except Exception:
                # ignore malformed nucleosome objects
                continue


    def _get_state(self, takenuc_snapshot:bool=False, takecs_snapshot:bool=False) -> SimulationState:

        cs = np.array([self.nuc[i].n_closed for i in range(self.num_nuc)])
        cs_total = cs.sum()

        detached = np.array([self.nuc[i].detached for i in range(self.num_nuc)])
        detached_tot = detached.sum()

        # detach_times = np.array([self.nuc[i].detach_time for i in range(self.num_nuc)])
        # detach_times_total = detach_times.sum()

        frac_blocked = []
        for i in range(self.num_nuc):
            nuc = self.nuc[i]
            t_total = getattr(nuc, "t_total", 0.0)
            t_block = getattr(nuc, "t_block", 0.0)
            # if t_total > 0:
            #     frac_blocked.append(t_block / t_total)
            # else:
            #     frac_blocked.append(0.0)

        if not takecs_snapshot:
            cs = None

        if takenuc_snapshot:
            nucs_snapshot = [self.nuc[i].state.copy() for i in range(self.num_nuc)]
        else:
            nucs_snapshot = None
        return SimulationState(time=self.t, 
                                cs=cs,
                                cs_total=cs_total,
                                detached_total=detached_tot,
                                bprot=self.prot.N_bound,
                                t_blocked=t_block,
                                nucs_snapshot=nucs_snapshot)

    def run(self) -> Iterator[SimulationState]:

        """Run the Gillespie simulation for tpoints steps."""

        i_record = 0
        num_points = len(self.t_points)
        while i_record < num_points:

            # logger.info(f'Simulation step {n+1}/{self.STEPS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<\n')

            rates:List[Rates] = [self.calculate_rates(idx) for idx in range(self.num_nuc)]
            total_rate = sum(sum(r.total.values()) for r in rates)

            # logger.info(f'Rates at step {n}: {rates}')
            # logger.info(f'Total rate at step {n}: {total_rate}')

            if total_rate <= 0:
                print(f'No reactions possible, exiting simulation : total_rate = {total_rate}')
                print(f'self._get_state() at step {i_record}: {self._get_state()}')
                # Yield remaining time point states
                while i_record < num_points:
                    self.t = self.t_points[i_record]
                    self._update_detachment_flags()
                    yield self._get_state()
                    i_record += 1
                break

            # advance reaction time
            u_ = self._uniform_pos_arg()
            dt = np.log(1 / u_) / total_rate
            te = self.t + dt

             # accumulate end‑blocking / residence time for the current state
            self._accumulate_end_blocking(dt)

            # Yield at crossed time points
            while i_record < num_points and self.t_points[i_record] < te:
                self.t = float(self.t_points[i_record])   # <-- advance time to that grid point
                self._update_detachment_flags()           # <-- check τ_min at this time
                yield self._get_state()
                i_record += 1
            
            if i_record >= num_points:
                break

            self.t = te
            reaction_choice = self._choose_reaction(rates)
            react_rates = rates[reaction_choice.nuc_idx].persite

            # Perform reaction
            self.perform_reaction(reaction_choice,
                                    persite=react_rates)

            # Update the species count
            self._update_species_count(reaction_choice)

            self._update_detachment_flags()

            # ###>>>>>>>>>>>>>>>>>>>>>NEW PART<<<<<<<<<<<<<<<<<<<<<<<<<
            # # Check if nucleosome is detached

            # if self.nuc[0].detached==1:  # Assuming single nuc per sim
            #     # Fill remaining time points with frozen state (ensure correct length)
            #     # Advance self.t along t_points so downstream code sees monotonically increasing times.
            #     while i_record < num_points:
            #         self.t = self.t_points[i_record]
            #         yield self._get_state()
            #         i_record += 1
            #     break  # i_record now == num_points; final fill loop will be skipped
            # ###<<<<<<<<<<<<<<<<<<<<<<<<<NEW PART>>>>>>>>>>>>>>>>>>>>>>>>>
            
        # Fill any remaining
        while i_record < num_points:
            self.t = self.t_points[i_record]
            self._update_detachment_flags()
            yield self._get_state()
            i_record += 1


    def run_steps(self) -> Iterator[SimulationState]:
        step = 0
        while step < self.max_steps:
            rates = [self.calculate_rates(i) for i in range(self.num_nuc)]
            total_rate = sum(sum(r.total.values()) for r in rates)
            if total_rate <= 0:
                return
            dt = np.log(1 / self._uniform_pos_arg()) / total_rate
            self.t += dt
            choice = self._choose_reaction(rates)
            self.perform_reaction(choice, persite=rates[choice.nuc_idx].persite)
            self._update_species_count(choice)


            state = self._get_state(takecs_snapshot=False, takenuc_snapshot=False)
            if state.cs_total == 0:
                # logger.info(f"All nucleosomes are fully unwrapped at time {self.t} and step {step}, exiting simulation.")
                return

            yield state
            step += 1




if __name__ == "__main__":
    
    import time
    from src.core.build_nucleosomes import build_nucleosomes_from_file, nucleosome_generator
    import itertools
    import matplotlib.pyplot as plt
    
    logger = get_logger(__name__, log_file=None, level='INFO')

    file_path_bound = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/boundprom/breath_energy/001.tsv" 
    file_path_unbound = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/unboundprom/breath_energy/001.tsv"

    ids_bound = ["ENST00000210633.4"] ## bound_ids
    ids_unbound = ["ENST00000695399.1"] ###unbound_ids

    k_wrap = 1.0
    binding_sites = 14
    k_unbind = 0.01
    k_bind = 1.0
    p_conc = 0.1
    cooperativity = 0.0
    
    t_max = 10000.0
    t_steps = 10000
    inf_protamine = True
    t_points = np.linspace(0, t_max, t_steps)

    gen = nucleosome_generator(file_path=file_path_unbound, k_wrap=k_wrap,
                               binding_sites=binding_sites,
                               ids=ids_unbound,
                               subids=np.arange(1965, 2200).tolist())
    gen = itertools.islice(gen, 1)


    # Collect trajectories from each simulation run
    all_times = []
    all_cs = []   
    for nuc_idx, nuc in enumerate(gen):
        print(f"Processing nucleosome {nuc_idx}: {nuc.id}, subid: {nuc.subid}")
        nucs = Nucleosomes(
            k_wrap       = k_wrap,
            kT           = 1.0,
            nucleosomes  = [nuc],  # Wrap single nucleosome
            binding_sites= binding_sites,
        )
        prot_inst = protamines(k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity)
        sim = GillespieSimulator(nuc_inst=nucs, 
                                prot_inst=prot_inst,
                                t_points=t_points, 
                                max_steps=None,
                                inf_protamine=inf_protamine, seed=4, 
                                tau_min=60.0)
        times = []
        cs_list = []
        for st in sim.run():
            times.append(st.time)
            cs_list.append(st.cs_total)
        # Convert to numpy arrays
        times = np.array(times)
        cs_arr = np.array(cs_list)  # shape (n_steps,)
        all_times.append(times)
        all_cs.append(cs_arr)

    # Assume all runs yield the same time points (so we can use the first run's times)
    if all_cs:
        avg_times = all_times[0]
        avg_cs = np.mean(np.vstack(all_cs), axis=0)

        plt.figure(figsize=(8, 4))
        plt.plot(avg_times, avg_cs, color="C0", lw=2, label="Average Total Wrapped")
        plt.xlabel("Time")
        plt.ylabel("Total wrapped (n_closed)")
        plt.xlim(0, t_max)
        plot_title = f"Average Total Wrapped Over Time (k_wrap={k_wrap}, k_unbind={k_unbind}, k_bind={k_bind}, p_conc={p_conc}, coop={cooperativity})"
        plt.title(plot_title)
        plt.legend()
        
        # Create a file name including the parameter values
        file_name = f"avg_total_wrapped_kwrap{k_wrap}_k_unbind{k_unbind}_k_bind{k_bind}_pconc{p_conc}_coop{cooperativity}.png"
        # plt.savefig(file_name, dpi=300)
        plt.show()
    else:
        print("No simulation runs were completed.")

    import sys
    sys.exit(0)








































































    # nucleosomes_instance = build_nucleosomes_from_file(file_path, k_wrap=350, kT=1.0, binding_sites=14, max_nucs=1)
    # protamines_instance = protamines(k_unbind=0.23, k_bind=1, p_conc=0.1, cooperativity=10.0)

    # simulation = GillespieSimulator(nuc_inst=nucleosomes_instance,
    #                                 prot_inst=protamines_instance,
    #                                 t_points=None,
    #                                 max_steps=10000, inf_protamine=True)
    from src.core.helper.estimate_dt import estimate_timescales

    pilot_steps = 10000
    t_max = 10.0
    logger.info(f"Running pilot simulation with {pilot_steps} steps and t_max={t_max}")
    start_time = time.time()
    avg_tau, tau_decorr = estimate_timescales(nucleosomes_instance, protamines_instance, pilot_steps, t_max, logger=logger)
    pilot_time = time.time() - start_time
    logger.info(f"Pilot run took {pilot_time:.2f} seconds.")


    print(f"Estimated average tau: {avg_tau:.2e} s, decorrelation time: {tau_decorr:.2e} s")
    import sys
    sys.exit(0)

    scale_factor = 10.0  # Scale decorrelation time by this factor
    num_points_main = 1000  # Number of points in the main simulation
    t_max_main = max(tau_decorr * scale_factor, t_max * 2)  # e.g., 10x decorrelation time
    spacing = tau_decorr / 2  # Resolve ~10 points per decorrelation
    logger.info(f"Main simulation t_max={t_max_main:.2f} s, spacing={spacing:.2f} s")
    logger.info(f"Estimated decorrelation time: {tau_decorr:.2f} s, avg tau: {avg_tau:.2e} s")
    num_points_main = min(num_points_main, int(t_max_main / spacing) + 1)
    logger.info(f"Number of points in main simulation: {num_points_main}")

    t_points_main = np.linspace(0, t_max_main, num_points_main)
    logger.info(f"Time points for main simulation: {t_points_main}")

    # reset_simulation(nucleosomes_instance, protamines_instance)

    import sys
    sys.exit(0)

    t_points_main = np.linspace(0, 100.0, 10000)
    simulation = GillespieSimulator(nucleosomes_instance, protamines_instance, t_points_main, max_steps=None)

    # for state in simulation.run():
    #     logger.info(state)

    # logger.info('Simulation completed')

    times = []
    cs_list = []    
    bprot_list = []

    for st in simulation.run():
        times.append(st.time)
        cs_list.append(st.cs_total)
        bprot_list.append(st.bprot)

    # import numpy as np
    import matplotlib.pyplot as plt

    times = np.array(times)
    cs_arr = np.array(cs_list)     # shape (n_steps, num_nuc)
    bprot = np.array(bprot_list)   # shape (n_steps,)

    # 1) plot total closed sites
    total_closed = cs_list
    plt.figure(figsize=(8,4))
    plt.plot(times, total_closed, color="C0", lw=2)
    plt.xlabel("Time")
    plt.ylabel("Total wrapped (n_closed)")
    plt.title("Total Wrapped Over Time")

    # 2) plot bound protamine
    plt.figure(figsize=(8,4))
    plt.plot(times, bprot, color="C1", lw=2)
    plt.xlabel("Time")
    plt.ylabel("Number of bound protamines")
    plt.title("Bound Protamine Over Time")

    plt.show()

