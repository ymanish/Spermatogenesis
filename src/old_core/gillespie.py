#/src/core/gillespie.py
import numpy as np
import math
import csv

from dataclasses import dataclass
from typing import List, Iterator, Tuple, Dict
from src.config.custom_type import SimulationState, Rates, ReactionChoice, ReactionType, REACTION_TARGET_STATE

from src.core.nucleosome import nucleosome
from src.core.protamine import protamines
from src.cyt_script.tricodec import int_to_tri14, tri14_to_int
from src.cyt_script.edit_tricodec import edit_tricodec
from src.utils.logger_util import get_logger



logger = get_logger(__name__, log_file=None, level='INFO')

logger.info('Starting Gillespie simulation')

class GillespieSimulator():
    def __init__(self, nuc_inst: nucleosome, prot_inst: protamines,  num_nucleosomes: int, STEPS:int=1000):
        self.nuc = nuc_inst
        self.prot = prot_inst
        self.STEPS = STEPS # Number of steps
        self.t = 0
        self.num_nuc = num_nucleosomes
        self.nuc_fall_flag = False
        # self.uniform_pos_arg_njit = nb.njit(self.uniform_pos_arg)


    def calculate_rates(self, one_nuc_state: np.int32) -> Rates:
        nucleo_str = int_to_tri14(int(one_nuc_state))

        nucleo = np.frombuffer(nucleo_str.encode('ascii'),
                               dtype=np.uint8) - ord('0')
        # logger.info(f"Converted nucleosome state to numpy array: {nucleo}")

        assert nucleo.ndim == 1, f"nucleo must be a 1D numpy array, got {nucleo.ndim}D, process one nucleosome at a time."

        total_rates = {k: 0 for k in ReactionType}
        persite_rates = {k: {} for k in ReactionType}

        histone_occupied_indx = np.ravel(np.where(nucleo == 0))
        if len(histone_occupied_indx) == 0:
            total_rates[ReactionType.UNWRAPPING] = 0
            persite_rates[ReactionType.UNWRAPPING] = {}

            total_rates[ReactionType.REWRAPPING] = 0
            persite_rates[ReactionType.REWRAPPING] = {}


        else:
            unwrapping_sites_rate = self.nuc.unwrapping(histone_occupied_indx)
            total_rates[ReactionType.UNWRAPPING] = sum(unwrapping_sites_rate.values())
            persite_rates[ReactionType.UNWRAPPING] = unwrapping_sites_rate


            rewrapping_sites_rate = self.nuc.rewrapping(nucleo, histone_occupied_indx)
            total_rates[ReactionType.REWRAPPING] = sum(rewrapping_sites_rate.values())
            persite_rates[ReactionType.REWRAPPING] = rewrapping_sites_rate


        protein_bound_indexes = np.ravel(np.where(nucleo == 2))
        if len(protein_bound_indexes) == 0:
            total_rates[ReactionType.UNBINDING] = 0
            persite_rates[ReactionType.UNBINDING] = {}
        else:
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

        new_state = edit_tricodec(value=self.nuc.nuc_state[nuc_idx], 
                                  position=chosen_site, 
                                  new_digit=new_digit)
        
        self.nuc.nuc_state[nuc_idx] = new_state
        return


    def _update_species_count(self, reaction:ReactionChoice):
        if reaction.reaction == 0:
            # Open site
            self.nuc.N_closed -= 1
        elif reaction.reaction == 1:
            # Close site
            self.nuc.N_closed += 1
        elif reaction.reaction == 2:
            # Protamine binds
            self.prot.P_free -= 1
            self.prot.N_bound += 1
        else:
            # Protamine unbinds
            self.prot.P_free += 1
            self.prot.N_bound -= 1


    def run(self) -> Iterator[SimulationState]:

        """Run the Gillespie simulation for N steps."""


        for n in range(self.STEPS):

            # logger.info(f'Simulation step {n+1}/{self.STEPS} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

            rates:List[Rates] = [self.calculate_rates(one_nuc_state=nc) for nc in self.nuc.nuc_state]
            total_rate = sum(sum(r.total.values()) for r in rates)

            # logger.info(f'Rates at step {n}: {rates}')
            # logger.info(f'Total rate at step {n}: {total_rate}')

            if total_rate <= 0:
                logger.info(f'No reactions possible, exiting simulation : total_rate = {total_rate}')
                return None
            reaction_choice = self._choose_reaction(rates)
            react_rates = rates[reaction_choice.nuc_idx].persite

            # Perform reaction
            self.perform_reaction(reaction_choice,
                                    persite=react_rates)


            # advance reaction time
            u_ = self._uniform_pos_arg()
            dt = np.log(1 / u_) / total_rate
            self.t += dt

            # Update the species count
            self._update_species_count(reaction_choice)

            yield SimulationState(  time            =self.t,
                                    cs              =self.nuc.N_closed,
                                    bprot           =self.prot.N_bound,
                                    nucs_snapshot   =self.nuc.nuc_state)





if __name__ == "__main__":



    # logger.info(f"int_to_tri14(45627): {int_to_tri14(45627)}")
    # logger.info(f"tri14_to_int('11110000000000'): {tri14_to_int('11110000000000')}")

    # new_value = edit_tricodec(value=45627, position=3, new_digit=2)
    # logger.info(f"edit_tricodec(45627, 3, 2): {new_value}")
    # logger.info(f"int_to_tri14(edit_tricodec(45627, 3, 2)): {int_to_tri14(new_value)}")

    # import sys
    # sys.exit()

    nuc_instance = nucleosome(k_unwrap=4.0,
                                   k_wrap=21.0,
                                   num_nucleosomes=2,
                                   binding_sites=14)

    protamines_instance = protamines(k_unbind=213,
                                        k_bind=123,
                                        p_conc=1.0,
                                        cooperativity=2.0)

    simulation = GillespieSimulator(nuc_inst=nuc_instance,
                                    prot_inst=protamines_instance,
                                    num_nucleosomes=2,
                                    STEPS=10)

    for state in simulation.run():
        logger.info(state)

    logger.info('Simulation completed')
