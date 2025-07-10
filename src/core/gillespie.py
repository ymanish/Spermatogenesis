#/src/core/gillespie.py
import numpy as np
import math
import csv

from dataclasses import dataclass
from typing import List
import os
import sys
import datetime
import argparse



@dataclass
class SimulationState:
    time: float
    closed_sites: int
    bound_protamines: int
    nucleosome_states: List[np.ndarray]


class GillespieSimulator():
    def __init__(self, nucleosme_instance, protamines_instance,  num_nucleosomes, N=1000, ONE_NUCLEOSOME_BREATHING=False):
        self.nucleosme = nucleosme_instance
        self.protamines = protamines_instance
        self.N = N # Number of steps
        self.t = 0
        self.num_nucleosomes = num_nucleosomes
        self.nuc_fall_flag = False
        self.only_one_nucleosome_breathing = ONE_NUCLEOSOME_BREATHING
        # self.uniform_pos_arg_njit = nb.njit(self.uniform_pos_arg)



    def batch_write_derivates(self, TM, N_Cl, N_Op, P_molec, N_B, NS, counter):
        # Write Nucleosome_state array to .npy file
        # print(NS)
        # print(N_Op)
        np.save(NUCLEOSOME_STATE_RECORD_DIR + f'Nucleosome_state_{counter}.npy', NS)

        # Open the file in write mode
        with open(NUCLEOSOME_STATE_RECORD_DIR + f"Deriavte_output_{counter}.csv", 'w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            writer.writerow(['Time', 'N_closed', 'N_open', 'P_free', 'N_bound'])
            
            # Write the arrays to the file
            for row in zip(TM, N_Cl, N_Op, P_molec, N_B):
                writer.writerow(row)
        
        return None

    def calculate_rates(self, nucleo):
        histone_occupied_indexes = np.ravel(np.where(nucleo == 0))
        protein_bound_indexes = np.ravel(np.where(nucleo == 2))
        open_indexes = np.ravel(np.where(nucleo == 1))


        unwrapping_sites_rate, rate_open = self.nucleosme.unwrapping(histone_occupied_indexes)

        rewrapping_sites_rate, rate_close = self.nucleosme.rewrapping(nucleo, histone_occupied_indexes, keep_histone=self.only_one_nucleosome_breathing)


        bound_sites_rate, rate_bind =  self.protamines.protein_binding(open_indexes)

        # unbound_sites_rate, rate_unbind =  self.protamines.protein_unbinding(protein_bound_indexes)
        unbound_sites_rate, rate_unbind =  self.protamines.protein_unbinding_coop(nucleo, protein_bound_indexes)

        # print(nucleo)
        # print('Unwrapping rate', unwrapping_sites_rate)
        # print('Rewrapping rate', rewrapping_sites_rate)
        # print('Binding rate', bound_sites_rate)
        # print('Unbinding rate', unbound_sites_rate)

        return rate_open, rate_close, rate_bind, rate_unbind, \
               unwrapping_sites_rate, rewrapping_sites_rate,\
               bound_sites_rate, unbound_sites_rate

    def uniform_pos_arg(self):
        return np.random.uniform(0.0, 1.0)

    def perform_reaction(self, nucleo, react_id, rates):
        # nucleosome is a list of sites, with some value (say, 0 for closed, 1 for open/unbound, 2 for open/bound)

        rates_value = list(rates[react_id + 4].values())
        rates_keys = list(rates[react_id + 4].keys())

        if react_id == 0:  # Opening reaction
            # Choose a closed site to open
            #         closed_sites = np.where(nucleosome == 0)

            site_to_open_ind = np.random.choice(len(rates_value),
                                                p=np.divide(rates_value,
                                                            sum(rates_value)))

            #         site_to_open = closed_sites[0][0]
            nucleo[rates_keys[site_to_open_ind]] = 1

        elif react_id == 1:  # Closing reaction
            # Choose an open/unbound site to close
            #         closed_sites = np.where(nucleosome == 0)

            site_to_close_ind = np.random.choice(len(rates_value),
                                                 p=np.divide(rates_value,
                                                             sum(rates_value)))

            #         site_to_close = (closed_sites[0][0])-1
            nucleo[rates_keys[site_to_close_ind]] = 0

        elif react_id == 2:  # Binding reaction
            # Choose an open/unbound site for protamine to bind
            #         unbound_sites = np.where(nucleosome == 1)

            site_to_bind_ind = np.random.choice(len(rates_value),
                                                p=np.divide(rates_value,
                                                            sum(rates_value)))

            #         site_to_bind = np.random.choice(unbound_sites[0])
            nucleo[rates_keys[site_to_bind_ind]] = 2

        elif react_id == 3:  # Unbinding reaction
            # Choose an open/bound site for protamine to unbind
            #         bound_sites = np.where(nucleosome == 2)
            site_to_unbind_ind = np.random.choice(len(rates_value),
                                                  p=np.divide(rates_value,
                                                              sum(rates_value)))
            nucleo[rates_keys[site_to_unbind_ind]] = 1

        else:
            raise ValueError('Invalid reaction index')
        return nucleo



    def simulate_main(self):

        
        times = []
        N_closed_array = []
        N_open_array = []
        P_free_array = []
        N_bound_array = []
        Nucleosome_state = []
        nucleosome_fall_time = []
        ft_nucleosome_fell = np.nan

        for n in range(1, self.N+1):
            # Calculate total number of unbound sites in all nucleosomes
            # total_unbound_sites = func_cnt(system_vars.nucleosomes, ele=1)
            # print(n)
            # print(self.p_conc)
            # print(self.P_free)

            all_rates = [self.calculate_rates(nucleo=nucleosome) for nucleosome in self.nucleosme.nucleosomes]
            # print(self.nucleosomes)
            # print(all_rates)
            ### We select the first four values because the structure for a entry in all_rates contains a tuple.
            # The first four are the rates sum of all the rates for binding, unbinding, unwrap, wrap and the remaining four are the dictionary for their corresponding rates
            #Example: (8, 21, 720000000.0, 0, {0: 4, 12: 4}, {13: 21}, {13: 720000000.0}, {})
            total_rate = sum([sum(rates[:4]) for rates in all_rates])

            # Calculate time to next reaction
            u_ = self.uniform_pos_arg()
            dt = np.log(1 / u_) / total_rate
            # print(dt)
            self.t += dt

            # Select nucleosome and reaction to perform
            nucleosome_idx = np.random.choice(self.num_nucleosomes,
                                                p=[sum(rates[:4]) / total_rate for rates in all_rates])
            
            # print(np.divide(list(all_rates[nucleosome_idx][:4]),sum(all_rates[nucleosome_idx][:4])))

            reaction_idx = np.random.choice(len(all_rates[0][:4]), p=np.divide(list(all_rates[nucleosome_idx][:4]),
                                                                                sum(all_rates[nucleosome_idx][:4])))
            reaction_rates = all_rates[nucleosome_idx]

            # Perform reaction
            updated_nucleosome = self.perform_reaction(nucleo = self.nucleosme.nucleosomes[nucleosome_idx],
                                                        react_id = reaction_idx,
                                                        rates=reaction_rates)

            self.nucleosme.nucleosomes[nucleosome_idx] = updated_nucleosome

            # print(self.nucleosme.nucleosomes[nucleosome_idx])

            # Update protamine concentration if necessary

            if reaction_idx == 0:
                # Open site
                self.nucleosme.N_closed -= 1
                self.nucleosme.N_open += 1
            elif reaction_idx == 1:
                # Close site
                self.nucleosme.N_closed += 1
                self.nucleosme.N_open -= 1
            elif reaction_idx == 2:
                # Protamine binds
                self.nucleosme.N_open -= 1
                # self.protamines.P_free -= 1
                self.protamines.N_bound += 1
            else:
                # Protamine unbinds
                self.nucleosme.N_open += 1
                # self.protamines.P_free += 1
                self.protamines.N_bound -= 1
            # print('State of the simulation step ', self.P_free, self.N_bound, self.N_open)
            # print('Nucleosome:', self.nucleosomes[nucleosome_idx])
            
            if len(np.where(self.nucleosme.nucleosomes[nucleosome_idx] == 0)[0])==0 and (self.nucleosme.record_state[nucleosome_idx] == 1):
                ft_nucleosome_fell = self.t
                self.nucleosme.record_state[nucleosome_idx]=0
                nucleosome_fall_time.append(self.t)

                print('Step and Time at which nucleosome fell ', n, self.t)

                if np.count_nonzero(self.nucleosme.record_state == 1) == 0:
                    print(self.nucleosme.record_state)
                    # print('All nucleosomes have fallen')
                    self.nuc_fall_flag = True

                # if self.only_one_nucleosome_breathing:
                #     self.nuc_fall_flag = True
            # print('The nucleosome state ', np.where(self.nucleosme.nucleosomes[nucleosome_idx] == 0)[0])
            # print(self.nucleosme.nucleosomes[nucleosome_idx])

            times.append(self.t)
            N_closed_array.append(self.nucleosme.N_closed)
            # N_open_array.append(self.nucleosme.N_open)
            # P_free_array.append(self.protamines.P_free)
            N_bound_array.append(self.protamines.N_bound)
            Nucleosome_state.append(np.concatenate(self.nucleosme.nucleosomes))
            

            if self.nuc_fall_flag and not self.only_one_nucleosome_breathing:
                print('All nucleosomes have fallen')
                return nucleosome_fall_time, N_closed_array, N_bound_array, times, self.nucleosme.nucleosomes
        # return nucleosome_fall_time, N_closed_array, N_bound_array, times, self.nucleosme.nucleosomes
        return SimulationState(time=self.t, closed_sites=self.nucleosme.N_closed, bound_protamines=self.protamines.N_bound, nucleosome_states=self.nucleosme.nucleosomes)

