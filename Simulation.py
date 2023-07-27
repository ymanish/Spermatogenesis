import sys

import pandas as pd
import numpy as np
import random
# random.seed(a = 1)
# from numba import njit, types
# import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt
import math
import concurrent.futures
import time
from statistics import mean


def uniform_pos_arg():
    return np.random.uniform(0.0, 1.0)

# uniform_pos_arg_njit = nb.njit(uniform_pos_arg)


def func_cnt(arrs, ele):
    tot_zero_els = 0
    for arr in arrs:
        zero_els = np.count_nonzero(arr==ele)
        tot_zero_els = tot_zero_els + zero_els
    return tot_zero_els


def site_opening(site_indexes, k_open):
    cl_sites_can_open = dict()
    sum_rate = 0

    if len(site_indexes[0]) == 0:
        cl_sites_can_open = dict()

    elif site_indexes[0][0] != site_indexes[0][-1]:
        cl_sites_can_open[site_indexes[0][0]] = k_open
        cl_sites_can_open[site_indexes[0][-1]] = k_open

    else:
        cl_sites_can_open[site_indexes[0][0]] = k_open

    if len(cl_sites_can_open) > 0:
        sum_rate = sum(cl_sites_can_open.values())

    return cl_sites_can_open, sum_rate


def site_closing(nuce, site_indexes, k_close):
    unb_sites_can_close = dict()
    sum_rate = 0
    if len(site_indexes[0]) > 0:
        try:
            assert site_indexes[0][0] - 1 >= 0
            if nuce[site_indexes[0][0] - 1] == 1:
                unb_sites_can_close[site_indexes[0][0] - 1] = k_close

        except (AssertionError, IndexError) as msg:
            pass

        try:
            if nuce[site_indexes[0][-1] + 1] == 1:
                unb_sites_can_close[site_indexes[0][-1] + 1] = k_close
        except IndexError:
            pass

    if len(unb_sites_can_close) > 0:
        sum_rate = sum(unb_sites_can_close.values())

    return unb_sites_can_close, sum_rate


def unbound_site(nuce, op_indexes, pt_indexes, k_bind, cf, P):
    unb_sites = dict()
    # neigh_dict=dict()
    sum_rate = 0
    if len(op_indexes[0]) != 0:

        if len(pt_indexes[0]) != 0:

            for i in op_indexes[0]:
                neig_count = 0
                ###count for the neighbors
                if max(i - 1, 0) in pt_indexes[0]:
                    k = i - 1
                    while nuce[max(k, 0)] == 2 and k >= 0:
                        neig_count += 1
                        k = k - 1

                if i + 1 in pt_indexes[0]:
                    k = i + 1
                    while nuce[min(k, 13)] == 2 and k <= 13:
                        neig_count += 1
                        k = k + 1
                #                 print(P)
                unb_sites[i] = k_bind * (1 + cf * neig_count) * P
        #             neigh_dict[i] = neig_count
        else:
            for i in op_indexes[0]:
                unb_sites[i] = k_bind

    if len(unb_sites) > 0:
        sum_rate = sum(unb_sites.values())

    return unb_sites, sum_rate


def bound_site(pt_indexes, k_unbind):
    bd_sites = dict()
    sum_rate = 0

    if len(pt_indexes[0]) != 0:
        for i in pt_indexes[0]:
            bd_sites[i] = k_unbind

    if len(bd_sites) > 0:
        sum_rate = sum(bd_sites.values())

    return bd_sites, sum_rate


def calculate_rates(nucleosome, P_free, all_unbound_sites, k_open, k_close, k_unbind, k_bind, cf):
    # nucleosome is a list of sites, with some value (say, 0 for closed, 1 for open/unbound, 2 for open/bound)

    # Calculate only those sites that can be opened at a time, with their rates.
    #     closed_sites = np.count_nonzero(nucleosome==0)

    cl_site_indexes = np.where(nucleosome == 0)
    pt_site_indexes = np.where(nucleosome == 2)
    op_sites_indexes = np.where(nucleosome == 1)

    closed_sites_can_open_rate, rate_open = site_opening(cl_site_indexes, k_open)

    unbound_sites_can_close_rate, rate_close = site_closing(nucleosome, cl_site_indexes, k_close)

    unbound_sites_rate, rate_bind = unbound_site(nucleosome, op_sites_indexes, pt_site_indexes, k_bind, cf, P_free)

    bound_sites_rate, rate_unbind = bound_site(pt_site_indexes, k_unbind)

    #     print(closed_sites, unbound_sites, bound_sites)
    # Calculate rate of each type of reaction
    #     rate_open = sum(closed_sites_can_open_rates.values)
    #     rate_close = system_vars.k_close * unbound_sites_can_close

    #     # Binding rate depends on the fraction of total unbound sites that are in this nucleosome
    #     if all_unbound_sites !=0:
    #         rate_bind = system_vars.k_bind *unbound_sites* P_free
    #     else:
    #         rate_bind = 0

    #     rate_unbind = system_vars.k_unbind * bound_sites

    return rate_open, rate_close, rate_bind, rate_unbind, closed_sites_can_open_rate, unbound_sites_can_close_rate, unbound_sites_rate, bound_sites_rate


def perform_reaction(nucleosome, reaction_idx, rates):
    # nucleosome is a list of sites, with some value (say, 0 for closed, 1 for open/unbound, 2 for open/bound)

    rates_value = list(rates[reaction_idx + 4].values())
    rates_keys = list(rates[reaction_idx + 4].keys())

    if reaction_idx == 0:  # Opening reaction
        # Choose a closed site to open
        #         closed_sites = np.where(nucleosome == 0)

        site_to_open_ind = np.random.choice(len(rates_value),
                                            p=np.divide(rates_value,
                                                        sum(rates_value)))

        #         site_to_open = closed_sites[0][0]
        nucleosome[rates_keys[site_to_open_ind]] = 1

    elif reaction_idx == 1:  # Closing reaction
        # Choose an open/unbound site to close
        #         closed_sites = np.where(nucleosome == 0)

        site_to_close_ind = np.random.choice(len(rates_value),
                                             p=np.divide(rates_value,
                                                         sum(rates_value)))

        #         site_to_close = (closed_sites[0][0])-1
        nucleosome[rates_keys[site_to_close_ind]] = 0

    elif reaction_idx == 2:  # Binding reaction
        # Choose an open/unbound site for protamine to bind
        #         unbound_sites = np.where(nucleosome == 1)

        site_to_bind_ind = np.random.choice(len(rates_value),
                                            p=np.divide(rates_value,
                                                        sum(rates_value)))

        #         site_to_bind = np.random.choice(unbound_sites[0])
        nucleosome[rates_keys[site_to_bind_ind]] = 2

    elif reaction_idx == 3:  # Unbinding reaction
        # Choose an open/bound site for protamine to unbind
        #         bound_sites = np.where(nucleosome == 2)
        site_to_unbind_ind = np.random.choice(len(rates_value),
                                              p=np.divide(rates_value,
                                                          sum(rates_value)))
        nucleosome[rates_keys[site_to_unbind_ind]] = 1

    else:
        raise ValueError('Invalid reaction index')
    return nucleosome



def get_mean_value(array_ALL, time_run=False):
    temp_array = []

    for k in array_ALL:

        temp_array.append(k[-1])

    return mean(temp_array)


class system_init:
    def __init__(self, c, k_open, k_close, k_bind, k_unbind, cooperativity_factor):
        self.P_concentration = c * math.pow(10, -6)  # in moles/liters
        self.num_nucleosomes = 1  # number of nucleosomes

        self.N_closed = self.num_nucleosomes * 14
        self.N_open = 0
        self.P_free = self.P_concentration * 6 * math.pow(10, 23) * math.pow(10, -15)  # initial concentration of protamines molecules/cubic_micrometer
        self.N_bound = 0

        print(self.P_free)

        # Define rate constants
        self.k_open = k_open  # rate constant for sites opening
        self.k_close = k_close  # rate constant for sites closing
        self.k_bind = k_bind  # rate constant for protamines binding
        self.k_unbind = k_unbind  # rate constant for protamines unbinding

        self.N = 200
        self.cooperativity_factor = cooperativity_factor

        self.nucleosomes = [np.zeros(14, dtype=int) for _ in range(self.num_nucleosomes)]

#         self.t_max = 100
        # Initialize time
        self.t = 0

        # Initialize arrays to store time evolution of the system
        self.times = [0]
        self.N_closed_array = [self.N_closed]
        self.N_open_array = [self.N_open]
        self.P_free_array = [self.P_free]
        self.N_bound_array = [self.N_bound]

        self.State_array = np.array(self.nucleosomes)


# Main loop
def main(my_conc, con_n,  k_open, k_close, k_bind, k_unbind, cop_factor, time_run=False):
    times_ALL = []
    N_closes_array_ALL = []
    N_open_array_ALL = []
    P_free_array_ALL = []
    N_bound_array_ALL = []

    N_Iteration = 10

    for trac in range(N_Iteration):
        print('Iteration number ', trac)

        system_vars = system_init(my_conc, k_open=k_open, k_close=k_close,
                                  k_unbind=k_unbind, k_bind=k_bind, cooperativity_factor=cop_factor)
        #         my_conc = my_conc*math.pow(10, -6)
        #         system_init.P_free = my_conc * 6 * math.pow(10, 23) * math.pow(10, -15)
        ###I want to define the mayclass oject here with the same values for each loop

        for n in range(1, system_vars.N):
            #         while system_vars.t < system_vars.t_max:

            # Calculate total number of unbound sites in all nucleosomes
            total_unbound_sites = func_cnt(system_vars.nucleosomes, ele=1)
            #     print(total_unbound_sites)
            #     print('NEW-----------------------')
            # Calculate reaction rates for all nucleosome
            #         print(system_vars.nucleosomes)

            all_rates = [calculate_rates(nucleosome, system_vars.P_free, total_unbound_sites,
                                         system_vars.k_open, system_vars.k_close, system_vars.k_unbind,
                                         system_vars.k_bind,
                                         system_vars.cooperativity_factor)
                         for nucleosome in system_vars.nucleosomes]

            # Calculate total rate
            total_rate = sum([sum(rates[:4]) for rates in all_rates])

            # Calculate time to next reaction
            u_ = uniform_pos_arg()
            dt = np.log(1 / u_) / total_rate
            system_vars.t += dt

            # Select nucleosome and reaction to perform
            nucleosome_idx = np.random.choice(system_vars.num_nucleosomes,
                                              p=[sum(rates[:4]) / total_rate for rates in all_rates])

            reaction_idx = np.random.choice(len(all_rates[0][:4]), p=np.divide(list(all_rates[nucleosome_idx][:4]),
                                                                               sum(all_rates[nucleosome_idx][:4])))
            reaction_rates = all_rates[nucleosome_idx]

            # Perform reaction
            updated_nucleosome = perform_reaction(system_vars.nucleosomes[nucleosome_idx], reaction_idx, reaction_rates)

            system_vars.nucleosomes[nucleosome_idx] = updated_nucleosome

            # Update protamine concentration if necessary

            if reaction_idx == 0:
                # Open site
                system_vars.N_closed -= 1
                system_vars.N_open += 1
            elif reaction_idx == 1:
                # Close site
                system_vars.N_closed += 1
                system_vars.N_open -= 1
            elif reaction_idx == 2:
                # Protamine binds
                system_vars.N_open -= 1
                system_vars.P_free -= 1
                system_vars.N_bound += 1
            else:
                # Protamine unbinds
                system_vars.N_open += 1
                system_vars.P_free += 1
                system_vars.N_bound -= 1

            # Record state if necessary
            # system_vars.times.append(system_vars.t)
            # system_vars.N_closed_array.append(system_vars.N_closed)
            # system_vars.N_open_array.append(system_vars.N_open)
            # system_vars.P_free_array.append(system_vars.P_free)
            # system_vars.N_bound_array.append(system_vars.N_bound)
        #     State_array = np.vstack([State_array, np.array(nucleosomes)])

        # times_ALL.append(system_vars.times)
        # N_closes_array_ALL.append(system_vars.N_closed_array)
        # N_open_array_ALL.append(system_vars.N_open_array)
        # P_free_array_ALL.append(system_vars.P_free_array)
        # N_bound_array_ALL.append(system_vars.N_bound_array)

        times_ALL.append(system_vars.t)
        N_closes_array_ALL.append(system_vars.N_closed)
        N_open_array_ALL.append(system_vars.N_open)
        P_free_array_ALL.append(system_vars.P_free)
        N_bound_array_ALL.append(system_vars.N_bound)

    # times_ALL = np.array(times_ALL)
    # N_closes_array_ALL = np.array(N_closes_array_ALL)
    # N_bound_array_ALL = np.array(N_bound_array_ALL)
    # N_open_array_ALL = np.array(N_open_array_ALL)


    bound_mean = mean(N_bound_array_ALL)
    open_mean = mean(N_open_array_ALL)
    closed_mean = mean(N_closes_array_ALL)
    time_mean = mean(times_ALL)


    # bound_mean = get_mean_value(N_bound_array_ALL, time_run=False)
    # open_mean = get_mean_value(N_open_array_ALL, time_run=False)
    # closed_mean = get_mean_value(N_closes_array_ALL, time_run=False)
    # if time_run:
    #     time_mean = system_vars.t_max
    # else:
    #     time_mean = get_mean_value(times_ALL, time_run=False)
    #
    # with open(RESULT_DIR + con_n + '_bound.txt', 'wb') as f:
    #     np.savetxt(f, N_bound_array_ALL)
    #
    # with open(RESULT_DIR + con_n + '_open.txt', 'wb') as f:
    #     np.savetxt(f, N_open_array_ALL)
    #
    # with open(RESULT_DIR + con_n + '_times.txt', 'wb') as f:
    #     np.savetxt(f, times_ALL)
    #
    # with open(RESULT_DIR + con_n + '_closed.txt', 'wb') as f:
    #     np.savetxt(f, N_closes_array_ALL)

    return bound_mean, open_mean, closed_mean, time_mean, my_conc




if __name__== '__main__':

    start = time.perf_counter()

    concentration_dict = {'1': 0.1, '2': 0.2, '3': 0.4, '4': 0.7, '14': 0.8,
                          '15': 0.9, '5': 1.0,
                          '6': 1.3, '7': 1.6, '8': 1.9, '9': 2.2, '10': 2.5,
                          '11': 2.8, '12': 3.1, '13': 3.5, '16': 4.0,
                          '17':4.2, '18':5.0}


    Sim_data = pd.DataFrame(columns=['C', 'bound_prot', 'nuc_open', 'nuc_closed'])
    shape_list = dict()
    temp_bound = []
    temp_open = []
    temp_closed = []
    end_time = []
    con = []

    param_n = sys.argv[1]
    # RESULT_DIR = r"/group/cmcb-files/pol_schiessel/05_Projekte/manish/Spermatogensis/results_26_6/" + str(param_n)+"/"
    RESULT_DIR = r"C:\Users\maya620d\PycharmProjects\Spermatogensis\results\/"


    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = []
        for i, (key, value) in enumerate(concentration_dict.items()):
            print(key, value)
            pool.append(executor.submit(main, my_conc=value, con_n=key, k_open=21*float(sys.argv[2]),
                                        k_close=21, k_bind=0.59,
                                        k_unbind=241.59, cop_factor=float(sys.argv[3])))

        for j in concurrent.futures.as_completed(pool):
            bound_value, open_value, closed_value, time_value, con_value = j.result()

            temp_bound.append(bound_value)

            temp_open.append(open_value)
            temp_closed.append(closed_value)
            end_time.append(time_value)
            con.append(con_value)

            # shape_list[key] = dat_shape
            ### Call a function to draw the graph between number of bounds and time


    Sim_data['C'] = con
    Sim_data['bound_prot'] = temp_bound
    Sim_data['nuc_open'] = temp_open
    Sim_data['nuc_closed'] = temp_closed
    Sim_data['end_time'] = end_time
    print(Sim_data)
    Sim_data.to_csv(RESULT_DIR + str(param_n)+"_param.csv")


    end = time.perf_counter()
    print(f'Finished in {round(end - start, 2)} second(s)')
    print(f'Parameter {param_n} is done>>>>>>>>>>>>>>>>>>>>')
