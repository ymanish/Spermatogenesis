import pandas as pd
import numpy as np
import random
# random.seed(a = 1)
from numba import njit, types
import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt
import math
from statistics import mean


def get_mean_value(array_ALL, time_run=False):
    temp_array = []
    for k in array_ALL:
        temp_array.append(k[-1])
    return mean(temp_array)

class nucleosme:
    def __init__(self, k_unwrap, k_wrap, num_nucleosomes, binding_sites=14):
        self.k_unwrap = k_unwrap
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        self.nucleosomes = [np.zeros(self.binding_sites, dtype=int) for _ in range(num_nucleosomes)]
        self.N_closed = num_nucleosomes * binding_sites
        self.N_open = 0

    def sequence(self):
        pass
    def breathing_rates(self):
        pass

    def site_opening(self, close_site_ind):
        cl_sites_can_open = dict()
        sum_rate = 0

        if len(close_site_ind[0]) == 0:
            cl_sites_can_open = dict()

        elif close_site_ind[0][0] != close_site_ind[0][-1]:
            cl_sites_can_open[close_site_ind[0][0]] = self.k_unwrap
            cl_sites_can_open[close_site_ind[0][-1]] = self.k_unwrap

        else:
            cl_sites_can_open[close_site_ind[0][0]] = self.k_unwrap

        if len(cl_sites_can_open) > 0:
            sum_rate = sum(cl_sites_can_open.values())

        return cl_sites_can_open, sum_rate

    def site_closing(self, nuce, site_indexes):
        unb_sites_can_close = dict()
        sum_rate = 0
        if len(site_indexes[0]) > 0:
            try:
                assert site_indexes[0][0] - 1 >= 0
                if nuce[site_indexes[0][0] - 1] == 1:
                    unb_sites_can_close[site_indexes[0][0] - 1] = self.k_wrap

            except (AssertionError, IndexError) as msg:
                pass

            try:
                if nuce[site_indexes[0][-1] + 1] == 1:
                    unb_sites_can_close[site_indexes[0][-1] + 1] = self.k_wrap
            except IndexError:
                pass

        if len(unb_sites_can_close) > 0:
            sum_rate = sum(unb_sites_can_close.values())

        return unb_sites_can_close, sum_rate




class protamines:
    def __init__(self, k_unbind, k_bind, p_conc, cooperativity=1):

        """defines the unbinding rate of protamine for each bound site
        :param k_unbind: unbinding rate
        :param k_bind: binding rate
        :param p_conc: protamine concentration in microMolar
        :param cooperativity: cooperativity factor"""



        self.k_unbind = k_unbind
        self.k_bind = k_bind
        self.cooperativity = cooperativity
        self.p_conc = p_conc
        self.P_free = self.p_conc * 6 * math.pow(10, 23) * math.pow(10, -15)  # initial concentration of protamines molecules/cubic_micrometer
        self.N_bound = 0


    def prot_rates(self):
        print('lrATEF')
        pass

    def unbound_site(self, nuce, op_indexes, pt_indexes):
        """defines the binding rate of protamine for each unbound site
        :param op_indexes: contains the indexes for the open unbound sites
        :param pt_indexes: contains the indexes for the protamine bound site
         :return: dictionary with key as the index of the unbound sites and value as its binding rate.
                    Also returns the sum of all the binding rates"""

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

                    unb_sites[i] = self.k_bind  * (1 + self.cooperativity * neig_count) * self.P_free
            #             neigh_dict[i] = neig_count
            else:
                for i in op_indexes[0]:
                    unb_sites[i] = self.k_bind* self.P_free

        if len(unb_sites) > 0:
            sum_rate = sum(unb_sites.values())

        return unb_sites, sum_rate



    def bound_site(self, pt_indexes):
        """defines the unbinding rate of protamine for each bound site
        :param pt_indexes: contains the indexes for the protamine bound site
         :return: dictionary with key as the index of the bound sites and value as its unbinding rate.
                    Also returns the sum of all the unbinding rates"""

        bd_sites = dict()
        sum_rate = 0

        if len(pt_indexes[0]) != 0:
            for i in pt_indexes[0]:
                bd_sites[i] = self.k_unbind

        if len(bd_sites) > 0:
            sum_rate = sum(bd_sites.values())

        return bd_sites, sum_rate


class Simulation(nucleosme, protamines):
    def __init__(self, k_unwrap, k_wrap, k_unbind, k_bind, p_conc, num_nucleosomes, cooperativity=1, binding_sites=14, N=20):

        nucleosme.__init__(self, k_unwrap, k_wrap, num_nucleosomes, binding_sites)
        protamines.__init__(self, k_unbind, k_bind, p_conc, cooperativity)

        self.N = N
        # self.uniform_pos_arg_njit = nb.njit(self.uniform_pos_arg)
        self.t = 0
        self.num_nucleosomes =  num_nucleosomes


        # self.p_conc = p_conc
        # self.k_unwrap = k_unwrap
        # self.k_wrap = k_wrap
        # self.binding_sites = binding_sites
        # self.k_unbind = k_unbind
        # self.k_bind = k_bind
        # self.cooperativity = cooperativity


    def simulate_main(self):
        times = []
        N_closed_array = []
        N_open_array = []
        P_free_array = []
        N_bound_array = []

        for n in range(1, self.N):
            # Calculate total number of unbound sites in all nucleosomes
            # total_unbound_sites = func_cnt(system_vars.nucleosomes, ele=1)
            # print(n)
            # print(self.p_conc)
            # print(self.P_free)

            all_rates = [self.calculate_rates(nucleo=nucleosome) for nucleosome in self.nucleosomes]
            # print(all_rates)
            ### We select the first four values because the structure for a entry in all_rates contains a tuple.
            # The first four are the rates sum of all the rates for binding, unbinding, unwrap, wrap and the remaining four are the dictionary for their corresponding rates
            #Example: (8, 21, 720000000.0, 0, {0: 4, 12: 4}, {13: 21}, {13: 720000000.0}, {})
            total_rate = sum([sum(rates[:4]) for rates in all_rates])

            # Calculate time to next reaction
            u_ = self.uniform_pos_arg()
            dt = np.log(1 / u_) / total_rate
            self.t += dt

            # Select nucleosome and reaction to perform
            nucleosome_idx = np.random.choice(self.num_nucleosomes,
                                              p=[sum(rates[:4]) / total_rate for rates in all_rates])

            reaction_idx = np.random.choice(len(all_rates[0][:4]), p=np.divide(list(all_rates[nucleosome_idx][:4]),
                                                                               sum(all_rates[nucleosome_idx][:4])))
            reaction_rates = all_rates[nucleosome_idx]

            # Perform reaction
            updated_nucleosome = self.perform_reaction(nucleo = self.nucleosomes[nucleosome_idx],
                                                       react_id = reaction_idx,
                                                       rates=reaction_rates)

            self.nucleosomes[nucleosome_idx] = updated_nucleosome

            # print(self.nucleosomes[nucleosome_idx])

            # Update protamine concentration if necessary

            if reaction_idx == 0:
                # Open site
                self.N_closed -= 1
                self.N_open += 1
            elif reaction_idx == 1:
                # Close site
                self.N_closed += 1
                self.N_open -= 1
            elif reaction_idx == 2:
                # Protamine binds
                self.N_open -= 1
                self.P_free -= 1
                self.N_bound += 1
            else:
                # Protamine unbinds
                self.N_open += 1
                self.P_free += 1
                self.N_bound -= 1



            # Record state if necessary
            times.append(self.t)
            N_closed_array.append(self.N_closed)
            N_open_array.append(self.N_open)
            P_free_array.append(self.P_free)
            N_bound_array.append(self.N_bound)

        return times, N_closed_array, N_open_array, P_free_array, N_bound_array





    def calculate_rates(self, nucleo):
        cl_site_indexes = np.where(nucleo == 0)
        pt_site_indexes = np.where(nucleo == 2)
        op_sites_indexes = np.where(nucleo == 1)

        closed_sites_can_open_rate, rate_open = self.site_opening(cl_site_indexes)

        unbound_sites_can_close_rate, rate_close = self.site_closing(nucleo, cl_site_indexes)

        unbound_sites_rate, rate_bind = self.unbound_site(nucleo, op_sites_indexes, pt_site_indexes)

        bound_sites_rate, rate_unbind = self.bound_site(pt_site_indexes)

        return rate_open, rate_close, rate_bind, rate_unbind, \
               closed_sites_can_open_rate, unbound_sites_can_close_rate,\
               unbound_sites_rate, bound_sites_rate

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


if __name__== '__main__':
    times_ALL = []
    N_closes_array_ALL = []
    N_open_array_ALL = []
    P_free_array_ALL = []
    N_bound_array_ALL = []
    N_cycles = 2 ## Number of nucleosomes to run for the simulation

    Sim_data = pd.DataFrame(columns=['C', 'bound_prot', 'nuc_open', 'nuc_closed'])
    shape_list = dict()
    temp_bound = []
    temp_open = []
    temp_closed = []
    end_time = []
    con = []
    #

    for cycles in range(N_cycles):
        s1 = Simulation(k_unwrap=4, k_wrap=21, k_unbind=23,  k_bind=2, p_conc=0.6,  num_nucleosomes=1, N=3)
        times, N_closed_array, N_open_array, P_free_array, N_bound_array = s1.simulate_main()
        # times_ALL.append(times)
        # N_closes_array_ALL.append(N_closed_array)
        # N_open_array_ALL.append(N_open_array)
        # P_free_array_ALL.append(P_free_array)
        # N_bound_array_ALL.append(N_bound_array)
        #
        # times_ALL = np.array(times_ALL)
        # N_closes_array_ALL = np.array(N_closes_array_ALL)
        # N_bound_array_ALL = np.array(N_bound_array_ALL)
        # N_open_array_ALL = np.array(N_open_array_ALL)

        temp_bound.append(N_bound_array[-1])
        temp_open.append(N_open_array[-1])
        temp_closed.append(N_closed_array[-1])
        end_time.append(times[-1])

    bound_mean = mean(N_bound_array)
    open_mean = mean(N_open_array)
    closed_mean = mean(N_closed_array)
    time_mean = mean(end_time)
    print(N_bound_array_ALL)


