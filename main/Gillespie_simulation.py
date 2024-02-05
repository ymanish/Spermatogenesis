import pandas as pd
import numpy as np
import random
# random.seed(a = 1)
#from numba import njit, types
#import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt
import math
from statistics import mean
from initialise import *
import concurrent.futures
import time
from operator import add

def get_mean_value(array_ALL, time_run=False):
    temp_array = []
    for k in array_ALL:
        temp_array.append(k[-1])
    return mean(temp_array)

class nucleosme:

    Olson_prob_df_di = None
    Olson_prob_df_tri = None
    Olson_prob_df_mono = None
    MD_param_df = None
    SAXS_df = None
    stationary_array = None

    def __init__(self, k_unwrap, k_wrap, num_nucleosomes, binding_sites=14):
        self.k_unwrap = k_unwrap
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        self.nucleosomes = [np.zeros(self.binding_sites, dtype=int) for _ in range(num_nucleosomes)]
        self.N_closed = num_nucleosomes * binding_sites
        self.N_open = 0
        self.binding_bp_right = [7, 18, 30, 39, 50, 60, 70, 81, 91, 101, 112, 122, 132, 146]
        self.binding_bp_left = [0, 14, 24, 34, 45, 55, 65, 76, 86, 96, 107, 116, 128, 139]

        if self.Olson_prob_df_di is None:
            self.Olson_prob_df_di = pd.read_csv(Olson_prob_di_loc)
            self.DI_PROB_dict = self.Olson_prob_df_di.to_dict()

        if self.Olson_prob_df_mono is None:
            self.Olson_prob_df_mono = pd.read_csv(Olson_prob_mono_loc)
            self.MONO_PROB_dict = self.Olson_prob_df_mono.to_dict()

        if self.Olson_prob_df_tri is None:
            self.Olson_prob_df_tri = pd.read_csv(Olson_prob_tri_loc)

        if self.MD_param_df is None:
            self.MD_param_df = pd.read_csv(MD_param_loc)
            self.MD_param_df.rename(columns={'Unnamed: 0': 'steps'}, inplace=True)

        if self.SAXS_df is None:
            self.SAXS_df = pd.read_csv(SAXS_loc)

        if self.stationary_array is None:
            self.stationary_array = np.loadtxt(SAXS_DIST)
            
    def sequence(self):
        S = 'ACGCGGATCAAATTT'
        return S
    
    def find_closest_fraction(self,  bp_unwrap):
        closest_bp = None
        closest_distance = float('inf')


        for _, row in self.SAXS_df.iterrows():
            bp = row['Basepair']
            distance = abs(bp - bp_unwrap)

            if distance < closest_distance:
                closest_distance = distance
                closest_bp = bp

                fraction = row['Fraction']
                if closest_distance == 0:
                    return fraction

        if closest_bp is not None:
            return fraction

        return None
    




    def Free_energy_wrapped_DNA(self, seq, start, end, di_prob_dict, mono_prob_dict):

        KbT = 1
        prob = mono_prob_dict[seq[start]][start]

        if prob == 0:
            prob = 0.25
        # count = 0
        # print(seq[start:end])

        for j in range(start + 1, end):
            # print(seq[j - 1:j + 1], seq[j - 1])
            pfac = di_prob_dict[seq[j - 1:j + 1]][j - 1] / mono_prob_dict[seq[j - 1]][j - 1]
            prob *= pfac
            # count = count +1
        # print(count)
        return -KbT * math.log(prob)



    def generate_matrix(self, di_bp):

        labels = ['shift', 'slide', 'rise', 'roll', 'tilt', 'twist']
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

        # Fill the matrix from the dataframe
        # print(MD_param_df[['steps', di_bp]])
        for _, row in self.MD_param_df[['steps', di_bp]].iterrows():
            step, value = row['steps'], row[di_bp]
            i, j = step.split('-')
            matrix.at[i, j] = value * 1.69  ## multiplying with 1.69 makes it KT units from the kcal/mole
            matrix.at[j, i] = value * 1.69  # since it's symmetri


        matrix_np = matrix.to_numpy()
        det_value = np.linalg.det(matrix_np)
        log_value = np.log(det_value)

        return log_value

    def Free_DNA(self, seq, start, end):
        kBT = 1
        sum_log_det_l = 0
        seq_ = seq[start:end]
        # print(seq_)
        pairs = [seq_[i:i + 2] for i in range(len(seq_)-1)]
        # print(pairs)
        for p in pairs:

            sum_log_det_l = sum_log_det_l + self.generate_matrix(p)

        F_free_DNA_l = kBT * sum_log_det_l / 2

        return F_free_DNA_l




    def site_opening(self, close_site_ind):
        cl_sites_can_open = dict()
        sum_rate = 0

        if len(close_site_ind) == 0:
            cl_sites_can_open = dict()

        elif close_site_ind[0] != close_site_ind[-1]:
            cl_sites_can_open[close_site_ind[0]] = self.k_unwrap
            cl_sites_can_open[close_site_ind[-1]] = self.k_unwrap

        else:
            cl_sites_can_open[close_site_ind[0]] = 0.001

        if len(cl_sites_can_open) > 0:
            sum_rate = sum(cl_sites_can_open.values())

        return cl_sites_can_open, sum_rate

    # def site_closing(self, nuce, site_indexes):
    #     unb_sites_can_close = dict()
    #     sum_rate = 0
    #     # print('Nucleosome', nuce)
    #     # print('Closed Sites', site_indexes)
    #     if len(site_indexes[0]) > 0:
    #         try:
    #             assert site_indexes[0][0] - 1 >= 0
    #             if nuce[site_indexes[0][0] - 1] == 1:
    #                 unb_sites_can_close[site_indexes[0][0] - 1] = self.k_wrap
    #
    #         except (AssertionError, IndexError) as msg:
    #             pass
    #
    #         try:
    #             if nuce[site_indexes[0][-1] + 1] == 1:
    #                 unb_sites_can_close[site_indexes[0][-1] + 1] = self.k_wrap
    #         except IndexError:
    #             pass
    #
    #     if len(unb_sites_can_close) > 0:
    #         sum_rate = sum(unb_sites_can_close.values())
    #
    #     return unb_sites_can_close, sum_rate

    
    def calculate_opening_bp(self, site_ind):
        left_bp = 0
        right_bp = 0
        left_bp_prev = 0
        right_bp_prev = 0
        
        if len(site_ind) > 2:
            site_ind = site_ind[1:-1]
        # print(site_ind)
        if len(site_ind) > 0:
            if site_ind[0] - 1 >= 0:
                left_bp = self.binding_bp_left[site_ind[0]]
                left_bp_prev = self.binding_bp_left[site_ind[0] - 1]
            if site_ind[-1] + 1 < len(self.binding_bp_left):
                right_bp = 146 - self.binding_bp_right[site_ind[-1]]
                right_bp_prev = 146 - self.binding_bp_right[site_ind[-1]+1]
        # print(left_bp, right_bp)
        # print(left_bp_prev, right_bp_prev)

        return left_bp_prev + right_bp_prev, left_bp + right_bp
    
    
    
    
    def site_closing(self, seq, nuce, close_site_ind, alpha):
        K_wrapped_rates = dict()

        sum_rate = 0
        beta = 1


        seq_601 = 'CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT'
        # K_eq_601 = {'0': 1e-4, '1': 1e-5, '2': 4e-6, '3': 3e-6, '4': 2e-6, '5': 2e-6, '6': 1e-6, '7': 1e-6, '8': 3e-6, '9': 6e-6, '10': 3e-5, '11': 5e-5, '12': 3e-4, '13': 1e-5}
        # K_eq_601 = {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 1, '13': 1}
        # print(close_site_ind)
        # close_site_ind = [6, 12]

        # nuce = [1,1,1,1,1,1,0,1,1,1,1,1,0,1]
        if len(close_site_ind) > 0:
            # close_site_ind =  [6]
            ##Find the nucleosome state
            # print('L, R ', close_site_ind[0], (13-close_site_ind[-1]))
            L = close_site_ind[0]
            R = 13-close_site_ind[-1]
            # print(self.stationary_array[L][R])

            if L-1>=0 and nuce[L-1]==1:
                # print('opening happened from left side', L-1, R)
                
                current_state = max(self.stationary_array[L][R], math.pow(10, -5))
                prev_state = max(self.stationary_array[L-1][R], math.pow(10, -5))
                # print('K_eq_unwrapping ', current_state/prev_state)
                K_eq_601 = current_state/prev_state

                left_start = self.binding_bp_left[close_site_ind[0] - 1]
                left_end = self.binding_bp_left[close_site_ind[0]]

                # print(left_start, left_end)
                # print(left_end-left_start)
                # print(len(seq_601[left_start:left_end]))

                E_elastic_left = self.Free_energy_wrapped_DNA(seq, left_start, left_end, self.DI_PROB_dict, self.MONO_PROB_dict)
                E_elastic_left_601 = self.Free_energy_wrapped_DNA(seq_601, left_start, left_end, self.DI_PROB_dict, self.MONO_PROB_dict)
                # print('Left Elastic energry for test DNA and 601 DNA ', E_elastic_left, E_elastic_left_601)

                E_free_DNA_left = self.Free_DNA(seq, left_start, left_end)
                E_free_DNA_left_601 = self.Free_DNA(seq_601, left_start, left_end)
                # print('Left Free DNA energy for test DNA and 601 DNA', E_free_DNA_left, E_free_DNA_left_601	)
                
                
                K_wrapped_rates[close_site_ind[0]- 1] = self.k_unwrap /K_eq_601*(math.exp(
                    -beta * ((E_free_DNA_left_601 - E_elastic_left_601)  - (E_free_DNA_left-E_elastic_left)+ alpha))) 
                




            if R-1 >=0 and nuce[min(close_site_ind[-1]+1, 13)] == 1:
                # print('opening happened from Right side', L, R-1)
                current_state = max(self.stationary_array[L][R], math.pow(10, -5))
                prev_state = max(self.stationary_array[L][R-1], math.pow(10, -5))
                # print('K_eq_unwrapping ', current_state/prev_state)
                K_eq_601 = current_state/prev_state
                
                right_start = self.binding_bp_right[close_site_ind[-1]]
                right_end = self.binding_bp_right[close_site_ind[-1] + 1]
                # print(right_start, right_end)
                # print(right_end-right_start)
                # print(seq[right_start:right_end])

                E_elastic_right = self.Free_energy_wrapped_DNA(seq, right_start, right_end, self.DI_PROB_dict, self.MONO_PROB_dict)
                E_elastic_right_601 = self.Free_energy_wrapped_DNA(seq_601, right_start, right_end, self.DI_PROB_dict, self.MONO_PROB_dict)

                # print('Right Elastic energry for test DNA and 601 DNA ', E_elastic_right, E_elastic_right_601)

                E_free_DNA_right = self.Free_DNA(seq, right_start, right_end)
                E_free_DNA_right_601 = self.Free_DNA(seq_601, right_start, right_end)

                # print('Right Free_DNA_energy for 601 DNA', E_free_DNA_right, E_free_DNA_right_601)

                K_wrapped_rates[close_site_ind[-1] + 1] = self.k_unwrap /K_eq_601*(math.exp( -beta * ((E_free_DNA_right_601 - E_elastic_right_601) - (E_free_DNA_right - E_elastic_right)+ alpha)))



        if len(K_wrapped_rates) > 0:  
            sum_rate = sum(K_wrapped_rates.values())
            # print('Wrapping rates ', K_wrapped_rates)
            return K_wrapped_rates, sum_rate
        
        else:
            return K_wrapped_rates, sum_rate 



        # import sys
        # sys.exit()

        # # print(self.calculate_opening_bp(close_site_ind[0]))
        # bp_unwrap_prev, bp_unwrap = self.calculate_opening_bp(close_site_ind[0])

        # # print(self.find_closest_fraction(bp_unwrap_prev))
        # # print(self.find_closest_fraction(bp_unwrap))
        # K_eq_601 = self.find_closest_fraction(bp_unwrap)/self.find_closest_fraction(bp_unwrap_prev)

        # # print(self.find_closest_fraction(self.calculate_opening_bp(close_site_ind[0])))
        # if len(close_site_ind[0]) > 0:

        #     if (nuce[close_site_ind[0][0] - 1] == 1) and (close_site_ind[0][0] - 1) >= 0:  ## checking if the nucleosome site left to the closed is open and not bound by protamines
        #         left_start = self.binding_bp_left[close_site_ind[0][0] - 1]
        #         # left_end = self.binding_bp_right[close_site_ind[0][0]]
        #         left_end = self.binding_bp_left[close_site_ind[0][0]]
        #         # print(left_start, left_end)
        #         # print((seq[left_start:left_end + 1]))
        #         # print('Left side length, ', len(seq[left_start:left_end]))


        #         # left_closest_fraction = self.find_closest_fraction(left_end)
        #         # last_left_closest_fraction = self.find_closest_fraction(left_start)
                
        #         # K_eq_unwrap = 


        #         E_elastic_left = self.Free_energy_wrapped_DNA(seq, left_start, left_end, self.DI_PROB_dict, self.MONO_PROB_dict)

        #         E_elastic_left_601 = self.Free_energy_wrapped_DNA(seq_601, left_start, left_end, self.DI_PROB_dict, self.MONO_PROB_dict)
        #         # print('Left Elastic energry for test DNA and 601 DNA ', E_elastic_left, E_elastic_left_601)

        #         E_free_DNA_left = self.Free_DNA(seq, left_start, left_end)
        #         E_free_DNA_left_601 = self.Free_DNA(seq_601, left_start, left_end)

        #         # print('Left Free DNA energy for test DNA and 601 DNA', E_free_DNA_left, E_free_DNA_left_601	)

        #         # K_wrapped_rates[close_site_ind[0][0] - 1] = self.k_unwrap * math.exp(
        #         #     -beta * (E_elastic_left - E_free_DNA_left + alpha*len(seq[left_start:left_end + 1])))
                

        #         # prefactor = K_eq_601[str(int(close_site_ind[0][0]-1))]

        #         K_wrapped_rates[close_site_ind[0][0] - 1] = self.k_unwrap /K_eq_601*(math.exp(
        #             -beta * ((E_free_DNA_left_601 - E_elastic_left_601)  - (E_free_DNA_left-E_elastic_left)+ alpha))) 
                
        #         # K_wrapped_rates[close_site_ind[0][0] - 1] = self.k_unwrap /prefactor*(math.exp(
        #         #     -beta * (-np.log(prefactor) - (E_elastic_left - E_free_DNA_left)+ alpha))) 

        #     # print('Right-------------------------------')
        #     if close_site_ind[0][-1] != 13:
        #         if nuce[close_site_ind[0][-1]+ 1] == 1:
        #             right_start = self.binding_bp_right[close_site_ind[0][-1]]
        #             # right_start = self.binding_bp_left[close_site_ind[0][-1]]
        #             right_end = self.binding_bp_right[close_site_ind[0][-1] + 1]
        #             #
        #             # print(right_start, right_end)
        #             # print(seq[right_start:right_end + 1])
        #             # print('Rigth side length ', len(seq[right_start:right_end + 1]))

        #             E_elastic_right = self.Free_energy_wrapped_DNA(seq, right_start, right_end, self.DI_PROB_dict, self.MONO_PROB_dict)
        #             E_elastic_right_601 = self.Free_energy_wrapped_DNA(seq_601, right_start, right_end, self.DI_PROB_dict, self.MONO_PROB_dict)

        #             # print('Right Elastic energry for test DNA and 601 DNA ', E_elastic_right, E_elastic_right_601)

        #             E_free_DNA_right = self.Free_DNA(seq, right_start, right_end)
        #             E_free_DNA_right_601 = self.Free_DNA(seq_601, right_start, right_end)

        #             # print('Right Free_DNA_energy for 601 DNA', E_free_DNA_right, E_free_DNA_right_601)

        #             # K_wrapped_rates[close_site_ind[0][-1] + 1] = self.k_unwrap * math.exp(
        #             #     -beta * (E_elastic_right - E_free_DNA_right + alpha*len(seq[right_start:right_end + 1])))

        #             # prefactor = K_eq_601[str(int(close_site_ind[0][-1]+1))]


        #             K_wrapped_rates[close_site_ind[0][-1] + 1] = self.k_unwrap /K_eq_601*(math.exp( -beta * ((E_free_DNA_right_601 - E_elastic_right_601) - (E_free_DNA_right - E_elastic_right)+ alpha)))


        # if len(K_wrapped_rates) > 0:
        #     sum_rate = sum(K_wrapped_rates.values())
        # # print('Wrapping rates ', K_wrapped_rates)
        # return K_wrapped_rates, sum_rate


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
        self.p_conc = p_conc * math.pow(10, -6)  # in moles/liters
        self.P_free = self.p_conc * 6 * math.pow(10, 23) * math.pow(10, -15)  # initial concentration of protamines molecules/cubic_micrometer
        self.N_bound = 0


    def prot_rates(self):
        print('lrATEF')
        pass

    # def unbound_site(self, nuce, op_indexes, pt_indexes):
    #     """defines the binding rate of protamine for each unbound site
    #     :param op_indexes: contains the indexes for the open unbound sites
    #     :param pt_indexes: contains the indexes for the protamine bound site
    #      :return: dictionary with key as the index of the unbound sites and value as its binding rate.
    #                 Also returns the sum of all the binding rates"""
    #
    #     unb_sites = dict()
    #     # neigh_dict=dict()
    #     sum_rate = 0
    #     if len(op_indexes[0]) != 0:
    #
    #         if len(pt_indexes[0]) != 0:
    #
    #             for i in op_indexes[0]:
    #                 neig_count = 0
    #                 ###count for the neighbors
    #                 if max(i - 1, 0) in pt_indexes[0]:
    #                     k = i - 1
    #                     while nuce[max(k, 0)] == 2 and k >= 0:
    #                         neig_count += 1
    #                         k = k - 1
    #
    #                 if i + 1 in pt_indexes[0]:
    #                     k = i + 1
    #                     while nuce[min(k, 13)] == 2 and k <= 13:
    #                         neig_count += 1
    #                         k = k + 1
    #
    #                 unb_sites[i] = self.k_bind  * (1 + self.cooperativity * neig_count) * self.P_free
    #         #             neigh_dict[i] = neig_count
    #         else:
    #             for i in op_indexes[0]:
    #                 unb_sites[i] = self.k_bind* self.P_free
    #
    #     if len(unb_sites) > 0:
    #         sum_rate = sum(unb_sites.values())
    #
    #     return unb_sites, sum_rate


    def unbound_site(self, op_indexes):
        unb_sites = dict()
        # neigh_dict=dict()
        sum_rate = 0
        if len(op_indexes) != 0:
            for i in op_indexes:
                unb_sites[i] = self.k_bind * self.P_free
                # print(self.P_free)
        if len(unb_sites) > 0:
            sum_rate = sum(unb_sites.values())

        return unb_sites, sum_rate


    # def bound_site(self, pt_indexes):
    #     """defines the unbinding rate of protamine for each bound site
    #     :param pt_indexes: contains the indexes for the protamine bound site
    #      :return: dictionary with key as the index of the bound sites and value as its unbinding rate.
    #                 Also returns the sum of all the unbinding rates"""
    #
    #     bd_sites = dict()
    #     sum_rate = 0
    #
    #     if len(pt_indexes[0]) != 0:
    #         for i in pt_indexes[0]:
    #             bd_sites[i] = self.k_unbind
    #
    #     if len(bd_sites) > 0:
    #         sum_rate = sum(bd_sites.values())
    #
    #     return bd_sites, sum_rate

    def delta_E(self, p, J, s_l=0, s_r=0):
        ### get the energy required to move from a bound state to unbound state
        ## s_l and s_r are the left and right neighbours to the bound site.
        ## refer to the task documentation for the derivation of the formulae. The energy here is in terms of Kb*T.

        U = np.log(0.00244 * p)  ## in terms of KT

        #     J=12.6/(1-s_l-s_r)

        #     return (U + J)-J*0.5*(s_l+s_r)
        return J * (s_l + s_r) + U

    def get_spin_value(self, v):
        if v == 2:
            return 1
        else:
            return 0

    ##### With Coooperativity, here the cooperativity will make the protamine unbinding difficult.
    def bound_site(self, nuce, pt_indexes, B=1):
        bd_sites = dict()
        sum_rate = 0
        beta = B
        J = self.cooperativity

        if len(pt_indexes) != 0:
            for i in pt_indexes:

                if i == 0:
                    s_right = self.get_spin_value(nuce[i + 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = self.k_bind * self.P_free / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_r=s_right))

                elif i == 13:
                    s_left = self.get_spin_value(nuce[i - 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = self.k_bind * self.P_free / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_l=s_left))

                else:
                    s_left = self.get_spin_value(nuce[i - 1])
                    s_right = self.get_spin_value(nuce[i + 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = self.k_bind * self.P_free / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_l=s_left, s_r=s_right))
            # print('Function Bpund Site Pfree', self.P_free)
            # print('Function Bpund Site Kbind' ,self.k_bind)
            # print('Function Bpund Site J', J)
        if len(bd_sites) > 0:
            sum_rate = sum(bd_sites.values())

        return bd_sites, sum_rate



class Simulation(nucleosme, protamines):
    def __init__(self, k_unwrap, k_wrap, k_unbind, k_bind, p_conc, num_nucleosomes, sequence, cooperativity=1, binding_sites=14, N=1000):

        nucleosme.__init__(self, k_unwrap, k_wrap, num_nucleosomes, binding_sites)
        protamines.__init__(self, k_unbind, k_bind, p_conc, cooperativity)

        self.N = N
        # self.uniform_pos_arg_njit = nb.njit(self.uniform_pos_arg)
        self.t = 0
        self.num_nucleosomes =  num_nucleosomes
        self.Seq = sequence
        self.add_param = 0 ### KT this is per unit length
        self.nuc_fall_flag = False

        # self.p_conc = p_conc
        # self.k_unwrap = k_unwrap
        # self.k_wrap = k_wrap
        # self.binding_sites = binding_sites
        # self.k_unbind = k_unbind
        # self.k_bind = k_bind
        # self.cooperativity = cooperativity
        # print(self.nucleosomes)
        # import sys
        # sys.exit()
    def simulate_main(self):
        times = []
        N_closed_array = []
        N_open_array = []
        P_free_array = []
        N_bound_array = []
        ft_nucleosome_fell = 0
        for n in range(1, self.N):
            # Calculate total number of unbound sites in all nucleosomes
            # total_unbound_sites = func_cnt(system_vars.nucleosomes, ele=1)
            # print(n)
            # print(self.p_conc)
            # print(self.P_free)

            all_rates = [self.calculate_rates(nucleo=nucleosome) for nucleosome in self.nucleosomes]
            # print(self.nucleosomes)
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
            # print('State of the simulation step ', self.P_free, self.N_bound, self.N_open)
            # print('Nucleosome:', self.nucleosomes[nucleosome_idx])

            if len(np.where(self.nucleosomes[nucleosome_idx] == 0)[0])==0 and (self.nuc_fall_flag ==False):
                ft_nucleosome_fell = self.t
                self.nuc_fall_flag =True
                print('Step and Time at which nucleosome fell ', n, self.t)
            # Record state if necessary
            times.append(self.t)
            N_closed_array.append(self.N_closed)
            N_open_array.append(self.N_open)
            P_free_array.append(self.P_free)
            N_bound_array.append(self.N_bound)

            if self.nuc_fall_flag:
                return ft_nucleosome_fell, times, N_closed_array, N_open_array, P_free_array, N_bound_array,self.nucleosomes

        return ft_nucleosome_fell, times, N_closed_array, N_open_array, P_free_array, N_bound_array, self.nucleosomes





    def calculate_rates(self, nucleo):
        cl_site_indexes = np.ravel(np.where(nucleo == 0))
        pt_site_indexes = np.ravel(np.where(nucleo == 2))
        op_sites_indexes = np.ravel(np.where(nucleo == 1))


        closed_sites_can_open_rate, rate_open = self.site_opening(cl_site_indexes)

        unbound_sites_can_close_rate, rate_close = self.site_closing(self.Seq, nucleo, cl_site_indexes, alpha=self.add_param)

        unbound_sites_rate, rate_bind = self.unbound_site(op_sites_indexes)

        bound_sites_rate, rate_unbind = self.bound_site(nucleo, pt_site_indexes)


        print(nucleo)
        print('Unwrapping rate', closed_sites_can_open_rate)
        print('Rewrapping rate', unbound_sites_can_close_rate)


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

    start = time.perf_counter()

    times_ALL = []
    N_closes_array_ALL = []
    N_open_array_ALL = []
    P_free_array_ALL = []
    N_bound_array_ALL = []

    Sim_data = pd.DataFrame(columns=['C', 'bound_prot', 'nuc_open', 'nuc_closed'])
    shape_list = dict()
    temp_bound = []
    temp_open = []
    temp_closed = []
    end_time = []
    con = []
    temp_nuc_lifetime = []

    count_open_site= [0]*15

    #
    sum_nuc_state = [0] * 14  # Assuming the length of your arrays is 14

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        pool = []
        for i in range(N_Independent_nucleosomes):
            print(i)
            pool.append(executor.submit(Simulation(k_unwrap=K_UNWRAP, k_wrap=K_WRAP, k_unbind=K_DES,  k_bind=K_ADS,
                                                   p_conc=P_CONC,  cooperativity=COOPERATIVITY,
                                                   num_nucleosomes=System_Nucleosomes,
                                                   N=Simulation_steps, sequence=SEQUENCE).simulate_main))

        plot_cnt=0
        for j in concurrent.futures.as_completed(pool):
            nuc_lifetime, times, N_closed_array, N_open_array, P_free_array, N_bound_array, nuc_state = j.result()



            # Plotting the figure for evolution of number of open sites and the bound protamine for each of the independent nucleosome.

            times_sampled = times[::10]
            N_closed_sampled = N_closed_array[::10]
            N_open_sampled = N_open_array[::10]
            P_free_sampled = P_free_array[::10]
            N_bound_sampled = N_bound_array[::10]

            N_open_sample_total = list(map(add, N_open_sampled, N_bound_sampled))
            N_open_total = list(map(add, N_open_array, N_bound_array))

            # print(nuc_state)


            sum_nuc_state = [sum(x) for x in zip(sum_nuc_state, nuc_state[0])]

            index = np.count_nonzero(nuc_state[0] == 1)

            count_open_site[index] = count_open_site[index] + 1

            
            # plt.figure(figsize=(6, 6))
            
            # # plt.plot(times_sampled, N_closed_sampled, label='N Closed', color='blue')
            # plt.plot(times_sampled, N_open_sample_total, label='N Open', color='green')
            # # plt.plot(times_sampled, P_free_sampled, label='P Free', color='red')
            # plt.plot(times_sampled, N_bound_sampled, label='N Bound', color='purple')
            # plt.axvline(x=nuc_lifetime, color='black')
            
            # plt.xlabel('Time')
            # plt.ylabel('Values')
            # plt.title('Line Charts for Nucleosome States')
            # plt.legend()
            
            # plt.savefig(str(plot_cnt)+'.png')
            # plot_cnt = plot_cnt+1


            ##Keeping only the last entry of the simulation
            end_time.append(times[-1])
            temp_open.append(N_open_array[-1])
            temp_bound.append(N_bound_array[-1])
            temp_closed.append(N_closed_array[-1])
            temp_nuc_lifetime.append(nuc_lifetime) ### Store the time when nucleosome fell.

        


        acces_Pk = np.divide(sum_nuc_state, N_Independent_nucleosomes)
        print(sum_nuc_state)
        print(acces_Pk)

        plt.figure(figsize=(6, 6))
        plt.plot(acces_Pk)
        plt.grid()
        plt.show()

        print(count_open_site)
        plt.figure(figsize=(6, 6))
        plt.plot(count_open_site)
        plt.grid()
        plt.show()
        # plt.savefig(RESULT_DIR+'Figures/Bind_site_access_Prob.png')


        bound_mean = mean(temp_bound)
        open_mean = mean(temp_open)
        closed_mean = mean(temp_closed)
        time_mean = mean(end_time)
        nuc_lifetime_mean = mean(temp_nuc_lifetime)


        # with open(RESULT_DIR+'temp/'+str(Param_ind)+'_'+str(SEQ_id)+'.txt', 'w') as out_file:
        #     ID = ['Paramid:'+str(Param_ind),
        #           'Seqid:'+str(SEQ_id),
        #           'P_conc:'+str(P_CONC),
        #           'Cooperativity:'+str(COOPERATIVITY),
        #           'Open:'+str(open_mean),
        #           'Closed:' + str(closed_mean),
        #           'Bound:' + str(bound_mean),
        #           'Time:' + str(time_mean),
        #           'Nuc_LT:' + str(nuc_lifetime_mean)
        #           ]

        #     out_file.write('>'+ '|'.join(ID)+'\n'+SEQUENCE+'\n')

            # temp_bound.append(bound_value)
            #
            # temp_open.append(open_value)
            # temp_closed.append(closed_value)
            # end_time.append(time_value)
            # con.append(con_value)


    end = time.perf_counter()
    print(f'Finished in {round(end - start, 2)} second(s)')
    print(f'Parameter Protamine:{P_CONC} and Cooperativity:{COOPERATIVITY} with Sequence_id:{SEQ_id} is done>>>>>>>>>>>>>>>>>>>>')
    # for cycles in range(N_nucleosomes):
    #     print(cycles)
    #     s1 = Simulation(k_unwrap=4, k_wrap=21, k_unbind=23,  k_bind=2, p_conc=0.0,  cooperativity=1, num_nucleosomes=1, N=10000, sequence=SEQUENCE)
    #     times, N_closed_array, N_open_array, P_free_array, N_bound_array = s1.simulate_main()
    #     # times_ALL.append(times)
    #     # N_closes_array_ALL.append(N_closed_array)
    #     # N_open_array_ALL.append(N_open_array)
    #     # P_free_array_ALL.append(P_free_array)
    #     # N_bound_array_ALL.append(N_bound_array)
    #     #
    #     # times_ALL = np.array(times_ALL)
    #     # N_closes_array_ALL = np.array(N_closes_array_ALL)
    #     # N_bound_array_ALL = np.array(N_bound_array_ALL)
    #     # N_open_array_ALL = np.array(N_open_array_ALL)
    #
    #     ##captures the last event of the nucleosome
    #     # temp_bound.append(N_bound_array[-1])
    #     # temp_open.append(N_open_array[-1])
    #     # temp_closed.append(N_closed_array[-1])
    #     # end_time.append(times[-1])
    #
    #     ##capture full journey of each nucleosome
    #     temp_bound.append(N_bound_array)
    #     temp_open.append(N_open_array)
    #     temp_closed.append(N_closed_array)
    #     end_time.append(times)

    # print(temp_open)
    # print(temp_bound)

#     for i in range(10):
#         plt.figure(figsize=(8, 6))
#         plt.scatter(end_time[i], temp_open[i], label=f"Nucleosome {i + 1}")
#         plt.title(f"Nucleosome {i + 1}")
#         plt.xlabel('Time Step')
#         plt.ylabel('Open Sites')
#         plt.legend()
#         plt.show()
#
#
#
#
#     # Plot the mean across all nucleosomes
#     mean_open_sites = np.mean(temp_open, axis=0)
#     mean_time_steps = np.mean(end_time, axis=0)
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(mean_time_steps, mean_open_sites, label="Mean Across Nucleosomes", color="red")
#     plt.title("Mean Across Nucleosomes")
#     plt.xlabel('Time Step')
#     plt.ylabel('Open Sites')
#     plt.legend()
#     plt.show()
#
#     # bound_mean = mean(N_bound_array)
#     # open_mean = mean(N_open_array)
#     # closed_mean = mean(N_closed_array)
#     # time_mean = mean(end_time)
#     # print(N_bound_array_ALL)
#
#
# # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# # print(tf.config.list_physical_devices('GPU'))
#
#
# import time_series_generator