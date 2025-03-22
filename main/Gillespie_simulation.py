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
import datetime
import csv
from nucleosome_breath import NucleosomeBreath


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

    def __init__(self, k_unwrap, k_wrap, num_nucleosomes, Nucleosome_breath_instance, binding_sites=14):
        self.k_unwrap = k_unwrap
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        self.nucleosomes = [np.zeros(self.binding_sites, dtype=int) for _ in range(num_nucleosomes)]
        self.N_closed = num_nucleosomes * binding_sites
        self.N_open = 0
        self.binding_bp_right = [7, 18, 30, 39, 50, 60, 70, 81, 91, 101, 112, 122, 132, 146]
        self.binding_bp_left = [0, 14, 24, 34, 45, 55, 65, 76, 86, 96, 107, 116, 128, 139]
        self.nucleosome_breath = Nucleosome_breath_instance   

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

    
           # Mask to select only the left side of the diagonal (including diagonal)
            mask = np.zeros(self.stationary_array.shape, dtype=bool)
            for i in range(self.stationary_array.shape[0]):
                for j in range(self.stationary_array.shape[0]-i):
                    mask[i, j] = 1

            # Extracting the valid values
            valid_values = self.stationary_array[mask]

            # Calculate the standard deviation of these values
            std_dev = np.std(valid_values)

            noise_level = STD_DEV_NOISE * std_dev  # Define noise level as 10% of the standard deviation
            
            # np.random.seed(0)

            # Generate Gaussian noise
            noise = np.random.normal(0, noise_level, self.stationary_array.shape)
            noise[~mask] = 0
            noise_masked = np.clip(noise, 0, None)  # Example: clipping to ensure non-negative values


            self.stationary_array = self.stationary_array + noise_masked
            # Normalize the stationary_array
            self.stationary_array/= np.sum(self.stationary_array)

            # l = random.randint(0, 14)
            # r = random.randint(0, 14 - l)  # Select r such that r <= 14 - l

            # self.stationary_array[l][r] = self.stationary_array[l][r] + NOISE

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
            # matrix.at[i, j] = value * 1.69  ## multiplying with 1.69 makes it KT units from the kcal/mole
            # matrix.at[j, i] = value * 1.69  # since it's symmetri
            matrix.at[i, j] = value  
            matrix.at[j, i] = value   


        matrix_np = matrix.to_numpy()
        det_value = np.linalg.det(matrix_np)
        log_value = np.log(det_value)
        # log_value = np.log((det_value*math.pow(180, 6))/(math.pow(10, 6)*math.pow(np.pi, 6))) ### this will convert it to nm and radians units

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
            cl_sites_can_open[close_site_ind[0]] = LAST_SITE_OPENING_RATE

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


    def nucleosome_rewrapping(self, Seq, nuce, close_site_ind):
        beta = 1
        K_wrapped_rates = dict()
        sum_rate = 0
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        if len(close_site_ind) > 0:

            L = close_site_ind[0]
            R = 13-close_site_ind[-1]

            if L-1>=0 and nuce[L-1]==1:

                unwrapped_cur_state = self.nucleosome_breath.select_phosphate_bind_sites(left=close_site_ind[0])
                rewrap_state = self.nucleosome_breath.select_phosphate_bind_sites(left=close_site_ind[0]-1) 
                F_rewrap, S_rewrap, E_rewrap, F_Free_rewrap = self.nucleosome_breath.calculate_free_energy(Seq, site_loc=rewrap_state) 
                F_unwrapped, S_unwrapped, E_wrapped, F_free_unwrap = self.nucleosome_breath.calculate_free_energy(Seq, site_loc=unwrapped_cur_state)
                print('Rewrap_state ', rewrap_state)
                print('Unwrapped_state ', unwrapped_cur_state)

                delta_F_left = F_rewrap - F_unwrapped 
                print('Left delta Engergy ', delta_F_left, F_rewrap, F_unwrapped)
                K_wrapped_rates[close_site_ind[0]- 1] = self.k_unwrap * math.exp(-beta*delta_F_left)

            if R-1 >=0 and nuce[min(close_site_ind[-1]+1, 13)] == 1:
                
                R_unwrapped_cur_state = self.nucleosome_breath.select_phosphate_bind_sites(right=close_site_ind[-1])
                R_rewrap_state = self.nucleosome_breath.select_phosphate_bind_sites(right=close_site_ind[-1]+1) 

                R_F_rewrap, R_S_rewrap, R_E_rewrap, R_F_rewrapn = self.nucleosome_breath.calculate_free_energy(Seq, site_loc=R_rewrap_state)
                R_F_unwrapped, R_S_unwrapped, R_E_wrapped, R_F_unwrappedn = self.nucleosome_breath.calculate_free_energy(Seq, site_loc=R_unwrapped_cur_state)


                print('Right_Rewrap_state ', R_rewrap_state)
                print('Right_Unwrapped_state ', R_unwrapped_cur_state)
                delta_F_right =  R_F_rewrap - R_F_unwrapped 
                print('Right delta Engergy ', delta_F_right, R_F_rewrap, R_F_unwrapped)
                K_wrapped_rates[close_site_ind[-1] + 1] = self.k_unwrap * math.exp(-beta*delta_F_right)

        if len(K_wrapped_rates) > 0:  
            sum_rate = sum(K_wrapped_rates.values())
            # print('Wrapping rates ', K_wrapped_rates)
            return K_wrapped_rates, sum_rate
        
        else:
            return K_wrapped_rates, sum_rate   


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
                    bd_sites[i] = (self.k_bind * self.P_free )/ math.exp(beta * self.delta_E(p=self.P_free, J=J, s_r=s_right))

                elif i == 13:
                    s_left = self.get_spin_value(nuce[i - 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = (self.k_bind * self.P_free) / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_l=s_left))

                else:
                    s_left = self.get_spin_value(nuce[i - 1])
                    s_right = self.get_spin_value(nuce[i + 1])

                    #                 bd_sites[i] = k_unbind
                    bd_sites[i] = (self.k_bind * self.P_free) / math.exp(beta * self.delta_E(p=self.P_free, J=J, s_l=s_left, s_r=s_right))
            # print('Function Bpund Site Pfree', self.P_free)
            # print('Function Bpund Site Kbind' ,self.k_bind)
            # print('Function Bpund Site J', J)
        if len(bd_sites) > 0:
            sum_rate = sum(bd_sites.values())

        return bd_sites, sum_rate



class Simulation():
    def __init__(self, nucleosme_instance, protamines_instance,  num_nucleosomes, sequence, N=1000):
        self.nucleosme = nucleosme_instance
        self.protamines = protamines_instance
        self.N = N
        self.t = 0
        self.num_nucleosomes = num_nucleosomes
        self.Seq = sequence
        self.add_param = -0.5 
        self.nuc_fall_flag = False
        # self.record_state = record_state
        # self.uniform_pos_arg_njit = nb.njit(self.uniform_pos_arg)

    def simulate_main(self):

       
        times = []
        N_closed_array = []
        N_open_array = []
        P_free_array = []
        N_bound_array = []
        Nucleosome_state = []

        ft_nucleosome_fell = 0

        step_counter_1 = 0
        step_counter_2 = 0

        os.makedirs(NUCLEOSOME_STATE_RECORD_DIR, exist_ok=True)

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
                self.protamines.P_free -= 1
                self.protamines.N_bound += 1
            else:
                # Protamine unbinds
                self.nucleosme.N_open += 1
                self.protamines.P_free += 1
                self.protamines.N_bound -= 1
            # print('State of the simulation step ', self.P_free, self.N_bound, self.N_open)
            # print('Nucleosome:', self.nucleosomes[nucleosome_idx])

            if len(np.where(self.nucleosme.nucleosomes[nucleosome_idx] == 0)[0])==0 and (self.nuc_fall_flag ==False):
                ft_nucleosome_fell = self.t
                self.nuc_fall_flag =True
                print('Step and Time at which nucleosome fell ', n, self.t)
            # Record state if necessary
            # print(self.nucleosme.N_open)    
            # print(self.nucleosme.nucleosomes)    
            times.append(self.t)
            N_closed_array.append(self.nucleosme.N_closed)
            N_open_array.append(self.nucleosme.N_open)
            P_free_array.append(self.protamines.P_free)
            N_bound_array.append(self.protamines.N_bound)
            Nucleosome_state.append(np.concatenate(self.nucleosme.nucleosomes))
            

            if n%FILE_CHUNK_SIZE == 0:

                self.batch_write_derivates(times, N_closed_array, N_open_array, P_free_array, N_bound_array, Nucleosome_state, step_counter_1)
                print(f'{step_counter_1} records written.....')
                step_counter_1 += 1
                # Clear the arrays to free up memory
                Nucleosome_state.clear()
                times.clear()
                N_closed_array.clear()
                N_open_array.clear()
                P_free_array.clear()
                N_bound_array.clear()


            if self.nuc_fall_flag:
                return ft_nucleosome_fell, times, N_closed_array, N_open_array, P_free_array, N_bound_array,self.nucleosme.nucleosomes

        return ft_nucleosome_fell, times, N_closed_array, N_open_array, P_free_array, N_bound_array, self.nucleosme.nucleosomes

    
    

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
        cl_site_indexes = np.ravel(np.where(nucleo == 0))
        pt_site_indexes = np.ravel(np.where(nucleo == 2))
        op_sites_indexes = np.ravel(np.where(nucleo == 1))


        closed_sites_can_open_rate, rate_open = self.nucleosme.site_opening(cl_site_indexes)

        # unbound_sites_can_close_rate, rate_close = self.nucleosme.site_closing(self.Seq, nucleo, cl_site_indexes, alpha=self.add_param)

        unbound_sites_can_close_rate, rate_close = self.nucleosme.nucleosome_rewrapping(self.Seq, nucleo, cl_site_indexes)


        unbound_sites_rate, rate_bind =  self.protamines.unbound_site(op_sites_indexes)

        bound_sites_rate, rate_unbind =  self.protamines.bound_site(nucleo, pt_site_indexes)


        # print(nucleo)
        print('Unwrapping rate', closed_sites_can_open_rate)
        print('Rewrapping rate', unbound_sites_can_close_rate)
        # print('Binding rate', unbound_sites_rate)
        # print('Unbinding rate', bound_sites_rate)

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

def simulation_fn(index):
    Nucleosome_breath_instance = NucleosomeBreath()
    nucleosme_instance = nucleosme(k_unwrap=K_UNWRAP, k_wrap=K_WRAP, num_nucleosomes=System_Nucleosomes, Nucleosome_breath_instance=Nucleosome_breath_instance, binding_sites=NUC_BIND)
    protamines_instance = protamines(k_unbind=K_DES, k_bind=K_ADS, p_conc=P_CONC, cooperativity=COOPERATIVITY)
    
    simulation = Simulation(nucleosme_instance, protamines_instance, System_Nucleosomes, SEQUENCE, N=Simulation_steps)

    print('Simulation instance created for index:', index)
    return simulation 


def parallel_execute_simulation():
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        pool = []
        for i in range(N_Independent_nucleosomes):
            print(i)
            pool.append(executor.submit(simulation_fn(i).simulate_main))

        plot_cnt = 0
        for j in concurrent.futures.as_completed(pool):
            print('Simulation instance completed:', plot_cnt)
            nuc_lifetime, times, N_closed_array, N_open_array, P_free_array, N_bound_array, nuc_state = j.result()
            yield (nuc_lifetime, times, N_closed_array, N_open_array, P_free_array, N_bound_array, nuc_state)
            plot_cnt += 1

import psutil
import threading
def monitor_process_count():
    current_process = psutil.Process()
    while True:
        children = current_process.children()
        print(f"Number of child processes: {len(children)}")
        time.sleep(1)  # Check every second

if __name__== '__main__':

    start = time.perf_counter()

    # Start monitoring in a separate thread
    #monitor_thread = threading.Thread(target=monitor_process_count, daemon=True)
    #monitor_thread.start()



    # Nucleosome_breath_instance = NucleosomeBreath()
    # nucleosme_instance = nucleosme(k_unwrap=K_UNWRAP, k_wrap=K_WRAP, num_nucleosomes=System_Nucleosomes, Nucleosome_breath_instance=Nucleosome_breath_instance, binding_sites=NUC_BIND)

    # print(nucleosme_instance.Free_DNA(SEQUENCE, 0, 139))

    # print(SEQUENCE[0:14])
    # import sys
    # sys.exit()
    Sim_data = pd.DataFrame(columns=['C', 'bound_prot', 'nuc_open', 'nuc_closed'])
    shape_list = dict()
    Last_bound = []
    Last_open = []
    Last_closed = []
    end_time = []
    con = []
    END_nuc_lifetime = []

    


    def plot_nucleosome_evolution(tl, opn, bound, closed, free_prot, ct, nuc_lft=0, period=10):
        N_open_sample_total = list(map(add, opn[::period], bound[::period]))
        # print(N_open_sample_total)
        plt.figure(figsize=(20, 9))
        # print(tl)    
        # plt.plot(times_sampled, N_closed_sampled, label='N Closed', color='blue')
        plt.plot(tl[::period], N_open_sample_total, label='N Open', color='green')
        # plt.plot(times_sampled, P_free_sampled, label='P Free', color='red')
        plt.plot(tl[::period], N_bound_array[::period], label='N Bound', color='purple')
        # plt.axvline(x=nuc_lifetime, color='black')
        
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Line Charts for Nucleosome States')
        plt.legend()
        # Create a directory based on the date of the run
        os.makedirs(SIMULATION_PLOTS_DIR, exist_ok=True)
        
        plt.savefig(SIMULATION_PLOTS_DIR + str(ct) + '.png')
        plt.close()

        # plt.show()


    def acces_prob_plot():
        count_open_site= [0]*15

        sum_nuc_state = [0]*14  # Assuming the length of your arrays is 14

        os.makedirs(BREATHING_DIR, exist_ok=True)

        with open(State_record_file, 'r') as file:
            for line in file:
                nuc_state = np.array([int(x) for x in line.strip().replace('[','').replace(']','').split(',')])
                sum_nuc_state = [sum(x) for x in zip(sum_nuc_state, nuc_state)]
                index = np.count_nonzero(nuc_state == 1)
                count_open_site[index] = count_open_site[index] + 1


        acces_Pk = np.divide(sum_nuc_state, N_Independent_nucleosomes)

        plt.figure(figsize=(6, 6))
        plt.plot(acces_Pk)
        plt.grid()
        # plt.show()
        plt.savefig(BREATHING_DIR +'Bind_site_access_Prob.png')
        plt.close()

        print(count_open_site)
        plt.figure(figsize=(6, 6))
        plt.plot(count_open_site)
        plt.grid()
        # plt.show()
        plt.savefig(BREATHING_DIR+'Open_site_fraction.png')
        plt.close()

    


    def read_csv_files(directory):
        data = pd.DataFrame(columns=['Time', 'N_closed', 'N_open', 'P_free', 'N_bound'])  # Create an empty DataFrame to store the data
        
        # Iterate over each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):  # Check if the file is a .csv file
                filepath = os.path.join(directory, filename)  # Construct the full file path
                
                # Read the .csv file and append the data to the DataFrame
                df = pd.read_csv(filepath)
                data = pd.concat([data, df],  axis=0, ignore_index=True)  # Append the data to the DataFrame
        
        return data


    def get_indices_of_ones(data):
        matrix = np.zeros((15, 15))
        
        for row in data:
            row_indices = np.ravel(np.where(row == 0))
            # print(row)
            if len(row_indices) > 0:
                L = row_indices[0]
                R = 13-row_indices[-1]

                matrix[L, R] += 1
            else:
                print('Fallen Nucleosome')
                ## need to addd the for the other case

        return matrix
    # Define the file path

    os.makedirs(STATE_RECORD_DIR, exist_ok=True)
    State_record_file = STATE_RECORD_DIR + 'State_record_nuc.txt' ###Recording the final state of the nucleosome for each independent run/nucleosome 

    if SINGLE_NUCLEOSOME_EVOLUTION:
        simulation_OBJ = simulation_fn(0)
        nuc_lifetime, times, N_closed_array, N_open_array, P_free_array, N_bound_array, nuc_state = simulation_OBJ.simulate_main()
        
        
                
        files = os.listdir(NUCLEOSOME_STATE_RECORD_DIR)
        # print(NUCLEOSOME_STATE_RECORD_DIR)
        # print(files)
        os.makedirs(BREATHING_DIR, exist_ok=True)
        state_sum = np.zeros((1, 14))
        state_matrix = np.zeros((15, 15))
        # Loop over each file
        for filename in files:
            # Check if the file is a .npy file
            if filename.endswith('.npy'):
                # Construct the full file path
                filepath = os.path.join(NUCLEOSOME_STATE_RECORD_DIR, filename)
                
                # Load the .npy file
                data = np.load(filepath)

                m = get_indices_of_ones(data)
                state_matrix = np.add(state_matrix, m)

                summed_data = np.sum(data, axis=0)
                state_sum = np.add(state_sum, summed_data)




        
        prob_site_access = np.divide(state_sum, Simulation_steps)
        state_matrix_prob = np.divide(state_matrix, Simulation_steps)


        # print(prob_site_access)
        # print(state_matrix_prob)
        for i in range(prob_site_access.shape[0]):      
            plt.figure(figsize=(6, 6))
            plt.plot(prob_site_access[i])
            plt.grid()
            plt.savefig(BREATHING_DIR + f'Bind_site_access_Prob_{i}.png')
            plt.close()
            
        plt.figure(figsize=(10, 8))
        plt.imshow(state_matrix_prob, cmap='magma_r', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Current State')
        plt.ylabel('Next State')
        plt.title('State Transition Probability Matrix')
        plt.savefig(BREATHING_DIR + 'state_matrix_heatmap.png')
        plt.close()


        DF = read_csv_files(directory=NUCLEOSOME_STATE_RECORD_DIR)
        DF['N_open_total'] = DF['N_open'] + DF['N_bound']
        # print(DF)


        plt.figure(figsize=(10, 6))
        plt.plot(DF['Time'], DF['N_open_total'])
        plt.xlabel('Time')
        plt.ylabel('N_open')
        plt.title('Trajectory of N_open with Time')
        plt.grid(True)
        plt.savefig(BREATHING_DIR + 'N_open_trajectory.png')
        plt.close()

        # Plot the histogram of N_open
        plt.figure(figsize=(10, 6))
        plt.hist(DF['N_open_total'], bins=20, edgecolor='black')
        plt.xlabel('N_open_total')
        plt.ylabel('Frequency')
        plt.title('Histogram of N_open Distribution')
        plt.grid(True)
        plt.savefig(BREATHING_DIR + 'N_open_histogram.png')
        plt.close()
        
        
        # plot_nucleosome_evolution(tl=times,
        #                             opn=N_open_array,
        #                             bound=N_bound_array, 
        #                             closed=N_closed_array, 
        #                             free_prot=P_free_array, 
        #                             ct = 0,
        #                             nuc_lft=0, 
        #                             period=10)


    else:

        if os.path.exists(State_record_file):
            os.remove(State_record_file)

        with open(State_record_file, 'w') as file:
            counter = 0

            for result in parallel_execute_simulation():
                nuc_lifetime, times, N_closed_array, N_open_array, P_free_array, N_bound_array, nuc_state = result
                # print(nuc_state)
                # file.write(nuc_state[0] + '\n')
                # file.write(str(nuc_state[0])+ '\n')
                file.write(np.array2string(nuc_state[0], separator=',') + '\n')
                counter += 1
                if counter % 100 == 0:
                    # Flush the file buffer to ensure data is written immediately
                    file.flush()


                ##Keeping only the last entry of the simulation
                ### I can also get this from the State_record_file because it also contains the last state of the nucleosome
                end_time.append(times[-1])
                Last_open.append(N_open_array[-1])
                Last_bound.append(N_bound_array[-1])
                Last_closed.append(N_closed_array[-1])
                END_nuc_lifetime.append(nuc_lifetime) ### Store the time when nucleosome fell.

                if PLOT_NUC_STATE:
                    plot_nucleosome_evolution(tl=times,
                                            opn=N_open_array,
                                            bound=N_bound_array, 
                                            closed=N_closed_array, 
                                            free_prot=P_free_array, 
                                            ct = counter,
                                            nuc_lft=0, 
                                            period=10)

        # Close the file
        file.close()


        if BREATHING_ONLY:
            acces_prob_plot()
    
       


    # bound_mean = mean(temp_bound)
    # open_mean = mean(temp_open)
    # closed_mean = mean(temp_closed)
    # time_mean = mean(end_time)
    # nuc_lifetime_mean = mean(temp_nuc_lifetime)


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

