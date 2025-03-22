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
# from Reduced_initialise import initialise
import concurrent.futures
import time
from operator import add
import csv

from analytical_solution import Nucl_Breathing_Sol

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
        self.record_state = np.ones(num_nucleosomes)
        self.N_closed = num_nucleosomes * binding_sites
        self.N_open = 0


    def sequence(self):
        S = 'ACGCGGATCAAATTT'
        return S

    def unwrapping(self, histone_occupied_site):
        unwrapped_sites = dict()
        sum_rate = 0

        if len(histone_occupied_site) == 0:
            return unwrapped_sites, sum_rate

        elif histone_occupied_site[0] != histone_occupied_site[-1]:
            unwrapped_sites[histone_occupied_site[0]] = self.k_unwrap
            unwrapped_sites[histone_occupied_site[-1]] = self.k_unwrap

        else:
            unwrapped_sites[histone_occupied_site[0]] = self.k_unwrap


        sum_rate = sum(unwrapped_sites.values())

        return unwrapped_sites, sum_rate

    def rewrapping(self, nuce, histone_occupied_site, keep_histone=False):
        rewrapped_sites = dict()
        sum_rate = 0

        if keep_histone and len(histone_occupied_site) == 0:
            available_site = np.ravel(np.where(nuce == 1))
            for i in available_site:
                rewrapped_sites[i] = self.k_wrap

        elif len(histone_occupied_site) > 0:
          # Check the left of the first entry
            if histone_occupied_site[0] > 0 and nuce[histone_occupied_site[0] - 1] == 1:
                rewrapped_sites[histone_occupied_site[0] - 1] = self.k_wrap

            # Check the right of the last entry
            if histone_occupied_site[-1] < len(nuce) - 1 and nuce[histone_occupied_site[-1] + 1] == 1:
                rewrapped_sites[histone_occupied_site[-1] + 1] = self.k_wrap

    
        if len(rewrapped_sites) > 0:
            sum_rate = sum(rewrapped_sites.values())
    
        return rewrapped_sites, sum_rate


class protamines:
    def __init__(self, k_unbind, k_bind, p_conc, cooperativity=1):

        """defines the unbinding rate of protamine for each bound site
        :param k_unbind: unbinding rate
        :param k_bind: binding rate
        :param p_conc: protamine concentration in microMolar
        :param cooperativity: cooperativity factor"""



        self.k_unbind = k_unbind
        self.k_bind = (k_bind* math.pow(10, -2))/6 # the binding rate is provided in microMolar^-1s^-1. To convert it to s^-1*(molecules/micrometer^3)^-1, multiply by 10^15 for conversion to micrometer^3 from liter, 
                                                    # then devide by 6*10^23*10^-6 for conversion to molecules from micromolar to molecules. OR simply mutiply by (1/6)*1e-2

        
        self.cooperativity = cooperativity
        self.p_conc = p_conc * math.pow(10, -6)  # convert from micro_molar to Molar which is also moles/liter
        self.P_free = self.p_conc * 6 * math.pow(10, 23) * math.pow(10, -15)  # initial concentration of protamines molecules/cubic_micrometer
        self.N_bound = 0
        print('Protamine binding rate and molecules' , self.k_bind, self.P_free)
        # self.k_ratio = 0.00244
        self.k_ratio = self.k_bind/self.k_unbind 

    def protein_binding(self, open_sites):
        bind_sites = dict()
        # neigh_dict=dict()
        sum_rate = 0
        if len(open_sites) != 0:
            for i in open_sites:
                bind_sites[i] = self.k_bind* self.P_free
                # print(self.P_free)
            
            sum_rate = sum(bind_sites.values())

        return bind_sites, sum_rate


    def protein_unbinding(self, bound_sites):
        """defines the unbinding rate of protamine for each bound site
        :param bound_sites: contains the indexes for the protamine bound site
         :return: dictionary with key as the index of the bound sites and value as its unbinding rate.
                    Also returns the sum of all the unbinding rates"""
    
        unbound_sites = dict()
        sum_rate = 0
    
        if len(bound_sites) != 0:
            for i in bound_sites:
                unbound_sites[i] = self.k_unbind

            sum_rate = sum(unbound_sites.values())
    
        return unbound_sites, sum_rate

    def delta_E(self, p, J, s_l=0, s_r=0):
        ### get the energy required to move from a bound state to unbound state
        ## s_l and s_r are the left and right neighbours to the bound site.
        ## refer to the task documentation for the derivation of the formulae. The energy here is in terms of Kb*T.

        # U = np.log(0.00244 * p)  ## in terms of KT
        U = np.log(self.k_ratio * p)  ## in terms of KT

        #     J=12.6/(1-s_l-s_r)

        #     return (U + J)-J*0.5*(s_l+s_r)
        return J * (s_l + s_r) + U

    def get_spin_value(self, v):
        if v == 2:
            return 1
        else:
            return 0

    ##### With Coooperativity, here the cooperativity will make the protamine unbinding difficult.
    def protein_unbinding_coop(self, nuce, pt_indexes, B=1):
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
    def __init__(self, nucleosme_instance, protamines_instance,  num_nucleosomes, N=1000, ONE_NUCLEOSOME_BREATHING=False):
        self.nucleosme = nucleosme_instance
        self.protamines = protamines_instance
        self.N = N # Number of steps
        self.t = 0
        self.num_nucleosomes = num_nucleosomes
        self.nuc_fall_flag = False
        self.only_one_nucleosome_breathing = ONE_NUCLEOSOME_BREATHING
        # self.uniform_pos_arg_njit = nb.njit(self.uniform_pos_arg)

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
        return nucleosome_fall_time, N_closed_array, N_bound_array, times, self.nucleosme.nucleosomes

    
    

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

class Simulation_Core():

    def __init__(self, K_UNWRAP, K_WRAP, K_ADS, K_DES, P_CONC, COOPERATIVITY, System_Nucleosomes, Simulation_steps, N_Independent_nucleosomes, wait_step=False, NUC_BIND=14, ONE_NUCLEOSOME_BREATHING=False):
        self.K_UNWRAP = K_UNWRAP
        self.K_WRAP = K_WRAP
        self.K_ADS = K_ADS
        self.K_DES = K_DES
        self.P_CONC = P_CONC
        self.COOPERATIVITY = COOPERATIVITY
        self.System_Nucleosomes = System_Nucleosomes
        self.Simulation_steps = Simulation_steps
        self.N_Independent_nucleosomes = N_Independent_nucleosomes
        self.NUC_BIND = NUC_BIND
        self.ONE_NUCLEOSOME_BREATHING = ONE_NUCLEOSOME_BREATHING
        self.wait_step = wait_step

    def simulation_fn(self, index):
        nucleosme_instance = nucleosme(k_unwrap=self.K_UNWRAP,
                                        k_wrap=self.K_WRAP , 
                                        num_nucleosomes=self.System_Nucleosomes,
                                        binding_sites=self.NUC_BIND)
        
        protamines_instance = protamines(k_unbind=self.K_DES, 
                                         k_bind=self.K_ADS, 
                                         p_conc=self.P_CONC , 
                                         cooperativity=self.COOPERATIVITY)
        
        simulation = Simulation(nucleosme_instance, 
                                protamines_instance, 
                                num_nucleosomes=self.System_Nucleosomes, 
                                N=self.Simulation_steps,
                                ONE_NUCLEOSOME_BREATHING=self.ONE_NUCLEOSOME_BREATHING)

        print('Simulation instance created for index:', index)
        return simulation 


    def parallel_execute_simulation(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = []
            for i in range(self.N_Independent_nucleosomes):
                print(i)
                pool.append(executor.submit(self.simulation_fn(i).simulate_main))

            plot_cnt = 0
            for j in concurrent.futures.as_completed(pool):
                print('Simulation instance completed:', plot_cnt)
                nuc_lifetime, closed_evol, bound_evol, times, nuc_state = j.result()
                yield (nuc_lifetime, closed_evol, bound_evol, times, nuc_state)
                plot_cnt += 1


if __name__== '__main__':

    start = time.perf_counter()


    # Initialize a list to store all trajectories
    all_trajectories = []
    all_times = []

    all_y_values = []
    all_nuc_lifetimes = []
    
    period_plot = 1

    

    Simulation_Core_instance = Simulation_Core(K_UNWRAP=250, 
                                               K_WRAP=350, 
                                               K_ADS=2113, 
                                               K_DES=0.23,
                                                P_CONC=0, 
                                                COOPERATIVITY=0, 
                                                System_Nucleosomes=100, 
                                                Simulation_steps=100000, 
                                                N_Independent_nucleosomes=1, 
                                                wait_step=True,
                                                NUC_BIND=14, 
                                                ONE_NUCLEOSOME_BREATHING=False)
    



    for result in Simulation_Core_instance.parallel_execute_simulation():
        nuc_lifetime, closed_evol, bound_evol, times, nuc_state = result
        
        
        # Add the trajectory and times array to the lists
        all_trajectories.append(closed_evol[::period_plot])
        all_times.append(times[::period_plot])

        # Add the y-values for the new plot to the list
        all_y_values.append(list(range(Simulation_Core_instance.System_Nucleosomes, Simulation_Core_instance.System_Nucleosomes - len(nuc_lifetime), -1)))
        all_nuc_lifetimes.append(nuc_lifetime)


    all_trajectories = np.array(all_trajectories)
    all_times = np.array(all_times)
    all_nuc_lifetimes = np.array(all_nuc_lifetimes)
    all_y_values = np.array(all_y_values)/Simulation_Core_instance.System_Nucleosomes


    print(all_trajectories)

    print(all_y_values)
    print(all_nuc_lifetimes[0,:])


    ##### Analytical Solution Survival Probability
    # L_eff = 13.5
    # L=L_eff-0.5
    L=14
    # N_points =14
    ku = Simulation_Core_instance.K_UNWRAP
    kr = Simulation_Core_instance.K_WRAP
    # ku=210
    # kr=360
    ka = Simulation_Core_instance.K_ADS
    kd = Simulation_Core_instance.K_DES
    p = Simulation_Core_instance.P_CONC
    # n=4
    # x_eff=13.5
    # x0 = x_eff - 0.5
    x0 = 14
    # delta_x = 3.6*1e-9
    delta_x = 1

    # delta_t = 1e-3
    # v=2*(kr-ku)*(delta_x)
    v=2*(kr-ku)*(1- delta_x/(2*L))

    c2=(kr+ku)*(delta_x**2)
    
    t_array = all_nuc_lifetimes[0,:]
    # t_array=np.append(t_array, [0.0001])
    nuc_breath = Nucl_Breathing_Sol(v=v, 
                                    c2=c2, 
                                    L=L, 
                                    t_values=t_array, 
                                    x_0=x0, gamma=0, N_alpha=200)
    
    analytical_S = nuc_breath.analytical_survival_probability()
    print(analytical_S)


    
    # Functions to calculate phi_0
    def calculate_phi0_no_cooperativity(k_a, k_d, p):
        return (k_a * p) / (k_a * p + k_d)

    # def calculate_phi0_cooperativity( k_a, k_d, p, gamma):

    #     def equation(phi):
    #         return k_a * p * (1 - phi) - k_d * np.exp(-gamma * phi) * phi
    #     phi0_initial_guess = 0.5
    #     phi_0_solution, = fsolve(equation, phi0_initial_guess)
    #     phi_0_solution = np.clip(phi_0_solution, 0, 1)
    #     return phi_0_solution



    def phi_hill_equation( p, K_D, n):
        return p**n / (K_D + p**n)



     ### No cooperativity
    print('No cooperativity')
    phi0_no_cooperativity = calculate_phi0_no_cooperativity(ka, kd, p)
    print('phi ', phi0_no_cooperativity)
    k_r_eff_no_cooperativity = kr * (1 - phi0_no_cooperativity)
    print('kr_no_coop' , k_r_eff_no_cooperativity)
    print('ku', ku)
    print('kr', kr)

    beta_no_cooperativity = 2 * (k_r_eff_no_cooperativity - ku)
    print(beta_no_cooperativity)
    c2_no_cooperativity = ku + k_r_eff_no_cooperativity
    print(c2_no_cooperativity)

    nuc_breath_no_cooperativity = Nucl_Breathing_Sol(v=beta_no_cooperativity, 
                                c2=c2_no_cooperativity, 
                                L=L, 
                                t_values=t_array, 
                                x_0=x0 , gamma=0, N_alpha=200)

    analytical_S_no_cooperativity = nuc_breath_no_cooperativity.analytical_survival_probability()
    print(analytical_S_no_cooperativity)
    ### With cooperativity
    k_D = kd/ka

    phi0_cooperativity = phi_hill_equation(p,k_D, n=4)

    k_r_eff_cooperativity = kr * (1 - phi0_cooperativity)
    beta_cooperativity = 2 * (k_r_eff_cooperativity - ku)
    c2_cooperativity = ku + k_r_eff_cooperativity

   
    nuc_breath_cooperativity = Nucl_Breathing_Sol(v=beta_cooperativity, 
                                c2=c2_cooperativity, 
                                L=L, 
                                t_values=t_array, 
                                x_0=x0, gamma=0, N_alpha=200)

    analytical_S_cooperativity = nuc_breath_cooperativity.analytical_survival_probability()

    # Calculate the mean trajectory and times array
    mean_trajectory = [np.mean([trajectory[i] for trajectory in all_trajectories if i < len(trajectory)]) for i in range(max(len(trajectory) for trajectory in all_trajectories))]
    mean_times = [np.mean([times[i] for times in all_times if i < len(times)]) for i in range(max(len(times) for times in all_times))]
    
    
    # # Create first plot
    # plt.figure(figsize=(10, 6))
    # for k in range(len(all_trajectories)):
    #     print(k)
    #     sns.lineplot(x=all_times[k], y=all_trajectories[k], alpha=0.5, color='gray')
    # # sns.lineplot(x=times[::period_plot], y=closed_evol[::period_plot], color='gray')
    # sns.lineplot(x=mean_times, y=mean_trajectory, color='red')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Total Number of binding sites bound to histone')
    # plt.show()

    # Calculate the mean y-values for the new plot
    mean_y_values = [np.mean([y_values[i] for y_values in all_y_values if i < len(y_values)]) for i in range(max(len(y_values) for y_values in all_y_values))]
    mean_nuc_lifetime = [np.mean([nuc_lifetime[i] for nuc_lifetime in all_nuc_lifetimes if i < len(nuc_lifetime)]) for i in range(max(len(nuc_lifetime) for nuc_lifetime in all_nuc_lifetimes))]

    # Create second plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=all_nuc_lifetimes[0,:], y=all_y_values[0,:], color='black')
    sns.lineplot(x=mean_nuc_lifetime, y=mean_y_values, color='red')
    sns.lineplot(x=t_array, y=analytical_S, color='blue')
    plt.xlabel('Nucleosome lifetime')
    plt.ylabel('Survival Nucleosome')
    plt.show()



    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=all_nuc_lifetimes[0,:], y=all_y_values[0,:], color='black')
    sns.lineplot(x=mean_nuc_lifetime, y=mean_y_values, color='red')
    sns.lineplot(x=t_array, y=analytical_S_no_cooperativity, color='blue')
    plt.xlabel('Nucleosome lifetime')
    plt.ylabel('Survival Nucleosome')
    plt.show()
    end = time.perf_counter()
    print(f'Finished in {round(end - start, 2)} second(s)')









