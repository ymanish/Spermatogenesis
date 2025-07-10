#/src/scripts/run_simulation.py


import concurrent.futures
import time

from src.core.gillespie import Simulation



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











