import numpy as np
from Reduced_Gillespie_simulation import Simulation_Core
import os
import datetime
import sys

# Define the rates and concentrations
K_UNWRAP_values = float(sys.argv[1])
K_WRAP_values = float(sys.argv[2])
K_ADS_values = 2113 # k_on 2113 from bull protamine value, otherwise from diffusion calculation it is 3500
K_DES_values = 0.23 # 0.68*k_ads, otherwise from the bull protamine experiment it is 0.23
P_CONC_values = float(sys.argv[3])
Cooperativity = float(sys.argv[4])
Nucleosomes_per_sim = 1
Sim_steps = 1000000
Simulation_repeat= 1000
NUC_BIND = 14
one_nucleosome_breathing = True ## use when need to keep the histone attached to the DNA 

# RESULT_DIR = r"C:\Users\maya620d\PycharmProjects\Spermatogensis\Output\Experiment\Nucleosome_Breathing\All\/"
RESULT_DIR = r"/group/cmcb-files/pol_schiessel/05_Projekte/manish/Spermatogensis/results/Output/Experiment/Nucleosome_Breathing/All/"
# Get the current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")



if __name__ == '__main__':

    # Create an instance of Simulation_Coreq
    Simulation_Core_instance = Simulation_Core(K_UNWRAP=K_UNWRAP_values, K_WRAP=K_WRAP_values, K_ADS=K_ADS_values, K_DES=K_DES_values,
                                            P_CONC=P_CONC_values, COOPERATIVITY=Cooperativity, 
                                            System_Nucleosomes=Nucleosomes_per_sim, 
                                            Simulation_steps=Sim_steps, 
                                            N_Independent_nucleosomes=Simulation_repeat, 
                                            NUC_BIND=14, 
                                            ONE_NUCLEOSOME_BREATHING=one_nucleosome_breathing)

    # Initialize lists to store the trajectories, times, and nucleosome lifetimes
    all_trajectories = []
    all_times = []
    all_nuc_lifetimes = []


    
    # Create the directory path with the date
    directory = RESULT_DIR + f'{current_date}/K_UNWRAP={K_UNWRAP_values:.2f}_K_WRAP={K_WRAP_values:.2f}_P_CONC={P_CONC_values:.2f}_CO_{Cooperativity}_NUC_{Nucleosomes_per_sim}_REP_{Simulation_repeat}'

    if not os.path.exists(directory):
        os.makedirs(directory)



    # Run the simulation and store the results
    results = list(Simulation_Core_instance.parallel_execute_simulation())
    run_count = 1
    for result in results:
        nuc_lifetime, closed_evol, bound_evol, times, nuc_state = result
        all_trajectories.append(closed_evol)
        all_times.append(times)
        all_nuc_lifetimes.append(nuc_lifetime)
        # Save the closed_evol and times as a numpy file
        np.save(f'{directory}/{run_count}_closed_evol.npy', closed_evol)
        np.save(f'{directory}/{run_count}_times.npy', times)
        np.save(f'{directory}/{run_count}_nuc_lifetime.npy', nuc_lifetime)
        np.save(f'{directory}/{run_count}_bound_evol.npy', nuc_state)
        run_count += 1
