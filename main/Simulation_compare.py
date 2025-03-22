import matplotlib.pyplot as plt
import numpy as np
from Reduced_Gillespie_simulation import Simulation_Core
import seaborn as sns
import os
import pandas as pd
import datetime

# Define the rates and concentrations
K_UNWRAP_values = [210]
K_WRAP_values = [2000, 2500, 2600, 2700, 3000]
K_ADS_values = 2113
K_DES_values = 0.23
P_CONC_values = [0.0]
Nucleosomes_per_sim = 1
Sim_steps = 10000
Simulation_repeat= 1
NUC_BIND = 14
one_nucleosome_breathing = True
RESULT_DIR = r"C:\Users\maya620d\PycharmProjects\Spermatogensis\Output\Experiment\Nucleosome_Breathing\All\/"
# Get the current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# # Calculate the total number of subplots
# n_subplots = len(K_UNWRAP_values) * len(K_WRAP_values) * len(P_CONC_values)

# # Calculate the number of rows and columns for the subplots
# n_rows = int(np.sqrt(n_subplots))
# n_cols = n_subplots // n_rows + (n_subplots % n_rows > 0)

# # Initialize the figure
# fig, axs = plt.subplots(n_rows, n_cols)

# # Flatten the axs array
# axs = axs.flatten()

# Initialize a counter for the current subplot
subplot_counter = 0


if __name__ == '__main__':
    num_rows = len(K_UNWRAP_values) * len(P_CONC_values)
    num_cols = len(K_WRAP_values)
    for k, P_CONC in enumerate(P_CONC_values):
        for i, K_UNWRAP in enumerate(K_UNWRAP_values):
            for j, K_WRAP in enumerate(K_WRAP_values):
                fig, ax = plt.subplots()
                fig2, ax2 = plt.subplots()

                # Create an instance of Simulation_Core
                Simulation_Core_instance = Simulation_Core(K_UNWRAP=K_UNWRAP, K_WRAP=K_WRAP, K_ADS=2113, K_DES=0.23,
                                                        P_CONC=P_CONC, COOPERATIVITY=0, 
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
                directory = RESULT_DIR + f'{current_date}/K_UNWRAP={K_UNWRAP:.2f}_K_WRAP={K_WRAP:.2f}_P_CONC={P_CONC:.2f}'

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
                    np.save(f'{directory}/{run_count}/closed_evol.npy', closed_evol)
                    np.save(f'{directory}/{run_count}/times.npy', times)
                    np.save(f'{directory}/{run_count}/nuc_lifetime.npy', nuc_lifetime)
                    run_count += 1




                # Calculate the mean trajectory and times
                mean_trajectory = [np.mean([trajectory[i] for trajectory in all_trajectories if i < len(trajectory)]) for i in range(max(len(trajectory) for trajectory in all_trajectories))]
                mean_times = [np.mean([times[i] for times in all_times if i < len(times)]) for i in range(max(len(times) for times in all_times))]


                # print(all_nuc_lifetimes)

                # Plot the trajectories and the mean trajectory
                for trajectory, times in zip(all_trajectories, all_times):
                    ax.plot(times, trajectory, color='gray', alpha=0.5)
                ax.plot(mean_times, mean_trajectory, color='red', alpha=0.5, linewidth=2)

                # Add a title to the subplot
                ax.set_title(f'K_UNWRAP={K_UNWRAP}, K_WRAP={K_WRAP}, P_CONC={P_CONC}')

                if one_nucleosome_breathing:
                    
                    # Convert trajectory and times to a dataframe
                    df = pd.DataFrame({'Time': mean_times, 'Trajectory': mean_trajectory})

                    # Do something with the dataframe
                    df['Time_diff'] = df.groupby('Trajectory')['Time'].diff()
                    # Replace nan values in Time_diff with corresponding Time values
                    df['Time_diff'].fillna(df['Time'], inplace=True)
                    # Example: Print the dataframe
                    
                    # print(df)
                    # Calculate the mean Time_diff, variance, and count of values in each group
                    grouped_data = df.groupby('Trajectory')['Time_diff'].agg(
                                                                                mean='mean',
                                                                                var='var',
                                                                                count='count'
                                                                            )
                    # Convert the grouped data to a dataframe
                    grouped_df = pd.DataFrame(grouped_data)



                    print(grouped_df)
                    
                    
                # # Plot the PDF of nucleosome lifetimes
                # sns.kdeplot(all_nuc_lifetimes, ax=ax2)
                # # Calculate the number of real values in all_nuc_lifetimes
                # real_values_count = len([value for value in all_nuc_lifetimes if not np.isnan(value)])

                # # Add the count to the plot
                # ax2.text(0.5, 0.5, str(real_values_count), transform=ax2.transAxes)
                # # Limit the x-axis to non-negative values
                # ax2.set_xlim([0, None])
                # # Create a directory for the plots if it doesn't exist



                # if one_nucleosome_breathing:
                #     # Save the grouped_df as a CSV file
                #     grouped_df.to_csv(f'{directory}/grouped_data.csv', index=True)

                # Save the plots
                fig.savefig(f'{directory}/trajectory.png')
                fig2.savefig(f'{directory}/nucleosome_lifetime.png')

                # Close the plots
                plt.close(fig)
                plt.close(fig2)