import sys
import os
import datetime
sys.argv[1]
Param_ind = int(sys.argv[1])
NUC_BIND = 14
K_UNWRAP = float(sys.argv[2]) # 4
K_WRAP = float(sys.argv[3]) # 21
K_ADS = float(sys.argv[4]) # 2
K_DES = float(sys.argv[5]) # 23
P_CONC = float(sys.argv[6]) # in micro molar
COOPERATIVITY = float(sys.argv[7]) # KT
N_Independent_nucleosomes = 100 ## Number of nucleosomes to run for the simulation, independently

Simulation_steps = 2000
System_Nucleosomes = 100 ### Number of nuclesomes to include in a single simulation. Right now we do Mono nucleosomes, later need to increase to multiple nucleosomes with cooperativity effect among nucleosomes.

RUN_DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


RESULT_DIR = r"C:\Users\maya620d\PycharmProjects\Spermatogensis\Output\/"
STATE_RECORD_DIR = RESULT_DIR+RUN_DATE+'\/'
BREATHING_DIR = RESULT_DIR+RUN_DATE+'\Breathing_plots\/' 
SIMULATION_PLOTS_DIR = RESULT_DIR + RUN_DATE + "\simulation_plots\/" ### Directory to save the simulation plots for each of the run. It will create the evolution of nucleosome state over time for each run.'''
NUCLEOSOME_STATE_RECORD_DIR = RESULT_DIR + RUN_DATE + "\Derivate_state_record\/" 



