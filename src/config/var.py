#src/config/var.py
# Author: MY
# Last Updated: 2025-03-25
import sys
import datetime
import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-ind", type=int, required=True)
    parser.add_argument("--k-unwrap", type=float, required=True)
    parser.add_argument("--k-wrap", type=float, required=True)
    parser.add_argument("--k-ads", type=float, required=True)
    parser.add_argument("--k-des", type=float, required=True)
    parser.add_argument("--p-conc", type=float, required=True)
    parser.add_argument("--cooperativity", type=float, required=True)
    parser.add_argument("--seq-id", type=float, required=True)
    # parser.add_argument("--sequence", type=str, default=seq_601)
    return parser.parse_args()


cli_args = parse_cli_args()
print(cli_args)

Param_ind = cli_args.param_ind
NUC_BIND = 14
K_UNWRAP = cli_args.k_unwrap # 4
K_WRAP = cli_args.k_wrap # 21
K_ADS = cli_args.k_ads # 2
K_DES = cli_args.k_des # 23
P_CONC = cli_args.p_conc # in micro molar
COOPERATIVITY = cli_args.cooperativity # KT
N_Independent_nucleosomes = 1 ## Number of nucleosomes to run for the simulation, independently


Simulation_steps = 5000
System_Nucleosomes = 1 ### Number of nuclesomes to include in a single simulation. Right now we do Mono nucleosomes, later need to increase to multiple nucleosomes with cooperativity effect among nucleosomes.
PLOT_NUC_STATE = True
BREATHING_ONLY =True
SINGLE_NUCLEOSOME_EVOLUTION = True ###

FILE_CHUNK_SIZE = 1000





if __name__== "__main__":


    cli_args = parse_cli_args()
    print(cli_args)

    Param_ind = cli_args.param_ind
    NUC_BIND = 14
    K_UNWRAP = cli_args.k_unwrap # 4
    K_WRAP = cli_args.k_wrap # 21
    K_ADS = cli_args.k_ads # 2
    K_DES = cli_args.k_des # 23
    P_CONC = cli_args.p_conc # in micro molar
    COOPERATIVITY = cli_args.cooperativity # KT


# N_Independent_nucleosomes = 1 ## Number of nucleosomes to run for the simulation, independently

# Simulation_steps = 5000
# System_Nucleosomes = 1 ### Number of nuclesomes to include in a single simulation. Right now we do Mono nucleosomes, later need to increase to multiple nucleosomes with cooperativity effect among nucleosomes.
# PLOT_NUC_STATE = True
# BREATHING_ONLY =True
# SINGLE_NUCLEOSOME_EVOLUTION = True ###

# FILE_CHUNK_SIZE = 1000

# SEQ_id = float(sys.argv[8])

# RUN_DATE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# # seq_601 = 'CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT'


# STD_DEV_NOISE = 0
# LAST_SITE_OPENING_RATE = 0
# SEQUENCE = 'ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT'