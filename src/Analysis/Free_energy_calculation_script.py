import sys
sys.path.append(r"C:\Users\maya620d\PycharmProjects\Spermatogensis\/")

from nucleosome_breath import NucleosomeBreath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import concurrent.futures
import random
import logging
import time

logging.basicConfig(level=logging.INFO)


def calculate_gc_content(sequence):
    total_bases = len(sequence)
    gc_bases = sequence.count('G') + sequence.count('C')
    gc_content = (gc_bases / total_bases) * 100
    return gc_content


def calculate_energy_gc(seq, method):
    # logging.info('Starting calculation for sequence: %s', seq)
    # nucleosomebreath  = NucleosomeBreath(nuc_method=method, hang_dna_method='md')
    nucleosomebreath  = NucleosomeBreath(nuc_method=method)
    gc_content = calculate_gc_content(seq)
    # logging.info('GC content for sequence: %s is %s', seq, gc_content)
    Free_energy, entropy, enthalpy, free_DNA_Free_energy= nucleosomebreath.calculate_free_energy(seq601=seq,
                                                                        site_loc=nucleosomebreath.select_phosphate_bind_sites(left=0, right=13))
    # logging.info('Free energy: %s, entropy: %s, enthalpy: %s', Free_energy, entropy, enthalpy)
    return gc_content, Free_energy, entropy, enthalpy, free_DNA_Free_energy, seq

    

def sequences_list(gc):
    # # Define the possible nucleotides
    # nucleotides = ['A', 'C', 'G', 'T']

    # Define the GC content
    
    # Generate the random DNA sequences
    sequences = []
    # for gc in gc_content:
    for _ in range(3):
        GC_list = random.choices(['G', 'C'], k=int(147*gc))
        AT_list = random.choices(['A', 'T'], k=147-int(147*gc))
        seq_list = GC_list + AT_list
        random.shuffle(seq_list)
        seq = ''.join(seq_list)
        

        # seq = ''.join( + random.choices(['A', 'T'], k=147-int(147*gc)))
        # random.shuffle(seq)
        sequences.append(seq)

    return sequences


if __name__=='__main__':
    start = time.perf_counter()
    random.seed(42)
    param_id =sys.argv[1]
    # gc_content = np.linspace(2/147, 145/147, 144)
    # print(gc_content)
    # sys.exit()
    # random.seed(42)  # Set the seed for reproducibility
    # sequences = sequences_list(float(sys.argv[2]))
    directory_path =  r"C:\Users\maya620d\PycharmProjects\Spermatogensis\main\Analysis\random_sequences"

    file_path = os.path.join(directory_path, f"random_sequences_{sys.argv[2]}.txt")
    method = sys.argv[3]

    if len(sys.argv) < 5:
        MAX_WORKERS = 8
    else:
        MAX_WORKERS = int(sys.argv[4])


    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            sequences = [line.strip() for line in file.readlines()]
    else:
       print(file_path, " File does not exist")

    # sequences = eval(sys.argv[2])


    Energy_df = pd.DataFrame()
    GC_list = []
    Free_energy_list = []
    Entropy_list = []
    Enthalpy_list = []
    Free_DNA_energy = []
    Sequences_listing = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pool = []
        for i in range(len(sequences)):
            print(i)
            pool.append(executor.submit(calculate_energy_gc, sequences[i], method))
        

        # for val in executor.map(calculate_energy_gc, sequences):
        #     yield (val)
        plot_cnt = 0
        for j in concurrent.futures.as_completed(pool):
            print('Simulation instance completed:', plot_cnt)
            gc_content, Free_energy, entropy, enthalpy, free_DNA_Free_energy, sequ = j.result()
            plot_cnt += 1
            GC_list.append(gc_content)
            Free_energy_list.append(Free_energy)
            Entropy_list.append(entropy)
            Enthalpy_list.append(enthalpy)
            Free_DNA_energy.append(free_DNA_Free_energy)
            Sequences_listing.append(sequ)

    Energy_df['GC_content'] = GC_list
    Energy_df['Free_energy'] = Free_energy_list
    Energy_df['Entropy'] = Entropy_list
    Energy_df['Enthalpy'] = Enthalpy_list
    Energy_df['Free_DNA_energy'] = Free_DNA_energy
    Energy_df['Sequences'] = Sequences_listing

    if method == 'Hybrid':
        Energy_df.to_csv('/group/pol_schiessel/05_Projekte/manish/Nucleosome_Free_Energy_Hybrid_TH_MMC/' + str(param_id) + '_GC_content_vs_Free_energy.csv')
    elif method == 'Crystal':
        Energy_df.to_csv('/group/pol_schiessel/05_Projekte/manish/Nucleosome_Free_Energy_Crystal_TH_MMC/' + str(param_id) + '_GC_content_vs_Free_energy.csv')
    else: ### MD
        Energy_df.to_csv('/group/pol_schiessel/05_Projekte/manish/Nucleosome_Free_Energy_MD_TH_MMC/' + str(param_id) + '_GC_content_vs_Free_energy.csv')

      
    Energy_df.to_csv(str(param_id) + '_GC_content_vs_Free_energy.csv')
    # Energy_df.to_csv('/group/pol_schiessel/05_Projekte/manish/Nucleosome_Free_Energy_Hybrid_TH_MMC/' + str(param_id) + '_GC_content_vs_Free_energy.csv')
    
    end = time.perf_counter()
    print(f'Finished in {round(end - start, 2)} second(s)')
    