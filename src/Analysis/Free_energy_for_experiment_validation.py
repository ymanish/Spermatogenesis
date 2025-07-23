import sys
sys.path.append(r"C:\Users\maya620d\PycharmProjects\Spermatogensis\/")
# sys.path.append("/home/pol_schiessel/maya620d/Nucleosome_Free_Energy/")

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

def generate_sliding_sequence(sequence, seq_id):
    sliding_sequences = dict()
    window_size = 57
    final_length = 147
    padding = 'A'

    for i in range(len(sequence) - window_size + 1):
        sliding_seq = sequence[i:i+window_size]
        prefix = sequence[:i]
        suffix = sequence[i+window_size:]

        # Calculate the number of 'AT' sequences needed on each side
        total_padding_needed = final_length - len(sliding_seq)
        padding_needed_each_side = total_padding_needed // 2
        remainder_padding = total_padding_needed % 2

        # Create the padded sequence
        padded_sequence = (padding * (padding_needed_each_side // len(padding))) + sliding_seq + (padding * (padding_needed_each_side // len(padding)))
        left_padding = True
        for k in range(remainder_padding):
            if left_padding:
                padded_sequence = padding + padded_sequence
                left_padding=False
            else:
                padded_sequence = padded_sequence + padding
                left_padding=True
        # if len(padded_sequence) != final_length:
        #     padded_sequence = 'A'+ padded_sequence
        

        sequence_temp_array = [prefix, padded_sequence, suffix]

        sliding_sequences[seq_id+'_'+str(i)] = sequence_temp_array
    return sliding_sequences


def calculate_energy(seq, seq_subid, hang_seq_list, left_end=0, right_end=13):
    # logging.info('Starting calculation for sequence: %s', seq)
    nucleosomebreath  = NucleosomeBreath(nuc_method='hybrid', hang_dna_method='md')
    Free_energy, entropy, enthalpy, free_DNA_Free_energy = nucleosomebreath.calculate_free_energy(seq601=seq,
                                                                        site_loc=nucleosomebreath.select_phosphate_bind_sites(left=left_end, right=right_end))
    
    dimers_dict = nucleosomebreath.genstiff_hang.dimers
    hang_dna_energy = 0

    for s in hang_seq_list:
        if len(s) == 0:
            continue
        else:
            hang_dna_energy = hang_dna_energy + calculate_free_dna_energy(h_seq=s, 
                                                            dimers=dimers_dict)
    
    # logging.info('Free energy: %s, entropy: %s, enthalpy: %s', Free_energy, entropy, enthalpy)
    return seq_subid, Free_energy, entropy, enthalpy, free_DNA_Free_energy, hang_dna_energy

def calculate_free_dna_energy(h_seq:str, dimers:dict):

    kBT = 1
    sum_log_det_l = 0
    N = len(h_seq)-1
    for i in range(N):
        bp = h_seq[i:i+2].upper()
        pstiff = dimers[bp]['stiff']
        det_value = np.linalg.det(pstiff)

        sum_log_det_l = sum_log_det_l + np.log(det_value)


    F_free_DNA_h = kBT * sum_log_det_l / 2

    return F_free_DNA_h

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


    param_id =sys.argv[1]
    seq_id = str(sys.argv[2])
    # gc_content = np.linspace(2/147, 145/147, 144)
    # print(gc_content)
    # sys.exit()
    sequences = str(sys.argv[3])

    sequences_dic= generate_sliding_sequence(sequences, seq_id)

    Energy_df = pd.DataFrame()
    Seq_id_list = []
    Seq_list = []
    Free_energy_list = []
    Entropy_list = []
    Enthalpy_list = []
    Free_DNA_energy = []
    Hang_DNA_E = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        pool = []
        for key, values in sequences_dic.items():
            print(key, len(values))
            pool.append(executor.submit(calculate_energy, seq=values[1], 
                                        hang_seq_list=[values[0], values[2]],
                                          seq_subid=key, left_end=4, right_end=9))
        # for val in executor.map(calculate_energy_gc, sequences):
        #     yield (val)
        plot_cnt = 0
        for j in concurrent.futures.as_completed(pool):
            print('Simulation instance completed:', plot_cnt)
            id, Free_energy, entropy, enthalpy, free_DNA_Free_energy, hang_DNA_energy = j.result()
            plot_cnt += 1
            Seq_id_list.append(id)
            Seq_list.append(sequences_dic[id])
            Free_energy_list.append(Free_energy)
            Entropy_list.append(entropy)
            Enthalpy_list.append(enthalpy)
            Free_DNA_energy.append(free_DNA_Free_energy)
            Hang_DNA_E.append(hang_DNA_energy)
    
    Energy_df['ID'] = Seq_id_list
    Energy_df['Sequence'] = Seq_list
    Energy_df['Free_energy'] = Free_energy_list
    Energy_df['Entropy'] = Entropy_list
    Energy_df['Enthalpy'] = Enthalpy_list
    Energy_df['Free_DNA_energy'] = Free_DNA_energy
    Energy_df['Hang_DNA_energy'] = Hang_DNA_E

    print(Energy_df)

    Energy_df.to_csv(str(param_id) + '_Tetramer_Free_energy.csv')
    # Energy_df.to_csv('/group/pol_schiessel/05_Projekte/manish/Tetramer_Nucleosome_Free_Energy_Hybrid_60_A/' + str(param_id) + '_Tetramer_Free_energy.csv')

    end = time.perf_counter()
    print(f'Finished in {round(end - start, 2)} second(s)')
