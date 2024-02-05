import pandas as pd 
import numpy as np
import math as math


def Free_energy_wrapped_DNA(self, seq, start, end, di_prob_dict, mono_prob_dict):

    KbT = 1
    prob = mono_prob_dict[seq[start]][start]

    if prob == 0:
        prob = 0.25
    for j in range(start + 1, end + 1):

        pfac = di_prob_dict[seq[j - 1:j + 1]][j - 1] / mono_prob_dict[seq[j - 1]][j - 1]
        prob *= pfac

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
    pairs = [seq[i:i + 2] for i in range(len(seq[start:end + 1]) - 1)]
    for p in pairs:

        sum_log_det_l = sum_log_det_l + self.generate_matrix(p)

    F_free_DNA_l = kBT * sum_log_det_l / 2

    return F_free_DNA_l




if __name__ == '__main__':
    # Read in the data
    seq_601 = 'CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT'
    K_eq_601 = {'1': 1e-4, '2': 1e-5, '3': 4e-6, '4': 3e-6, '5': 2e-6, '6': 2e-6, '7': 1e-6, '8': 1e-6, '9': 3e-6, '10': 6e-6, '11':3e-5, '12':5e-5, '13':3e-4, '14':1e-3}

