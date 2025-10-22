import sys, os
num_threads = 3
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness, calculate_midstep_triads
from binding_model import binding_model_free_energy, binding_model_free_energy_old
from methods.PolyCG.polycg import so3

def revert_sense(seq):
    return ''.join([{'A':'T','T':'A','C':'G','G':'C'}[b] for b in seq[::-1]])


genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
seq601  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"

    
seq = seq601

    
stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)

triadfn = 'methods/State/Nucleosome.state'
nuctriads = read_nucleosome_triads(triadfn)

midstep_constraint_locations = [
    2, 6, 14, 17, 24, 29, 
    34, 38, 45, 49, 55, 59, 
    65, 69, 76, 80, 86, 90, 
    96, 100, 107, 111, 116, 121, 
    128, 131, 139, 143
]


# FOR NOW WE USE THE FIXED MIDSTEP TRIADS AS MU_0
# Find midstep triads in fixed framework for comparison
nuc_mu0 = calculate_midstep_triads(
    midstep_constraint_locations,
    nuctriads
)
        
Kentries = np.array([1,1,1,10,10,10])*1
diags = np.concatenate([Kentries]*len(nuc_mu0))
K1_10 = np.diag(diags)


left_open = 0
right_open = 0

nuc_K_pos_resc_sym    = np.load('MDParams/nuc_K_pos_resc_sym.npy')

print('##################################')
print('Calculate model')
nucout = binding_model_free_energy(
    groundstate,
    stiffmat,    
    nuc_mu0,
    nuc_K_pos_resc_sym,
    left_open=left_open,
    right_open=right_open,
    use_correction=True,
)

alphas = nucout['alphas']

print(alphas)

mus = np.copy(nuc_mu0)
for i in range(len(mus)):
    mus[i] = mus[i] @ so3.se3_euler2rotmat(alphas[i])

