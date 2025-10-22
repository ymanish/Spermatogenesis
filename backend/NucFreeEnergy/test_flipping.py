import sys, os
num_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness


def flipseq(seq):
    bpdict = {'A':'T','T':'A','C':'G','G':'C'}
    return ''.join([bpdict[b] for b in seq][::-1])

def random_seq(N):
    return ''.join(['ATCG'[np.random.randint(4)] for i in range(N)])

def flip_triads(triads):
    ftr = np.copy(triads)
    ftr = ftr[::-1]
    ftr[:,:,1] *= -1
    ftr[:,:,2] *= -1
    return ftr

genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
triadfn = 'methods/State/Nucleosome.state'
nuctriads = read_nucleosome_triads(triadfn)
fnuctriads = flip_triads(nuctriads)
midstep_constraint_locations = [
    2, 6, 14, 17, 24, 29, 
    34, 38, 45, 49, 55, 59, 
    65, 69, 76, 80, 86, 90, 
    96, 100, 107, 111, 116, 121, 
    128, 131, 139, 143
]




num = 1000

maxdiff = 0

for i in range(num):
    print(f'sequence {i+1}')
    seq = random_seq(147)
    fseq = flipseq(seq)
    
    # print(seq)
    # print(fseq)
    
    use_group = True

    midstep_constraint_locations = [
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111, 116, 121, 
        128, 131, 139, 143
    ]

    stiffmat,groundstate = genstiff.gen_params(seq,use_group=use_group)
    Fdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads,use_correction=True)

    
    stiffmat,groundstate = genstiff.gen_params(seq,use_group=use_group)
    tfFdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,fnuctriads,use_correction=True)
    
    midstep_constraint_locations = [
        2, 6, 14, 17, 24, 29, 
        34, 38, 45, 49, 55, 59, 
        65, 69, 76, 80, 86, 90, 
        96, 100, 107, 111
    ]
    
    stiffmat,groundstate = genstiff.gen_params(fseq,use_group=use_group)
    fFdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads,use_correction=True)

    print(f"free    {Fdict['F_freedna']-fFdict ['F_freedna']}")
    print(f"flipped {Fdict['F']-fFdict ['F']}")
    print(f"trflip  {Fdict['F']-tfFdict ['F']}")
    
    
    # print(Fdict['F_enthalpy']-fFdict ['F_enthalpy'] )
    
    # diff = np.abs(Fdict['F']-fFdict ['F'])
    # diff = np.abs(Fdict['F_freedna']-fFdict ['F_freedna'])
    
    # if diff > maxdiff:
    #     maxdiff = diff
    #     print(maxdiff)
    