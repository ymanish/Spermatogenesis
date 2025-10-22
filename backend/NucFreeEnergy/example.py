import sys, os
num_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness


def right_handed_nuc(triads):
    rh = np.copy(triads)
    print(rh[0])
    rh[:,2] *= -1
    # rh[:,:,1] *= -1
    # rh = rh[::-1]
    print(rh[0])
    print(np.linalg.det(rh[0]))
    return rh
    
    # print(rh.shape)
    # sys.exit()
    
    
def revert_sense(seq):
    return ''.join([{'A':'T','T':'A','C':'G','G':'C'}[b] for b in seq[::-1]])


genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data
# genstiff = GenStiffness(method='md')   # alternatively you can use the 'crystal' method for the Olson data
seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
# seq601 = "GTAGCCCCGATCGATCGATCGCGCGATCTAGCTATATAAAAATCGCGGGCGATCTATTTTAGAGATCCTCTAAACCCGCATTCGCTCGCGCGCGCGCGATCTTATCTAGCTAGTACGATCGAAACTATCTAGCGACGATCATAAACG"
# seq601 = "GTAGCCCCGATCGATCGATCGCGCGATCTAGCTATATAAAAAGCGCGCGCGCGATCTTATCTAGCTAGTACGATCGAAACTATCTAGCGACGATCATAAACGTCGCGGGCGATCTATTTTAGAGATCCTCTAAACCCGCATTCGCTC"
# seq601 = "AAAAGCGCGCGCGCGATCTTATCTAGCTAGTACGATCGAAACTATCTAGCGACGATCATAAACGTCGCGGGCGATCTATTTTAGAGATCCTCTAAACCCGCATTCGCTCGTAGCCCCGATCGATCGATCGCGCGATCTAGCTATATA"

seq601  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"
# seq601 = 'TTCCACATGGATAATACAAGAGATTCATCGACGTGCTCATTTGGCATTAGGGCATCATCCTAATGAGATTCGGTGGCGCTAACAACTTCGCTGAAAGATCAGTGGAGCGAACTGCCCTACTGTTAATTGGGTACCAGACCTCCTCAC'

# seq601 = 'ATTTGGCCTTAAAAAAACTTCCCCCTTCGCTATACAAGAGATTCATCGGAAAGATCAGTGGAGCGAACTGCCCTACATCATCCTAATGAGATTCGGTGCTGTTAATTGGGTACCAGACTTCCACGCGAAAAAATCGCGGGGGCACGA'
    

seq = seq601

revseq = revert_sense(seq)

print(seq)
print(revseq)


stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)
rev_stiffmat,rev_groundstate = genstiff.gen_params(revseq,use_group=True)




triadfn = 'methods/State/Nucleosome.state'
nuctriads = read_nucleosome_triads(triadfn)



midstep_constraint_locations = [
    2, 6, 14, 17, 24, 29, 
    34, 38, 45, 49, 55, 59, 
    65, 69, 76, 80, 86, 90, 
    96, 100, 107, 111, 116, 121, 
    128, 131, 139, 143
]


# Fdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads,use_correction=False)
# # print(Fdict)

# gs = Fdict['gs']
# gs = Fdict['gs'].reshape((len(gs)//6,6))
# np.save('Data/601gs_theory_nc.npy',gs)

# Fdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads,use_correction=True)

# gs = Fdict['gs']
# gs = Fdict['gs'].reshape((len(gs)//6,6))
# np.save('Data/601gs_theory.npy',gs)

# sys.exit()


val = 'F'
# val = 'F_entropy'
# val = 'F_enthalpy'
# val = 'F_freedna'


Fdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads,use_correction=True)
print(Fdict[val])

Fdict = nucleosome_free_energy(rev_groundstate,rev_stiffmat,midstep_constraint_locations,nuctriads,use_correction=True)
print(Fdict[val])

print('#################################################')
print('#################################################')

for open in range(28):
    # print(f'open = {open}')
    
    cms = midstep_constraint_locations[open:]

    Fdict = nucleosome_free_energy(groundstate,stiffmat,cms,nuctriads,use_correction=True)
    # print(Fdict[val])
    v1 = Fdict[val]

    cms = midstep_constraint_locations[:len(midstep_constraint_locations)-open]
    Fdict = nucleosome_free_energy(rev_groundstate,rev_stiffmat,cms,nuctriads,use_correction=True)
    # print(Fdict[val])
    v2 = Fdict[val]

    diff = v1-v2
    
    print(f'difference = {diff}')