import numpy as np 
from methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness, soft_free_energy


genstiff = GenStiffness(method='hybrid')   # alternatively you can use the 'crystal' method for the Olson data

seq  = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"

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

fn = 'MDParams/Kmat_nucleosome.npy'
Kmat = np.load(fn)

Flefts  = []
Frights = []

print('F        F_enth  F_entr')
for i in range(15):
    left_open  = i*2 
    right_open = 0  
    
    K_resc = np.copy(Kmat)
    Fdict = soft_free_energy(groundstate,stiffmat,left_open,right_open,nuctriads,K_resc)
    print('%.2f   %.2f   %.2f'%(Fdict['F']-Fdict['F_free'],Fdict['F_enthalpy'],Fdict['F_entropy']-Fdict['F_free']))
    
    Flefts.append(Fdict)

print('F        F_enth  F_entr')
for i in range(15):
    left_open  = 0
    right_open = i*2   
    
    K_resc = np.copy(Kmat)
    Fdict = soft_free_energy(groundstate,stiffmat,left_open,right_open,nuctriads,K_resc)
    print('%.2f   %.2f   %.2f'%(Fdict['F']-Fdict['F_free'],Fdict['F_enthalpy'],Fdict['F_entropy']-Fdict['F_free']))
    
    Frights.append(Fdict)