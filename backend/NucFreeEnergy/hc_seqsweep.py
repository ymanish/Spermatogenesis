import sys, os
num_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"    # For libraries using OpenMP
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"    # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"  # For OpenBLAS
os.environ["BLIS_NUM_THREADS"] = f"{num_threads}"   # For BLIS

import numpy as np
import sys 
from  methods import nucleosome_free_energy, nucleosome_groundstate, read_nucleosome_triads, GenStiffness


def readseq(seqfn):
    with open(seqfn,'r') as f:
        lines = f.readlines()
        total_seq = ''
        for line in lines:
            if line.strip().upper().replace('A','').replace('T','').replace('C','').replace('G','') == '':
                total_seq += line.strip().upper()
        return total_seq



if __name__ == "__main__":
    
    method = 'hybrid'
    
    genstiff = GenStiffness(method=method)   # alternatively you can use the 'crystal' method for the Olson data
    
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    seqfn = sys.argv[1]
    if len(sys.argv) > 2:
        stepsize = int(sys.argv[2])
    else:
        stepsize = 1
    
    use_correction = False
    
    triadfn = 'methods/State/Nucleosome.state'
    nuctriads = read_nucleosome_triads(triadfn)
    
    base_len = 147        
    total_seq = readseq(seqfn)
    
    
    energy_data = []
    for i in range(0,len(total_seq)-base_len+1,stepsize):
        print(f'start id: {i}')
        seq = total_seq[i:i+base_len]
    
        stiffmat,groundstate = genstiff.gen_params(seq,use_group=True)
        midstep_constraint_locations = [
            2, 6, 14, 17, 24, 29, 
            34, 38, 45, 49, 55, 59, 
            65, 69, 76, 80, 86, 90, 
            96, 100, 107, 111, 116, 121, 
            128, 131, 139, 143
        ]

        Fdict = nucleosome_free_energy(groundstate,stiffmat,midstep_constraint_locations,nuctriads,use_correction=use_correction)
        print(Fdict)
        energy_data.append(Fdict)
        
    data = np.zeros((len(energy_data),5))
    for i in range(len(data)):
        Ddict = energy_data[i]
        data[i] = [Ddict['F'],Ddict['F_entropy'],Ddict['F_enthalpy'],Ddict['F_jacob'],Ddict['F_freedna']]

    print(data)
    outfn = seqfn+'_fe_th'
    if not use_correction:
        outfn += '_nc'
    
    np.save(outfn,data)