import sys, os
import numpy as np
import scipy as sp
from typing import List, Tuple, Callable, Any, Dict

from methods.PolyCG.polycg.SO3 import so3
from methods.PolyCG.polycg.transforms.transform_SO3 import euler2rotmat_so3
from methods.PolyCG.polycg.transforms.transform_marginals import send_to_back_permutation
from methods.midstep_composites import midstep_composition_transformation, midstep_se3_groundstate
from methods.read_nuc_data import read_nucleosome_triads, GenStiffness

from methods.PolyCG.polycg.cgnaplus import cgnaplus_bps_params
from methods.free_energy import nucleosome_free_energy


if __name__ == '__main__':
    
    seqsfns = sys.argv[1:]
    model = 'MD'
    model = 'crystal'
    
    for seqsfn in seqsfns:
        seqs = []
        with open(seqsfn, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip() != '']
            for line in lines:
                seqs.append(line)
        
        genstiff = GenStiffness(method=model)
        triadfn = os.path.join(os.path.dirname(__file__), 'methods/State/Nucleosome.state')
        nuctriads = read_nucleosome_triads(triadfn)

        midstep_constraint_locations = [
            2, 6, 14, 17, 24, 29, 
            34, 38, 45, 49, 55, 59, 
            65, 69, 76, 80, 86, 90, 
            96, 100, 107, 111, 116, 121, 
            128, 131, 139, 143
        ]
            
        fes = np.zeros((len(seqs),4))
        for i,seq in enumerate(seqs):
            print(i)

            stiff, gs = genstiff.gen_params(seq)
            # gs,stiff = cgnaplus_bps_params(seq,euler_definition=True,group_split=True)
            Fdict  = nucleosome_free_energy(
                gs,
                stiff,
                midstep_constraint_locations, 
                nuctriads
            )
            fes[i,0] = Fdict['F']
            fes[i,1] = Fdict['F_const']
            fes[i,2] = Fdict['F_entropy']
            fes[i,3] = Fdict['F_jacob']
            
            print(fes[i])
            
        outname = f'{os.path.splitext(seqsfn)[0]}_{model}_fe'
        np.save(outname,fes)
