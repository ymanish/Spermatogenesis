import sys, os, glob
import numpy as np
from scipy.sparse import lil_matrix
from typing import List, Tuple, Callable, Any, Dict
from .PolyCG.polycg.SO3 import so3

class GenStiffness:
    def __init__(self, method: str = "md", stiff_method: str = None, gs_method: str = None):
        self.method         = method
        self.stiff_method   = stiff_method
        self.gs_method      = gs_method
        self.load_from_file()

    def load_from_file(self):
        if self.method.lower() == "md":
            stiff_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
            gs_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
        elif "crystal" in self.method.lower():
            stiff_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
            gs_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
        elif "hybrid" in self.method.lower():
            stiff_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
            gs_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
        else:
            raise ValueError(f'Unknown method "{self.method}".')
        
        if self.stiff_method is not None:
            if self.stiff_method.lower() == 'md':
                stiff_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/MolecularDynamics"
                )
            elif self.stiff_method.lower() == 'crystal':
                stiff_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/Crystallography"
                )
            else:
                raise ValueError(f'Unknown stiff_method "{self.stiff_method}".')
            
        if self.gs_method is not None:
            if self.gs_method.lower() == 'md':
                gs_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/MolecularDynamics"
                )
            elif self.gs_method.lower() == 'crystal':
                gs_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/Crystallography"
                )
            else:
                raise ValueError(f'Unknown gs_method "{self.gs_method}".')
        
        bases = "ATCG"
        self.dimers = {}
        for b1 in bases:
            for b2 in bases:
                seq = b1 + b2
                self.dimers[seq] = self.read_dimer(seq, stiff_path, gs_path)

    def read_dimer(self, seq: str, stiff_path: str, gs_path: str):
        fnstiff = glob.glob(stiff_path + "/Stiffness*" + seq + "*")[0]
        fnequi = glob.glob(gs_path + "/Equilibrium*" + seq + "*")[0]

        equi = np.loadtxt(fnequi)
        stiff = np.loadtxt(fnstiff)
        equi_triad = so3.se3_midstep2triad(equi)
        stiff_group = so3.se3_algebra2group_stiffmat(
            equi, stiff, translation_as_midstep=True
        )
        dimer = {
            "seq": seq,
            "group_gs": equi_triad,
            "group_stiff": stiff_group,
            "equi": equi,
            "stiff": stiff,
        }
        return dimer

    def gen_params(self, seq: str, use_group: bool = False, sparse: bool = False):
        N = len(seq) - 1
        if sparse:
            stiff = lil_matrix((6 * N, 6 * N))
        else:
            stiff = np.zeros((6 * N, 6 * N))
        gs = np.zeros((N, 6))
        for i in range(N):
            bp = seq[i : i + 2].upper()
            if use_group:
                pstiff = self.dimers[bp]["group_stiff"]
                pgs = self.dimers[bp]["group_gs"]
            else:
                pstiff = self.dimers[bp]["stiff"]
                pgs = self.dimers[bp]["equi"]

            stiff[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = pstiff
            gs[i] = pgs
        
        if sparse:
            stiff = stiff.tocsc()
        return stiff,gs


class GenStiffness_:
    
    def __init__(self, method: str = 'md'):
        self.method = method
        self.load_from_file()
        
    def load_from_file(self):
        if self.method.lower() == 'md':
            path = os.path.join(os.path.dirname(__file__), 'Parametrization/MolecularDynamics')
        elif 'crystal' in self.method.lower():
            path = os.path.join(os.path.dirname(__file__), 'Parametrization/Crystallography')
        else:
            raise ValueError(f'Unknown method "{self.method}".')
        
        bases = 'ATCG'
        self.dimers = {}
        for b1 in bases:
            for b2 in bases:
                seq = b1+b2
                self.dimers[seq] = self.read_dimer(seq,path)
                
            
    def read_dimer(self, seq: str, path: str):
        fnstiff = glob.glob(path+'/Stiffness*'+seq+'*')[0]
        fnequi  = glob.glob(path+'/Equilibrium*'+seq+'*')[0]
        
        equi = np.loadtxt(fnequi)
        stiff = np.loadtxt(fnstiff)
        equi_triad = so3.se3_midstep2triad(equi)                
        stiff_group = so3.se3_algebra2group_stiffmat(equi,stiff,translation_as_midstep=True)  
                      
        dimer = {
            'seq' : seq,
            'group_gs':   equi_triad,
            'group_stiff':stiff_group,
            'equi': equi,
            'stiff' : stiff
            }
        return dimer
    
    def gen_params(self, seq: str, use_group: str=True):
        N = len(seq)-1
        stiff = np.zeros((6*N,6*N))
        gs    = np.zeros((N,6))
        for i in range(N):
            bp = seq[i:i+2].upper()
            if use_group:
                pstiff = self.dimers[bp]['group_stiff']
                pgs    = self.dimers[bp]['group_gs']
            else:
                pstiff = self.dimers[bp]['equi']
                pgs    = self.dimers[bp]['stiff']
            
            stiff[6*i:6*i+6,6*i:6*i+6] = pstiff
            gs[i] = pgs
        return stiff,gs 
    
def read_nucleosome_triads(fn):
    data = np.loadtxt(fn)
    N = len(data) // 12
    nuctriads = np.zeros((N,4,4))
    for i in range(N):
        tau = np.eye(4)
        pos   = data[i*12:i*12+3] / 10
        triad = data[i*12+3:i*12+12].reshape((3,3))
        triad = so3.euler2rotmat(so3.rotmat2euler(triad))
        tau[:3,:3] = triad
        tau[:3,3]  = pos
        nuctriads[i] = tau
    return nuctriads


    

        
        
        



if __name__ == '__main__':
    
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    # genstiff = GenStiffness(method='MD')
    
    # seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
    # stiff,gs = genstiff.gen_params(seq)

    # print(gs)
    # print(stiff)
    
    triadfn = os.path.join(os.path.dirname(__file__), 'State/Nucleosome.state')
    nuctriads = read_nucleosome_triads(triadfn)
    N = len(nuctriads)

    for i in range(N-1):
        
        g = np.linalg.inv(nuctriads[i]) @ nuctriads[i+1]
        X = so3.se3_rotmat2euler(g)
        X[:3] *= 180/np.pi
        # print(X)
    
    print(N)