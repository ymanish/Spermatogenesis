#/src/core/nucleosome.py
import numpy as np
import math


class nucleosome:

    def __init__(self, k_unwrap:float, k_wrap:float, num_nucleosomes:int, binding_sites:int=14):
        self.k_unwrap = k_unwrap
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        self.nuc_state = np.zeros(num_nucleosomes, dtype=np.int32) ## records the state of each nucleosome
        self.nuc_alive = np.ones(num_nucleosomes, dtype=np.int32) ### tells if nucleosome is alive
        self.N_closed = num_nucleosomes * binding_sites ## total closed sites


    def compute_G(self):
        S = 'ACGCGGATCAAATTT'
        return S

    def unwrapping(self, hist_occ_site:np.ndarray)-> dict:

        assert hist_occ_site.ndim == 1, "hist_occ_site must be a 1D numpy array"

        unwrapped_sites = dict()

        if len(hist_occ_site) == 0:
            return unwrapped_sites

        elif hist_occ_site[0] != hist_occ_site[-1]:
            unwrapped_sites[hist_occ_site[0]] = self.k_unwrap
            unwrapped_sites[hist_occ_site[-1]] = self.k_unwrap

        else:
            unwrapped_sites[hist_occ_site[0]] = self.k_unwrap

        return unwrapped_sites

    def rewrapping(self, nuce:np.ndarray, 
                        hist_occ_site:np.ndarray, 
                        keep_histone:bool=False)->dict:
        rewrapped_sites = dict()

        if keep_histone and len(hist_occ_site) == 0:
            available_site = np.ravel(np.where(nuce == 1))
            for i in available_site:
                rewrapped_sites[i] = self.k_wrap

        elif len(hist_occ_site) > 0:
          # Check the left of the first entry
            if hist_occ_site[0] > 0 and nuce[hist_occ_site[0] - 1] == 1:
                rewrapped_sites[hist_occ_site[0] - 1] = self.k_wrap

            # Check the right of the last entry
            if hist_occ_site[-1] < len(nuce) - 1 and nuce[hist_occ_site[-1] + 1] == 1:
                rewrapped_sites[hist_occ_site[-1] + 1] = self.k_wrap
                
        return rewrapped_sites
    

if __name__ == "__main__":

    nuc = nucleosome(k_unwrap=4.0, k_wrap=21.0, num_nucleosomes=1)
    histone_occupied_site = np.array([ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    nuce = np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    unwrapped_sites, sum_rate = nuc.unwrapping(histone_occupied_site)
    print("Unwrapped Sites:", unwrapped_sites)
    print("Sum Rate of Unwrapping:", sum_rate)

    rewrapped_sites, sum_rate = nuc.rewrapping(nuce=nuce, histone_occupied_site=histone_occupied_site)
    print("Rewrapped Sites:", rewrapped_sites)
    print("Sum Rate of Rewrapping:", sum_rate)

