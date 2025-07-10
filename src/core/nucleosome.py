#/src/core/nucleosome.py
import numpy as np
import math


class nucleosme:

    def __init__(self, k_unwrap, k_wrap, num_nucleosomes, binding_sites=14):
        self.k_unwrap = k_unwrap
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        self.nucleosomes = [np.zeros(self.binding_sites, dtype=int) for _ in range(num_nucleosomes)]
        self.record_state = np.ones(num_nucleosomes)
        self.N_closed = num_nucleosomes * binding_sites
        self.N_open = 0


    def sequence(self):
        S = 'ACGCGGATCAAATTT'
        return S

    def unwrapping(self, histone_occupied_site):
        unwrapped_sites = dict()
        sum_rate = 0

        if len(histone_occupied_site) == 0:
            return unwrapped_sites, sum_rate

        elif histone_occupied_site[0] != histone_occupied_site[-1]:
            unwrapped_sites[histone_occupied_site[0]] = self.k_unwrap
            unwrapped_sites[histone_occupied_site[-1]] = self.k_unwrap

        else:
            unwrapped_sites[histone_occupied_site[0]] = self.k_unwrap


        sum_rate = sum(unwrapped_sites.values())

        return unwrapped_sites, sum_rate

    def rewrapping(self, nuce, histone_occupied_site, keep_histone=False):
        rewrapped_sites = dict()
        sum_rate = 0

        if keep_histone and len(histone_occupied_site) == 0:
            available_site = np.ravel(np.where(nuce == 1))
            for i in available_site:
                rewrapped_sites[i] = self.k_wrap

        elif len(histone_occupied_site) > 0:
          # Check the left of the first entry
            if histone_occupied_site[0] > 0 and nuce[histone_occupied_site[0] - 1] == 1:
                rewrapped_sites[histone_occupied_site[0] - 1] = self.k_wrap

            # Check the right of the last entry
            if histone_occupied_site[-1] < len(nuce) - 1 and nuce[histone_occupied_site[-1] + 1] == 1:
                rewrapped_sites[histone_occupied_site[-1] + 1] = self.k_wrap

    
        if len(rewrapped_sites) > 0:
            sum_rate = sum(rewrapped_sites.values())
    
        return rewrapped_sites, sum_rate

