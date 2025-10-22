# src/core/nucleosomes.py
import numpy as np
import math
from typing import List, Iterable, Optional

class Nucleosome:
    __slots__ = ['id', 'subid', 'sequence', 'G_mat', 'state', 'n_closed', 'k_wrap', 'kT', 'binding_sites', 'detached', 'detach_time', 't_total', 't_block']

    def __init__(self, 
                 nuc_id: str, 
                 subid: int, 
                 sequence: str = None, 
                 G_mat: np.ndarray = None,
                 k_wrap: float = 1.0,
                 kT: float = 1.0,
                 binding_sites: int = 14):
        self.id = nuc_id
        self.subid = subid
        self.sequence = sequence
        self.state = np.zeros(binding_sites, dtype=np.uint8)  # 0: wrapped, 1: open unbound, 2: open bound
        self.n_closed = binding_sites
        self.k_wrap = k_wrap
        self.kT = kT
        self.binding_sites = binding_sites
        self.detached = 0  ## 1 means detached, 0 means attached
        self.detach_time = np.nan ## time when it detached; NaN means never detached
        self.t_total=0.0
        self.t_block=0.0    
        if G_mat is not None:
            self.G_mat = G_mat
        elif sequence is not None and sequence != "":
            self.G_mat = self.compute_G_from_sequence()
        else:
            raise ValueError("Either G_mat or sequence must be provided.")


    def compute_G_from_sequence(self) -> np.ndarray:
        G = np.zeros((self.binding_sites, self.binding_sites), dtype=float)
        ###Placeholder logic for G matrix computation based on sequence
        
        for i in range(self.binding_sites):
            for j in range(i + 1, self.binding_sites):
                G[i, j] = (j - i)  
        return G

    def unwrapping(self) -> dict:
        hist_occ_site = np.where(self.state == 0)[0]

        alpha = 0.0

        unwrapped_sites = {}
        if len(hist_occ_site) == 0:
            return unwrapped_sites

        left = hist_occ_site[0]
        right = hist_occ_site[-1]
        curr_left_end = left
        curr_right_end = right
        # print(f"Unwrapping sites: left={left}, right={right}, current range=({curr_left_end}, {curr_right_end})")
        G_current = self.G_mat[curr_left_end, curr_right_end] if curr_left_end <= curr_right_end else 0.0

        # Unwrap left
        new_left = curr_left_end + 1
        G_new_l = self.G_mat[new_left, curr_right_end] if new_left <= curr_right_end else 0.0
        dG_l = G_new_l - G_current
        rate_l = self.k_wrap * math.exp(-dG_l / self.kT) * math.exp(-alpha)  # Reduced rate for unwrapping left
        unwrapped_sites[left] = rate_l

        # Unwrap right if applicable
        if left != right:
            new_right = curr_right_end - 1
            G_new_r = self.G_mat[curr_left_end, new_right] if curr_left_end <= new_right else 0.0
            dG_r = G_new_r - G_current
            rate_r = self.k_wrap * math.exp(-dG_r / self.kT)*math.exp(-alpha) 
            unwrapped_sites[right] = rate_r
        # print(f"Unwrapped sites: {unwrapped_sites}")
        return unwrapped_sites

    def rewrapping(self) -> dict:

        hist_occ_site = np.where(self.state == 0)[0]
        rewrapped_sites = {}
        if self.detached == 1:
            return rewrapped_sites


        if len(hist_occ_site) > 0:
            if hist_occ_site[0] > 0 and self.state[hist_occ_site[0] - 1] == 1:
                rewrapped_sites[hist_occ_site[0] - 1] = self.k_wrap
            if hist_occ_site[-1] < self.binding_sites - 1 and self.state[hist_occ_site[-1] + 1] == 1:
                rewrapped_sites[hist_occ_site[-1] + 1] = self.k_wrap
            return rewrapped_sites


        if len(hist_occ_site) == 0:
            # ### Can modify to handle the case where nucleosome can rewrap when it has completely unwrapped also
            # ### Allowing the nucleation
            # open_sites = np.where(self.state == 1)[0]
            # for site in open_sites:
            #     rewrapped_sites[site] = self.k_wrap  # Use constant rate; could adjust if needed (e.g., lower for nucleation)

            k_nuc = self.k_wrap  # or a modest penalty like 0.3*self.k_wrap if you want nucleation slower than edge closure
            if self.state[0] == 1:
                rewrapped_sites[0] = k_nuc
            if self.state[self.binding_sites - 1] == 1:
                rewrapped_sites[self.binding_sites - 1] = k_nuc

            return rewrapped_sites

class Nucleosomes:
    def __init__(self, 
                 k_wrap: float, 
                 kT: float = 1.0, 
                 nucleosomes: Optional[Iterable[Nucleosome]] = None,
                 num_nucleosomes: int = 1,
                 binding_sites: int = 14, 
                 sequences: Optional[List[str]] = None, 
                 ids: Optional[List[str]] = None):
        self.k_wrap = k_wrap
        self.kT = kT
        self.binding_sites = binding_sites

        if nucleosomes is not None:
            # Materialize if iterable/generator for random access
            self.nucs = list(nucleosomes)
            self.num_nucleosomes = len(self.nucs)
            # Optional: Validate consistency (e.g., all have same binding_sites)
            # if self.nucs:
            #     self.binding_sites = self.nucs[0].binding_sites
        else:
            # Fallback to building from sequences/ids (legacy mode)
            if sequences is None:
                sequences = ["A" * 147] * num_nucleosomes
            if ids is None:
                ids = [f"nuc_{i}" for i in range(num_nucleosomes)]
            assert len(sequences) == num_nucleosomes == len(ids)

            self.num_nucleosomes = num_nucleosomes
            self.nucs = [
                Nucleosome(nuc_id=ids[i], subid=i, sequence=sequences[i], k_wrap=k_wrap, kT=kT, binding_sites=binding_sites)
                for i in range(num_nucleosomes)
            ]

    def __getitem__(self, idx: int) -> Nucleosome:
        return self.nucs[idx]

    def __len__(self) -> int:
        return self.num_nucleosomes
    
# if __name__ == "__main__":

