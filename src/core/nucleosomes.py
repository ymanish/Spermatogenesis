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
            raise NotImplementedError("G matrix computation from sequence is not implemented yet.")
        else:
            raise ValueError("Either G_mat or sequence must be provided.")


    def compute_G_from_sequence(self) -> np.ndarray:
        """
        Compute the G matrix based on the nucleosome sequence.
        Constructs the interaction/overlap matrix G for the instance's nucleosome binding
        sites. Each element G[i, j] should encode the interaction, compatibility, or penalty
        between binding site i and binding site j as derived from the underlying DNA /
        nucleosome sequence data held on the object.
        Warning
        -------
        This method currently contains a placeholder implementation. It does NOT perform a
        sequence-dependent calculation. Instead it returns a float numpy.ndarray of shape
        (self.binding_sites, self.binding_sites) with zeros on the diagonal and the integer
        distance (j - i) in the upper triangle (for j > i). This behavior is temporary and
        is not biologically meaningful. Replace with a proper sequence-based algorithm
        before using results for analysis or production.
        Returns
        -------
        numpy.ndarray
            Square matrix G with shape (binding_sites, binding_sites) and dtype float,
            representing pairwise site interactions as computed from sequence (placeholder
            semantics until implemented).
        Raises
        ------
        ValueError
            If self.binding_sites is not a positive integer.
        Notes
        -----
        - Implementers should document the expected normalization, symmetry (if any),
          and units/meaning of G entries once the real algorithm is added.
        - Consider raising NotImplementedError or providing a configuration flag if the
          placeholder should be avoided at runtime.
        """

        """Compute the G matrix based on the nucleosome sequence."""
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

    def compute_tau_slow_per_n(self) -> dict:
        """
        Compute local slow timescales tau_slow(n) for this nucleosome
        using its G_mat, k_wrap, kT, and binding_sites.

        tau_slow(n) ~ 1 / (total rate of wrapping/unwrapping transitions
        that change the total number of open contacts n).

        This method:
        1. Maps each valid matrix index (i,j) to total open contacts n
        2. Computes the slow exit rate from each (i,j) state
        3. Averages rates over all states with same n, weighted by Boltzmann factors
        4. Returns tau_slow[n] = 1 / average_rate for each n

        Returns
        -------
        tau_slow : dict
            tau_slow[n] = float (in units of 1/k_wrap, typically seconds)
            for n = 0 .. binding_sites-1.
            
        Notes
        -----
        - G_mat[i, j] represents the free energy for a state with:
          * i left contacts open
          * (L-1-j) right contacts open
          * Total n = i + (L-1-j) open contacts
        - The fully open state (n=L) is not in G_mat (assumed energy 0)
        - Unwrapping moves: (i,j) -> (i+1,j) [left] or (i,j-1) [right]
        - Wrapping moves: (i,j) -> (i-1,j) [left] or (i,j+1) [right]
        """
        G = self.G_mat
        L = self.binding_sites
        k_wrap = self.k_wrap
        kT = self.kT

        # Collect valid (i,j) indices: upper triangle i <= j
        states = []
        for i in range(L):
            for j in range(L):
                if i <= j:
                    states.append((i, j))

        # Map each (i,j) to total open contacts n
        # n = left_open + right_open = i + (L-1-j)
        n_values = {}
        for (i, j) in states:
            left_open = i
            right_open = (L - 1) - j
            n = left_open + right_open
            n_values[(i, j)] = n

        max_n = L - 1  # fully open (n=L) is not in G_mat

        tau_slow = {}

        for n in range(0, max_n + 1):
            # All (i,j) states with this total open n
            ij_at_n = [(i, j) for (i, j) in states if n_values[(i, j)] == n]
            if not ij_at_n:
                tau_slow[n] = math.inf
                continue

            # Boltzmann weights at fixed n
            energies = np.array([G[i, j] for (i, j) in ij_at_n])
            weights = np.exp(-energies / kT)
            Z_n = weights.sum()
            probs = weights / Z_n

            a_slow_avg = 0.0

            for p, (i, j) in zip(probs, ij_at_n):
                G_current = G[i, j]

                # Total slow rate out of (i,j)
                a_lr = 0.0

                # ---- Unwrap left: (i, j) -> (i+1, j) ----
                new_i = i + 1
                new_j = j
                if new_i < L:
                    if new_i <= new_j:
                        G_new_l = G[new_i, new_j]
                    else:
                        # Stepped outside triangle -> fully open/detached state
                        G_new_l = 0.0
                    dG_l = G_new_l - G_current
                    k_unwrap_left = k_wrap * math.exp(-dG_l / kT)
                    a_lr += k_unwrap_left

                # ---- Unwrap right: (i, j) -> (i, j-1) ----
                new_i = i
                new_j = j - 1
                if new_j >= 0:
                    if new_i <= new_j:
                        G_new_r = G[new_i, new_j]
                    else:
                        # new_i > new_j: fully open/detached
                        G_new_r = 0.0
                    dG_r = G_new_r - G_current
                    k_unwrap_right = k_wrap * math.exp(-dG_r / kT)
                    a_lr += k_unwrap_right

                # ---- Wrap left: (i, j) -> (i-1, j) ----
                if i > 0:
                    # Wrapping uses base rate k_wrap
                    a_lr += k_wrap

                # ---- Wrap right: (i, j) -> (i, j+1) ----
                right_open = (L - 1) - j
                if right_open > 0:
                    # Wrapping reduces right_open by 1 -> j -> j+1
                    if j + 1 < L:
                        a_lr += k_wrap

                # Accumulate weighted rate
                a_slow_avg += p * a_lr

            if a_slow_avg > 0.0:
                tau_slow[n] = 1.0 / a_slow_avg
            else:
                tau_slow[n] = math.inf

        return tau_slow

    def compute_tau_slow_per_ij(self) -> dict:
        """
        Compute local slow timescales tau_slow(i,j) for this nucleosome
        using its G_mat, k_wrap, kT, and binding_sites.

        tau_slow(i,j) ~ 1 / (total rate of wrapping/unwrapping transitions
        out of state (i,j)).

        Returns
        -------
        tau_slow_ij : dict
            tau_slow_ij[(i,j)] = float (seconds, if k_wrap is in 1/s)
            for all valid (i,j) with i <= j.

        Notes
        -----
        - G_mat[i, j] represents the free energy for a state with:
        * left_open  = i
        * right_open = (L-1-j)
        * total open n = i + (L-1-j)
        - Fully open (n = L) is treated as energy 0 outside G_mat.
        - Unwrapping moves:
            (i,j) -> (i+1,j)  [left]
            (i,j) -> (i,j-1)  [right]
        with k_unwrap = k_wrap * exp(-ΔG / kT).
        - Wrapping moves:
            (i,j) -> (i-1,j)  [left]
            (i,j) -> (i,j+1)  [right]
        with rate k_wrap (constant base rate).
        """
        G = self.G_mat
        L = self.binding_sites
        k_wrap = self.k_wrap
        kT = self.kT

        tau_slow_ij = {}

        # Loop over all valid (i,j) in the upper triangle
        for i in range(L):
            for j in range(i, L):  # ensure i <= j
                G_current = G[i, j]

                a_slow = 0.0  # total slow rate out of (i,j)

                # ---- Unwrap left: (i, j) -> (i+1, j) ----
                new_i = i + 1
                new_j = j
                if new_i < L:
                    if new_i <= new_j:
                        G_new_l = G[new_i, new_j]
                    else:
                        # stepped outside triangle: treat as fully detached (energy 0)
                        G_new_l = 0.0
                    dG_l = G_new_l - G_current
                    k_unwrap_left = k_wrap * math.exp(-dG_l / kT)
                    a_slow += k_unwrap_left

                # ---- Unwrap right: (i, j) -> (i, j-1) ----
                new_i = i
                new_j = j - 1
                if new_j >= 0:
                    if new_i <= new_j:
                        G_new_r = G[new_i, new_j]
                    else:
                        # outside triangle: fully detached (energy 0)
                        G_new_r = 0.0
                    dG_r = G_new_r - G_current
                    k_unwrap_right = k_wrap * math.exp(-dG_r / kT)
                    a_slow += k_unwrap_right

                # ---- Wrap left: (i, j) -> (i-1, j) ----
                if i > 0:
                    # Wrapping uses base rate k_wrap
                    a_slow += k_wrap

                # ---- Wrap right: (i, j) -> (i, j+1) ----
                right_open = (L - 1) - j
                if right_open > 0:
                    # j+1 reduces right_open by 1
                    if j + 1 < L:
                        a_slow += k_wrap

                # Convert to timescale
                if a_slow > 0.0:
                    tau_slow_ij[(i, j)] = 1.0 / a_slow
                else:
                    tau_slow_ij[(i, j)] = math.inf

        return tau_slow_ij


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

