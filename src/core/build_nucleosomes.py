# src/core/build_nucleosomes.py
import pandas as pd
from src.core.nucleosomes import Nucleosome, Nucleosomes
import numpy as np
import itertools

def nucleosome_generator(file_path: str, k_wrap: float, kT: float = 1.0, binding_sites: int = 14, ids: list = None, subids: list = None):
    """
    Reads the energy file and builds a Nucleosomes instance.
    Assumes the file has columns: dF, F_freedna, F_enthalpy, id, subid, right_open, left_open, dF_total
    Uses dF_total as the free energy for the state with given left_open and right_open.
    """
    current_key = None
    current_G = None
    current_id = None
    current_subid = None

    with open(file_path, 'r') as f:
        for line in f:           
            if line.strip() == '' or line.startswith('#'): 
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(f"Line does not have enough parts: {line.strip()}, "
                                 f"Check if the file format is correct.")
            try:
                # Parse relevant fields (ignore dF, F_freedna, F_enthalpy)
                nuc_id = parts[3]
                subid = int(parts[4])
                right_open = int(parts[5])
                left_open = int(parts[6])
                dF_total = float(parts[7])
            except ValueError:
                continue  # Skip parse errors
            

            if ids is not None and nuc_id not in ids:
                continue
            if subids is not None and subid not in subids:
                continue

            key = (nuc_id, subid)
            if key != current_key:
                if current_G is not None:
                    # print(f"Building nucleosome for {current_id} with subid {current_subid}")

                    yield Nucleosome(nuc_id=current_id, subid=current_subid, sequence=None, G_mat=current_G,
                                     k_wrap=k_wrap, kT=kT, binding_sites=binding_sites)
                    # print(nuc.G_mat.shape)

                    #     # Reset for new nucleosome
                    # disp = [
                    #     [ nuc.G_mat[i, j] for j in range(14) ]
                    #     for i in range(14)
                    # ]
                    # df = pd.DataFrame(
                    #     disp,
                    #     index=[f"left={i}"  for i in range(14)],
                    #     columns=[f"right={j}" for j in range(14)]
                    # )
                    # print(f"\nCurrent G matrix for {current_id} subid {current_subid}:")
                    # print(df.to_string())

                    # import sys
                    # sys.exit()
                   
                current_key = key
                current_id = nuc_id
                current_subid = subid
                current_G = np.zeros((binding_sites, binding_sites))

            ## Build Index Level Matrix for left and right ends of the nucleosome, ignoring the completely open state, 
            ## as it has 0 energy, which is trivial
            i = left_open
            j = (binding_sites-1) - right_open

            if (i <= j) and (left_open + right_open < binding_sites):
                current_G[i, j] = dF_total
        
    if current_G is not None:
        yield Nucleosome(nuc_id=current_id, subid=current_subid, sequence=None, G_mat=current_G,
                         k_wrap=k_wrap, kT=kT, binding_sites=binding_sites)


import warnings

def nucleosome_generator_with_effective_energies(file_path: str, 
                                                 k_wrap: float, 
                                                 k_bind: float, 
                                                 k_unbind: float,
                                                 p_conc: float, 
                                                 cooperativity: float,
                                                 kT: float = 1.0, 
                                                 binding_sites: int = 14, 
                                                 ids: list = None, 
                                                 subids: list = None):
    """
    Reads the energy file and builds a Nucleosomes instance with effective energies.
    
    .. deprecated:: 1.0.0
        This function is deprecated and will be removed in a future version.
        The effective energy calculation is now handled by the markov_solver module
        through the generator.py functions which compute protamine effects on-the-fly
        using the fast protamine limit approximation.
        
        Instead of pre-computing effective energies in G_mat, use:
        
        - nucleosome_generator() to load base nucleosomes
        - build_Q_matrix_from_nucleosome() from markov_solver.generator with 
          protamine_params to compute rates with protamine effects
        
    Old usage:
        >>> gen = nucleosome_generator_with_effective_energies(
        ...     file_path, k_wrap, k_bind, k_unbind, p_conc, cooperativity
        ... )
        
    New usage:
        >>> from src.analysis.markov_solver import build_Q_matrix_from_nucleosome
        >>> gen = nucleosome_generator(file_path, k_wrap, kT)
        >>> for nuc in gen:
        ...     protamine_params = {
        ...         'k_bind': k_bind,
        ...         'k_unbind': k_unbind,
        ...         'p_conc': p_conc,
        ...         'cooperativity': cooperativity
        ...     }
        ...     Q_full, transient, absorbing, idx_map = build_Q_matrix_from_nucleosome(
        ...         nuc, protamine_params=protamine_params
        ...     )
    
    Args:
        file_path: Path to TSV file with nucleosome energies
        k_wrap: Wrapping rate constant
        k_bind: Protamine binding rate
        k_unbind: Protamine unbinding rate
        p_conc: Protamine concentration
        cooperativity: Protamine cooperativity parameter (β*J)
        kT: Thermal energy (default: 1.0)
        binding_sites: Number of DNA binding sites (default: 14)
        ids: Filter by nucleosome IDs (optional)
        subids: Filter by nucleosome subids (optional)
        
    Yields:
        Nucleosome: Nucleosome instances with G_mat including protamine effects
        
    Warnings:
        DeprecationWarning: This function is deprecated
    """
    warnings.warn(
        "nucleosome_generator_with_effective_energies() is deprecated and will be removed in a future version. "
        "The effective energy calculation is now handled by the markov_solver module. "
        "Use nucleosome_generator() to load base nucleosomes, then build_Q_matrix_from_nucleosome() "
        "from markov_solver.generator with protamine_params to compute rates with protamine effects on-the-fly. "
        "See the function docstring for migration examples.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from src.core.ising_model import partition_function
    betamu = np.log(p_conc * k_bind / k_unbind)
    betaJ = cooperativity/kT

    # create vector for n = 0..binding_sites (0..14)
    n_range = np.arange(0, binding_sites + 1)

    # compute partition function for each n (partition_function(betamu, J, n))
    pf_vector = np.array([partition_function(int(n), betamu, betaJ) for n in n_range])
    print(f"Partition function vector Z(n) for n=0..{binding_sites}: {pf_vector}")
    # avoid log(0): set any zero partition function values to a very small number
    pf_vector_safe = np.where(pf_vector > 0, pf_vector, 1e-300)

    # free energy vector F(n) = -kT * ln Z(n)
    F_vector = -kT * np.log(pf_vector_safe)
    dF_vector = np.zeros_like(F_vector)
    dF_vector= F_vector - F_vector[-1]
    print(f"Effective free energy vector F(n) for n=0..{binding_sites}: {dF_vector}")

    current_key = None
    current_G = None
    current_id = None
    current_subid = None

    with open(file_path, 'r') as f:
        for line in f:           
            if line.strip() == '' or line.startswith('#'): 
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(f"Line does not have enough parts: {line.strip()}, "
                                 f"Check if the file format is correct.")
            try:
                # Parse relevant fields (ignore dF, F_freedna, F_enthalpy)
                nuc_id = parts[3]
                subid = int(parts[4])
                right_open = int(parts[5])
                left_open = int(parts[6])
                dF_total = float(parts[7])
            except ValueError:
                continue  # Skip parse errors
            

            if ids is not None and nuc_id not in ids:
                continue
            if subids is not None and subid not in subids:
                continue

            key = (nuc_id, subid)
            if key != current_key:
                if current_G is not None:
                    yield Nucleosome(nuc_id=current_id, subid=current_subid, sequence=None, G_mat=current_G,
                                     k_wrap=k_wrap, kT=kT, binding_sites=binding_sites)
                   
                current_key = key
                current_id = nuc_id
                current_subid = subid
                current_G = np.zeros((binding_sites, binding_sites))

            ## Build Index Level Matrix for left and right ends of the nucleosome, ignoring the completely open state, 
            ## as it has 0 energy, which is trivial
            i = left_open
            j = (binding_sites-1) - right_open

            if (i <= j) and (left_open + right_open < binding_sites):
                current_G[i, j] = dF_total + F_vector[left_open] + F_vector[right_open] - F_vector[-1]
                # current_G[i, j] = dF_total + F_vector[left_open + right_open] - F_vector[-1]


    if current_G is not None:
        yield Nucleosome(nuc_id=current_id, subid=current_subid, sequence=None, G_mat=current_G,
                         k_wrap=k_wrap, kT=kT, binding_sites=binding_sites)



def build_nucleosomes_from_file(
                                file_path: str,
                                k_wrap:     float,
                                kT:         float = 1.0,
                                binding_sites: int = 14,
                                max_nucs:     int = None, 
                                ids:        list = None,
                                subids: list = None
                            ) -> Nucleosomes:
    """
    Read up to max_nucs nucleosomes from file_path (all if max_nucs is None).
    """
    gen = nucleosome_generator(file_path, k_wrap, kT, binding_sites)
    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)

    nucleosomes_list = list(gen)
    return Nucleosomes(
        k_wrap       = k_wrap,
        kT           = kT,
        nucleosomes  = nucleosomes_list,
        binding_sites= binding_sites,
    )


if __name__ == "__main__":
    # Example usage
    file_path = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/boundprom/breath_energy/001.tsv" 
    nucleosomes_instance = build_nucleosomes_from_file(file_path, k_wrap=21.0, kT=1.0, binding_sites=14)
    
    # Print the nucleosome states
    for nuc in nucleosomes_instance:
        print(f"Nucleosome ID: {nuc.id}, SubID: {nuc.subid}, Gmat: {nuc.G_mat}")
        import sys
        sys.exit()