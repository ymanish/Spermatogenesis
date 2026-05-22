# src/core/build_nucleosomes.py
import pandas as pd
from src.core.nucleosomes import Nucleosome, Nucleosomes
import numpy as np
import itertools


def _parse_fasta(fa_path: str, id_style: str = "name") -> dict[str, str]:
    """Parse a FASTA file into ``{key: sequence}``."""
    if id_style not in ("coord", "name"):
        raise ValueError(f"id_style must be 'coord' or 'name', got {id_style!r}")

    records = {}
    current_key = None
    seq_parts = []

    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_key is not None:
                    records[current_key] = "".join(seq_parts).upper()
                header = line[1:]
                parts = header.split("|")
                if id_style == "coord":
                    current_key = parts[0].strip()
                elif len(parts) >= 2:
                    current_key = parts[1].strip()
                else:
                    current_key = parts[0].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_key is not None:
        records[current_key] = "".join(seq_parts).upper()

    return records


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


def nucleosome_generator_sprm(dataset_dir: str, k_wrap: float, kT: float = 1.0,
                               binding_sites: int = 14, global_ids: list = None,
                               fasta_path: str = None, fasta_id_style: str = 'name'):
    """
    Generator for the SPRM dataset format.

    Reads:
      - <dataset_dir>/energies.tsv   (tab-separated, header: global_id left_open right_open dF_total)
      - <dataset_dir>/id_lookup.tsv  (tab-separated, header: global_id seq_id)

    Args:
        dataset_dir: Directory containing energies.tsv and id_lookup.tsv
        k_wrap:      Wrapping rate constant
        kT:          Thermal energy (default: 1.0)
        binding_sites: Number of binding sites (default: 14)
        global_ids:  Optional list of global_ids to filter by
        fasta_path:  Optional FASTA file containing the 147-bp sequences keyed
                     by seq_id (the value in id_lookup.tsv).  When provided,
                     the matching sequence is attached to Nucleosome.sequence.
        fasta_id_style: Header parsing style for the FASTA, passed through to
                     ``_parse_fasta`` ('coord' for ``chr:start-end``-keyed
                     controls, 'name' for named-peak datasets).

    Yields:
        Nucleosome instances with nuc_id=seq_id (from lookup) and subid=global_id
    """
    import os

    energies_path = os.path.join(dataset_dir, 'energies.tsv')
    id_lookup_path = os.path.join(dataset_dir, 'id_lookup.tsv')

    seq_map = {}
    if fasta_path is not None:
        seq_map = _parse_fasta(fasta_path, id_style=fasta_id_style)

    # Build global_id -> seq_id mapping
    id_map = {}
    if os.path.exists(id_lookup_path):
        with open(id_lookup_path, 'r') as f:
            f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        id_map[int(parts[0])] = parts[1]
                    except ValueError:
                        continue

    global_ids_set = set(global_ids) if global_ids is not None else None

    current_gid = None
    current_G = None

    with open(energies_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            try:
                gid = int(parts[0])
                left_open = int(parts[1])
                right_open = int(parts[2])
                dF_total = float(parts[3])
            except ValueError:
                continue

            if global_ids_set is not None and gid not in global_ids_set:
                continue

            if gid != current_gid:
                if current_G is not None:
                    seq_id = id_map.get(current_gid, str(current_gid))
                    yield Nucleosome(nuc_id=seq_id, subid=current_gid,
                                     sequence=seq_map.get(seq_id),
                                     G_mat=current_G, k_wrap=k_wrap, kT=kT,
                                     binding_sites=binding_sites)
                current_gid = gid
                current_G = np.zeros((binding_sites, binding_sites))

            i = left_open
            j = (binding_sites - 1) - right_open
            if (i <= j) and (left_open + right_open < binding_sites):
                current_G[i, j] = dF_total

    if current_G is not None:
        seq_id = id_map.get(current_gid, str(current_gid))
        yield Nucleosome(nuc_id=seq_id, subid=current_gid,
                         sequence=seq_map.get(seq_id),
                         G_mat=current_G, k_wrap=k_wrap, kT=kT,
                         binding_sites=binding_sites)


if __name__ == "__main__":
    # Example usage
    file_path = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/boundprom/breath_energy/001.tsv"
    nucleosomes_instance = build_nucleosomes_from_file(file_path, k_wrap=21.0, kT=1.0, binding_sites=14)

    # Print the nucleosome states
    for nuc in nucleosomes_instance:
        print(f"Nucleosome ID: {nuc.id}, SubID: {nuc.subid}, Gmat: {nuc.G_mat}")
        import sys
        sys.exit()
