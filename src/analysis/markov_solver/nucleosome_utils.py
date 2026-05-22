"""
Nucleosome loading utilities.
"""
from typing import List, Optional, Tuple, Union
from pathlib import Path
import itertools

import numpy as np


def load_nucleosomes_from_file(
    file_path: Union[str, Path],
    k_wrap: float = 1.0,
    kT: float = 1.0,
    binding_sites: int = 14,
    max_nucs: Optional[int] = None,
    subids_range: Optional[Tuple[int, int]] = None,
) -> List:
    """
    Load nucleosomes from a file using memory-efficient generator pattern.
    
    Args:
        file_path: Path to nucleosome data file (TSV format)
        k_wrap: Wrapping rate constant (default: 21.0 s^-1)
        kT: Thermal energy (default: 1.0, dimensionless units)
        binding_sites: Number of binding sites (default: 14)
        max_nucs: Maximum number of nucleosomes to load (default: None = all)
        subids_range: Optional tuple (start, stop) for filtering subids
    Returns:
        List of Nucleosome objects loaded from file
    """
    from src.core.build_nucleosomes import nucleosome_generator
    
    file_path = Path(file_path)

    if subids_range is not None:
        gen = nucleosome_generator(
            file_path=str(file_path),
            k_wrap=k_wrap,
            kT=kT,
            binding_sites=binding_sites,
            subids=np.arange(*subids_range).tolist()
        )
    else:
        gen = nucleosome_generator(
            file_path=str(file_path),
            k_wrap=k_wrap,
            kT=kT,
            binding_sites=binding_sites
        )

    # Apply max_nucs limit if specified
    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)
    
    # Convert generator to list
    nucleosomes = list(gen)

    return nucleosomes


def load_nucleosomes_from_sprm(
    dataset_dir: Union[str, Path],
    k_wrap: float = 1.0,
    kT: float = 1.0,
    binding_sites: int = 14,
    max_nucs: Optional[int] = None,
    fasta_path: Optional[Union[str, Path]] = None,
    fasta_id_style: str = 'name',
) -> List:
    """
    Load nucleosomes from an SPRM dataset directory.

    Args:
        dataset_dir: Path to SPRM dataset directory (energies.tsv + id_lookup.tsv)
        k_wrap: Wrapping rate constant (default: 1.0 s^-1)
        kT: Thermal energy (default: 1.0, dimensionless units)
        binding_sites: Number of binding sites (default: 14)
        max_nucs: Maximum number of nucleosomes to load (default: None = all)
        fasta_path: Optional FASTA file with the 147-bp sequences keyed by
            seq_id (matches id_lookup.tsv).  When provided, sequences are
            attached to ``Nucleosome.sequence``.  When None, sequences remain
            ``None``.
        fasta_id_style: Header parsing style ('name' for named-peak datasets
            like ret_single_nuc, 'coord' for ``chr:start-end``-keyed controls).

    Returns:
        List of Nucleosome objects loaded from SPRM dataset
    """
    from src.core.build_nucleosomes import nucleosome_generator_sprm

    dataset_dir = Path(dataset_dir)

    gen = nucleosome_generator_sprm(
        dataset_dir=dataset_dir,
        k_wrap=k_wrap,
        kT=kT,
        binding_sites=binding_sites,
        fasta_path=str(fasta_path) if fasta_path is not None else None,
        fasta_id_style=fasta_id_style,
    )

    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)

    return list(gen)
