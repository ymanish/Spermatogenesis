"""
Nucleosome loading utilities.
"""
from typing import List, Optional, Tuple, Union, Dict
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
