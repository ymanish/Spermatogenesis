"""
Nucleosome Sampling Strategies
===============================

Memory-efficient methods for sampling nucleosomes from large datasets.

Uses reservoir sampling and lazy evaluation to avoid loading full datasets
into memory.

Author: MY
Date: 2025-11-14
"""

import numpy as np
import itertools
from typing import List, Iterator, Optional, TypeVar

from src.core.nucleosomes import Nucleosome


T = TypeVar('T')


def reservoir_sample(iterator: Iterator[T], k: int, seed: Optional[int] = None) -> List[T]:
    """
    Reservoir sampling: select k items uniformly at random from iterator
    without loading all items into memory.
    
    Algorithm: Vitter's Algorithm R (1985)
    
    Time complexity: O(n) where n = total items
    Space complexity: O(k) - only stores k items, not n
    
    Args:
        iterator: Iterator to sample from
        k: Number of items to sample
        seed: Random seed for reproducibility (optional)
    
    Returns:
        List of k randomly sampled items
    
    Examples:
        >>> gen = (i for i in range(1000))
        >>> sample = reservoir_sample(gen, k=10, seed=42)
        >>> len(sample)
        10
        >>> # Each item from original has equal probability of selection
    
    Notes:
        - Does NOT load full dataset into memory
        - Uniform distribution: each item has probability k/n
        - Works even if total size n is unknown
        - Single pass through data
    
    References:
        Vitter, J. S. (1985). "Random sampling with a reservoir".
        ACM Transactions on Mathematical Software, 11(1), 37-57.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Fill reservoir with first k items
    reservoir = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            # Randomly replace elements with decreasing probability
            j = np.random.randint(0, i + 1)
            if j < k:
                reservoir[j] = item
    
    return reservoir


def batcher(iterator: Iterator[T], size: int) -> Iterator[List[T]]:
    """
    Batch an iterator into chunks of given size.
    
    Args:
        iterator: Iterator to batch
        size: Size of each batch
    
    Yields:
        Lists of items, each of length `size` (last batch may be shorter)
    
    Examples:
        >>> items = range(25)
        >>> batches = list(batcher(items, size=10))
        >>> len(batches)
        3
        >>> [len(b) for b in batches]
        [10, 10, 5]
    
    Notes:
        - Lazy evaluation - doesn't materialize full iterator
        - Memory efficient for large datasets
        - Useful for parallel processing
    """
    it = iter(iterator)
    for first in it:
        batch = list(itertools.chain([first], itertools.islice(it, size - 1)))
        yield batch


def sample_nucleosomes(
    generator: Iterator[Nucleosome],
    n_nucleosomes: int,
    random_sample: bool = True,
    start_idx: int = 0,
    seed: Optional[int] = None,
    verbose: bool = True
) -> List[Nucleosome]:
    """
    Sample nucleosomes from a generator using memory-efficient methods.
    
    Two strategies:
    1. Random sampling: Reservoir sampling (uniform distribution, no full load)
    2. Sequential sampling: Skip to start_idx, take n_nucleosomes (deterministic)
    
    Args:
        generator: Nucleosome generator (iterator)
        n_nucleosomes: Number of nucleosomes to sample
        random_sample: If True, use reservoir sampling; if False, use sequential slicing
        start_idx: Starting index for sequential sampling (ignored for random sampling)
        seed: Random seed for reproducibility (only used for random sampling)
        verbose: Print progress information
    
    Returns:
        List of sampled Nucleosome objects
    
    Raises:
        ValueError: If no nucleosomes could be loaded
    
    Examples:
        >>> from src.core.build_nucleosomes import nucleosome_generator
        >>> 
        >>> # Random sampling
        >>> gen = nucleosome_generator(file_path, k_wrap=0.1)
        >>> nucs = sample_nucleosomes(gen, n_nucleosomes=20, random_sample=True, seed=42)
        >>> len(nucs)
        20
        >>> 
        >>> # Sequential sampling
        >>> gen = nucleosome_generator(file_path, k_wrap=0.1)
        >>> nucs = sample_nucleosomes(gen, n_nucleosomes=20, random_sample=False, start_idx=100)
        >>> # Returns nucleosomes [100, 101, ..., 119]
    
    Performance:
        - Random mode: O(n) time, O(k) memory where n=total, k=sample size
        - Sequential mode: O(start_idx + k) time, O(k) memory
        - Never loads full dataset into memory
    
    Memory Usage:
        - Each nucleosome: ~2 KB
        - 20 nucleosomes: ~0.04 MB
        - 100 nucleosomes: ~0.2 MB
        - 1000 nucleosomes: ~2 MB
    """
    if random_sample:
        # Use reservoir sampling - never loads full dataset!
        if verbose:
            print(f"Using reservoir sampling to select {n_nucleosomes} nucleosomes...")
        
        nucleosomes = reservoir_sample(
            generator, 
            k=n_nucleosomes,
            seed=seed
        )
        
        if verbose:
            print(f"✓ Randomly sampled {len(nucleosomes)} nucleosomes (seed={seed})")
    else:
        # Use batching for sequential selection
        # Skip to start_idx, then take n_nucleosomes
        if verbose and start_idx > 0:
            print(f"Skipping first {start_idx} nucleosomes...")
        
        # Skip initial nucleosomes
        gen_skipped = itertools.islice(generator, start_idx, None)
        
        # Take next n_nucleosomes using lazy evaluation
        nucleosomes = list(itertools.islice(gen_skipped, n_nucleosomes))
        
        if verbose:
            print(f"✓ Loaded {len(nucleosomes)} nucleosomes sequentially (start_idx={start_idx})")
    
    # Validation
    if len(nucleosomes) == 0:
        raise ValueError(
            f"No nucleosomes loaded! Check file path and indices. "
            f"(n_nucleosomes={n_nucleosomes}, start_idx={start_idx})"
        )
    
    # Memory reporting
    if verbose:
        mem_per_nuc_kb = 2  # Approximate memory per nucleosome
        total_mem_mb = len(nucleosomes) * mem_per_nuc_kb / 1024
        print(f"Memory usage: {len(nucleosomes)} nucleosomes × ~{mem_per_nuc_kb} KB ≈ {total_mem_mb:.2f} MB")
    
    return nucleosomes
