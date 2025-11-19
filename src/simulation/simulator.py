"""
Simulator Module
================

Functions for creating and configuring GillespieSimulator instances.

Author: MY
Date: 2025-11-16
"""

import numpy as np
from typing import Optional
from src.core.gillespie_simulator import GillespieSimulator
from src.core.nucleosomes import Nucleosome
from src.config.var import seed_for


def calculate_stride(total_steps: int, max_points: Optional[int]) -> int:
    """
    Calculate the stride for trajectory sampling.
    
    Args:
        total_steps: Total number of simulation steps
        max_points: Maximum number of points to save (None means save all)
    
    Returns:
        Stride value (1 means save every point)
    
    Example:
        >>> calculate_stride(1000, 100)
        10
        >>> calculate_stride(1000, None)
        1
    """
    if max_points is not None and max_points > 0:
        return max(1, int(np.ceil(total_steps / max_points)))
    return 1


def create_simulator(
    nuc: Nucleosome,
    build_params: dict,
    tau_points: np.ndarray,
    inf_protamine: bool,
    tau_min: Optional[float],
    replicate_num: int
) -> GillespieSimulator:
    """
    Create a fresh GillespieSimulator instance for a replicate.
    
    Each replicate gets its own simulator with fresh nucleosome and protamine
    instances to avoid state contamination between replicates.
    
    Args:
        nuc: Nucleosome instance
        build_params: Dictionary with factory functions:
            - 'nucs_factory': Function to create Nucleosomes instance
            - 'prot_factory': Function to create protamines instance
        tau_points: Array of dimensionless time points
        inf_protamine: Whether to use infinite protamine
        tau_min: Minimum tau value for renucleation (None to disable)
        replicate_num: Replicate number for seeding
    
    Returns:
        Configured GillespieSimulator instance
    
    Example:
        >>> from functools import partial
        >>> build_params = {
        ...     'nucs_factory': partial(create_nucleosomes_instance, k_wrap=1.0),
        ...     'prot_factory': partial(create_protamines_instance, prot_params={...})
        ... }
        >>> sim = create_simulator(nuc, build_params, tau_points, True, None, 0)
    """
    # Create fresh instances for this replicate
    nucs = build_params['nucs_factory'](nuc)
    prots = build_params['prot_factory']()
    
    # Generate reproducible seed
    seed = seed_for(nuc, replicate_num)
    
    return GillespieSimulator(
        nuc_inst=nucs,
        prot_inst=prots,
        t_points=None,
        max_steps=None,
        tau_points=tau_points,
        inf_protamine=inf_protamine,
        seed=seed,
        tau_min=tau_min
    )
