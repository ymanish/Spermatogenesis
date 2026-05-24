"""
Simulator Module
================

Functions for creating and configuring GillespieSimulator instances.
"""

import copy
import numpy as np
from typing import Optional
from src.core.gillespie_simulator import GillespieSimulator
from src.core.nucleosomes import Nucleosome
from src.core.protamine import protamines
from src.config.var import seed_for


def calculate_stride(total_steps: int, max_points: Optional[int]) -> int:
    """
    Calculate the stride for trajectory sampling.

    Args:
        total_steps: Total number of simulation steps
        max_points: Maximum number of points to save (None means save all)

    Returns:
        Stride value (1 means save every point)
    """
    if max_points is not None and max_points > 0:
        return max(1, int(np.ceil(total_steps / max_points)))
    return 1


def create_simulator(
    nuc: Nucleosome,
    prot_params: dict,
    tau_points: np.ndarray,
    inf_protamine: bool,
    tau_min: Optional[float],
    replicate_num: int
) -> GillespieSimulator:
    """
    Create a fresh GillespieSimulator for a single replicate.

    Each replicate gets an independent nucleosome copy and protamine instance
    so there is no state contamination between replicates.

    Args:
        nuc:           Nucleosome instance to simulate
        prot_params:   Dict with keys k_unbind, k_bind, p_conc, cooperativity
        tau_points:    Array of dimensionless time points
        inf_protamine: Whether to use infinite protamine supply
        tau_min:       Minimum dwell time for renucleation (None = disabled)
        replicate_num: Replicate index used for deterministic seeding
    """
    nuc_copy = copy.deepcopy(nuc)
    prot = protamines(**prot_params)
    seed = seed_for(nuc, replicate_num)

    return GillespieSimulator(
        nuc_inst=nuc_copy,
        prot_inst=prot,
        t_points=None,
        max_steps=None,
        tau_points=tau_points,
        inf_protamine=inf_protamine,
        seed=seed,
        tau_min=tau_min
    )
