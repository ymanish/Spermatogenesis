"""
Replicate Module
================

Functions for running single replicate simulations.

Author: MY
Date: 2025-11-16
"""

import numpy as np
from typing import Optional, Tuple, List
from src.core.gillespie_simulator import GillespieSimulator
from src.core.nucleosomes import Nucleosome
from .simulator import create_simulator
from .trajectory import store_trajectory_data


def process_simulation_states(
    sim: GillespieSimulator,
    save_trajectories: bool,
    eff_stride: int,
    total_steps: int
) -> Tuple[List[float], List[int], List[int], List[int], int, int, float]:
    """
    Process simulation states and collect trajectory data.
    
    Args:
        sim: GillespieSimulator instance
        save_trajectories: Whether to save trajectory data
        eff_stride: Stride for sampling trajectory points
        total_steps: Total number of steps
    
    Returns:
        Tuple of:
            - tau_times: List of tau time points
            - cs_totals: List of chromatin state totals
            - bprots: List of bound protamine counts
            - detached_totals: List of detached nucleosome counts
            - final_cs: Final chromatin state
            - final_bprot: Final bound protamine count
            - detach_time: First detachment time (NaN if never detached)
    """
    tau_times = []
    cs_totals = []
    bprots = []
    detached_totals = []
    detach_time = np.nan
    final_cs = 0
    final_bprot = 0
    
    for step, state in enumerate(sim.run()):
        # Always update final state
        final_cs = state.cs_total
        final_bprot = state.bprot
        
        # Save trajectory points if needed
        if save_trajectories and (step % eff_stride == 0 or step == total_steps - 1):
            tau_times.append(state.tau)
            cs_totals.append(state.cs_total)
            bprots.append(state.bprot)
            detached_totals.append(state.detached_total)
        
        # Record first detachment time
        if state.detached_total > 0 and np.isnan(detach_time):
            detach_time = state.time
    
    return tau_times, cs_totals, bprots, detached_totals, final_cs, final_bprot, detach_time


def aggregate_replicate_results(
    all_rep_cs: List[float],
    all_rep_bprot: List[float],
    all_rep_detach_times: List[float]
) -> Tuple[float, float, float]:
    """
    Aggregate results across replicates.
    
    Args:
        all_rep_cs: List of final chromatin state values
        all_rep_bprot: List of final bound protamine values
        all_rep_detach_times: List of detachment times
    
    Returns:
        Tuple of:
            - avg_final_cs: Average final chromatin state
            - avg_final_bprot: Average final bound protamine
            - avg_detach_time: Average detachment time (using nanmean)
    """
    avg_final_cs = np.mean(all_rep_cs)
    avg_final_bprot = np.mean(all_rep_bprot)
    avg_detach_time = np.nanmean(all_rep_detach_times)
    
    return avg_final_cs, avg_final_bprot, avg_detach_time


def run_single_replicate(
    nuc: Nucleosome,
    replicate_num: int,
    build_params: dict,
    tau_points: np.ndarray,
    inf_protamine: bool,
    tau_min: Optional[float],
    save_trajectories: bool,
    eff_stride: int,
    traj_data: dict
) -> Tuple[int, int, float]:
    """
    Run a single replicate simulation for a nucleosome.
    
    Args:
        nuc: Nucleosome instance
        replicate_num: Replicate number
        build_params: Dictionary with factory functions
        tau_points: Array of dimensionless time points
        inf_protamine: Whether to use infinite protamine
        tau_min: Minimum tau value for renucleation
        save_trajectories: Whether to save trajectory data
        eff_stride: Stride for sampling trajectory points
        traj_data: Dictionary to store trajectory data
    
    Returns:
        Tuple of (final_cs, final_bprot, detach_time)
    """
    total_steps = len(tau_points)
    
    # Create simulator with fresh instances
    sim = create_simulator(nuc, build_params, tau_points, inf_protamine, tau_min, replicate_num)
    
    # Process simulation states
    tau_times, cs_totals, bprots, detached_totals, final_cs, final_bprot, detach_time = \
        process_simulation_states(sim, save_trajectories, eff_stride, total_steps)
    
    # Store trajectory data if needed
    if save_trajectories:
        store_trajectory_data(traj_data, nuc, replicate_num, tau_times, cs_totals, bprots, detached_totals)
    
    return final_cs, final_bprot, detach_time
