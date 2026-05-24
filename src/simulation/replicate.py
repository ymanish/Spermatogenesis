"""
Replicate Module
================

Functions for running single replicate simulations.
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
    Iterate simulation states and collect trajectory data.

    Returns:
        (tau_times, cs_totals, bprots, detached_totals,
         final_cs, final_bprot, detach_time)
    """
    tau_times = []
    cs_totals = []
    bprots = []
    detached_totals = []
    detach_time = np.nan
    final_cs = 0
    final_bprot = 0

    for step, state in enumerate(sim.run()):
        final_cs = state.cs_total
        final_bprot = state.bprot

        if save_trajectories and (step % eff_stride == 0 or step == total_steps - 1):
            tau_times.append(state.tau)
            cs_totals.append(state.cs_total)
            bprots.append(state.bprot)
            detached_totals.append(state.detached_total)

        if state.detached_total > 0 and np.isnan(detach_time):
            detach_time = state.time

    return tau_times, cs_totals, bprots, detached_totals, final_cs, final_bprot, detach_time


def run_single_replicate(
    nuc: Nucleosome,
    replicate_num: int,
    prot_params: dict,
    tau_points: np.ndarray,
    inf_protamine: bool,
    tau_min: Optional[float],
    save_trajectories: bool,
    eff_stride: int,
    traj_data: dict
) -> Tuple[int, int, float]:
    """
    Run one replicate simulation for a nucleosome.

    Returns:
        (final_cs, final_bprot, detach_time)
    """
    total_steps = len(tau_points)
    sim = create_simulator(nuc, prot_params, tau_points, inf_protamine, tau_min, replicate_num)

    tau_times, cs_totals, bprots, detached_totals, final_cs, final_bprot, detach_time = \
        process_simulation_states(sim, save_trajectories, eff_stride, total_steps)

    if save_trajectories:
        store_trajectory_data(traj_data, nuc, replicate_num, tau_times, cs_totals, bprots, detached_totals)

    return final_cs, final_bprot, detach_time
