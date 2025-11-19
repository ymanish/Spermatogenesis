"""
Batch Processing Module
=======================

Functions for running simulations in batches with parallelization.

Author: MY
Date: 2025-11-16
"""

import os
import datetime as dt
import psutil
import numpy as np
from typing import List, Optional, Tuple
from src.core.nucleosomes import Nucleosome
import src.core.helper.bkeep as bk
from .simulator import calculate_stride
from .replicate import run_single_replicate, aggregate_replicate_results
from .trajectory import save_trajectories_to_parquet


def run_batch_simulations(
    batch: List[Nucleosome],
    k_wrap: float,
    tau_min: float,
    build_params: dict,
    tau_points: np.ndarray,
    inf_protamine: bool = True,
    kT: float = 1.0,
    binding_sites: int = 14,
    save_trajectories: bool = False,
    replicates: int = 1,
    maxpoints_saved_trajectories: int = 100
) -> Tuple[str, Optional[str]]:
    """
    Run Gillespie simulations for a batch of nucleosomes with replicates.
    
    Args:
        batch: List of nucleosome instances
        k_wrap: Nucleosome wrapping constant
        tau_min: Minimum tau for renucleation
        build_params: Dictionary with factory functions for creating fresh instances
        tau_points: Array of dimensionless time points
        inf_protamine: Whether to use infinite protamine
        kT: Boltzmann constant * temperature
        binding_sites: Number of binding sites
        save_trajectories: Whether to save trajectory data
        replicates: Number of replicates per nucleosome
        maxpoints_saved_trajectories: Maximum trajectory points to save
    
    Returns:
        Tuple of (tsv_path, parquet_path)
            - tsv_path: Path to temporary TSV file with summary data
            - parquet_path: Path to temporary Parquet file with trajectories (None if not saved)
    """
    start_g = dt.datetime.now()
    proc = psutil.Process(os.getpid())
    bk.WORKER_LOGGER.info(
        "Processing batch of %d sequences with %d replicates each",
        len(batch), replicates
    )

    # Initialize data structures - nested dict for efficient storage
    traj_data = {}  # Structure: traj_data[id][subid][replicate] = {time_series}
    tmpfile_tsv, writer_tsv = bk.new_batch_writer(fmt="tsv", suffix=".tsv")
    bk.WORKER_LOGGER.info("Temporary file created: %s", tmpfile_tsv)

    parquet_path = None
    if save_trajectories:
        parquet_path = bk.new_batch_writer(fmt="parquet", suffix=".parquet")
        bk.WORKER_LOGGER.info("Temporary Parquet file created: %s", parquet_path)

    # Calculate stride for trajectory sampling
    eff_stride = calculate_stride(len(tau_points), maxpoints_saved_trajectories)

    # Process each nucleosome in the batch
    for nuc_idx, nuc in enumerate(batch):
        all_rep_cs = []
        all_rep_bprot = []
        all_rep_detach_times = []
        
        # Run replicates
        for r in range(replicates):
            final_cs, final_bprot, detach_time = run_single_replicate(
                nuc, r, build_params, tau_points, inf_protamine, tau_min,
                save_trajectories, eff_stride, traj_data
            )
            
            all_rep_cs.append(final_cs)
            all_rep_bprot.append(final_bprot)
            all_rep_detach_times.append(detach_time)
        
        # Aggregate and write results
        avg_final_cs, avg_final_bprot, avg_detach_time = aggregate_replicate_results(
            all_rep_cs, all_rep_bprot, all_rep_detach_times
        )
        writer_tsv.writerow([
            nuc.id, nuc.subid, replicates,
            avg_final_cs, avg_final_bprot, avg_detach_time
        ])

    # Log performance metrics
    rss = proc.memory_info().rss / 2**20
    bk.WORKER_LOGGER.info(
        "Batch of %d done by %s; RSS %.1f MB; t %.1fs",
        len(batch), os.getpid(), rss,
        (dt.datetime.now() - start_g).total_seconds()
    )

    # Save trajectory data if enabled
    if save_trajectories:
        save_trajectories_to_parquet(traj_data, parquet_path)
    else:
        parquet_path = None

    bk.WORKER_LOGGER.info("Temporary file %s written", tmpfile_tsv.name)
    return tmpfile_tsv.name, parquet_path
