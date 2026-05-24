"""
Batch Processing Module
=======================

Functions for running simulations in batches with parallelization.
"""

import os
import datetime as dt
import psutil
import numpy as np
from typing import List, Optional, Tuple
from src.core.nucleosomes import Nucleosome
import src.core.helper.bkeep as bk
from .simulator import calculate_stride
from .replicate import run_single_replicate
from .trajectory import save_trajectories_to_parquet


def run_batch_simulations(
    batch: List[Nucleosome],
    prot_params: dict,
    tau_points: np.ndarray,
    tau_min: Optional[float],
    inf_protamine: bool = True,
    save_trajectories: bool = False,
    replicates: int = 1,
    maxpoints_saved_trajectories: int = 100
) -> Tuple[str, Optional[str]]:
    """
    Run Gillespie simulations for a batch of nucleosomes with replicates.

    Args:
        batch:                       List of Nucleosome instances
        prot_params:                 Dict with protamine parameters (k_unbind, k_bind, p_conc, cooperativity)
        tau_points:                  Array of dimensionless time points
        tau_min:                     Minimum dwell tau for renucleation (None = disabled)
        inf_protamine:               Whether to use infinite protamine supply
        save_trajectories:           Whether to write full trajectory data
        replicates:                  Number of replicates per nucleosome
        maxpoints_saved_trajectories: Maximum trajectory points to save per replicate

    Returns:
        (tsv_path, parquet_path) — paths to temporary output files
    """
    start_g = dt.datetime.now()
    proc = psutil.Process(os.getpid())
    bk.WORKER_LOGGER.info(
        "Processing batch of %d sequences with %d replicates each",
        len(batch), replicates
    )

    traj_data = {}
    tmpfile_tsv, writer_tsv = bk.new_batch_writer(fmt="tsv", suffix=".tsv")
    bk.WORKER_LOGGER.info("Temporary file created: %s", tmpfile_tsv)

    parquet_path = None
    if save_trajectories:
        parquet_path = bk.new_batch_writer(fmt="parquet", suffix=".parquet")
        bk.WORKER_LOGGER.info("Temporary Parquet file created: %s", parquet_path)

    eff_stride = calculate_stride(len(tau_points), maxpoints_saved_trajectories)

    for nuc_idx, nuc in enumerate(batch):
        all_rep_cs = []
        all_rep_bprot = []
        all_rep_detach_times = []

        for r in range(replicates):
            print(f"Simulating nucleosome {nuc_idx + 1}/{len(batch)}, "
                  f"Replicate {r + 1}/{replicates}: ID={nuc.id}, SubID={nuc.subid}")

            final_cs, final_bprot, detach_time = run_single_replicate(
                nuc, r, prot_params, tau_points, inf_protamine, tau_min,
                save_trajectories, eff_stride, traj_data
            )

            all_rep_cs.append(final_cs)
            all_rep_bprot.append(final_bprot)
            all_rep_detach_times.append(detach_time)

        avg_final_cs = np.mean(all_rep_cs)
        avg_final_bprot = np.mean(all_rep_bprot)
        avg_detach_time = np.nanmean(all_rep_detach_times)

        writer_tsv.writerow([
            nuc.id, nuc.subid, replicates,
            avg_final_cs, avg_final_bprot, avg_detach_time
        ])

    rss = proc.memory_info().rss / 2**20
    bk.WORKER_LOGGER.info(
        "Batch of %d done by %s; RSS %.1f MB; t %.1fs",
        len(batch), os.getpid(), rss,
        (dt.datetime.now() - start_g).total_seconds()
    )

    if save_trajectories:
        save_trajectories_to_parquet(traj_data, parquet_path)
    else:
        parquet_path = None

    bk.WORKER_LOGGER.info("Temporary file %s written", tmpfile_tsv.name)
    return tmpfile_tsv.name, parquet_path
