"""Batch worker: run one batch of nucleosomes × replicates, write per-batch
temp output files. Designed to be submitted to a ProcessPoolExecutor.
"""

import datetime as dt
import os
from typing import List, Optional, Tuple

import numpy as np
import psutil

from src.core.helper import bkeep as bk
from src.core.nucleosomes import Nucleosome
from src.gillespie_event.aggregate import aggregate_replicates
from src.gillespie_event.output import (
    write_batch_tsv,
    write_batch_survival,
    write_batch_trajectories,
)
from src.gillespie_event.replicate import run_single_replicate


def run_batch(
    batch: List[Nucleosome],
    prot_params: dict,
    tau_max: float,
    n_survival_points: int,
    inf_protamine: bool,
    replicates: int,
    save_trajectories: bool,
) -> Tuple[str, str, Optional[str]]:
    """Return (tmp_tsv_path, tmp_survival_path, tmp_traj_path_or_None)."""
    start = dt.datetime.now()
    proc = psutil.Process(os.getpid())
    bk.WORKER_LOGGER.info(
        "Processing batch of %d nucleosomes × %d replicates",
        len(batch), replicates,
    )

    tau_grid = np.linspace(0.0, tau_max, n_survival_points)

    aggs = []
    for nuc_idx, nuc in enumerate(batch):
        results = []
        for r in range(replicates):
            results.append(run_single_replicate(
                nuc=nuc,
                replicate_num=r,
                prot_params=prot_params,
                tau_max=tau_max,
                inf_protamine=inf_protamine,
            ))
        # One line per nucleosome (not per replicate) keeps logs manageable
        # on large datasets while still giving progress signal.
        bk.WORKER_LOGGER.info(
            "  nuc %d/%d done (id=%s, subid=%s, replicates=%d)",
            nuc_idx + 1, len(batch), nuc.id, nuc.subid, replicates,
        )
        aggs.append(aggregate_replicates(nuc, results, tau_grid))

    tmp_tsv, _writer = bk.new_batch_writer(fmt="tsv", suffix=".tsv")
    tmp_tsv.close()
    write_batch_tsv(aggs, tmp_tsv.name)

    tmp_surv_path = bk.new_batch_writer(fmt="parquet", suffix=".parquet")
    write_batch_survival(aggs, tmp_surv_path)

    tmp_traj_path: Optional[str] = None
    if save_trajectories:
        tmp_traj_path = bk.new_batch_writer(fmt="parquet", suffix=".parquet")
        write_batch_trajectories(aggs, tmp_traj_path)

    rss_mb = proc.memory_info().rss / 2**20
    elapsed = (dt.datetime.now() - start).total_seconds()
    bk.WORKER_LOGGER.info(
        "Batch done: %d nucs, RSS %.1f MB, %.1fs",
        len(batch), rss_mb, elapsed,
    )

    return tmp_tsv.name, tmp_surv_path, tmp_traj_path
