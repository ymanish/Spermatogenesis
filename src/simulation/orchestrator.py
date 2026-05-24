"""
Orchestrator Module
===================

Main orchestration function for running simulations with parallelization.
"""

import os
import itertools
import concurrent.futures
import numpy as np
from pathlib import Path
from functools import partial
from tqdm import tqdm
from typing import Optional
import logging

from src.core.build_nucleosomes import nucleosome_generator, nucleosome_generator_sprm
import src.core.helper.bkeep as bk
from src.config.custom_type import SimulationConfig

from .batch import run_batch_simulations
from .io import merge_output_files


def run_simulation(
    traj_outfile: Path,
    tsv_outfile: Path,
    config: SimulationConfig,
    file_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    max_nucs: Optional[int] = None,
    subids_range: Optional[tuple] = None
) -> None:
    """
    Run Gillespie simulations using a SimulationConfig object.

    Provide exactly one of:
      - file_path:   path to an old-format TSV nucleosome file
      - dataset_dir: path to an SPRM dataset directory (energies.tsv + id_lookup.tsv)

    Args:
        traj_outfile:  Output path for trajectory Parquet file
        tsv_outfile:   Output path for summary TSV file
        config:        SimulationConfig with all parameters
        file_path:     Old-format input TSV (mutually exclusive with dataset_dir)
        dataset_dir:   SPRM dataset directory (mutually exclusive with file_path)
        logger:        Logger instance (created if None)
        max_nucs:      Limit number of nucleosomes processed (for testing)
        subids_range:  (start, end) tuple to filter old-format subids (ignored for SPRM)
    """
    if logger is None:
        from src.utils.logger_util import get_logger
        logger = get_logger(__name__, log_file=None, level='INFO')

    if (file_path is None) == (dataset_dir is None):
        raise ValueError("Provide exactly one of file_path (old format) or dataset_dir (SPRM format).")

    # Build nucleosome generator
    if dataset_dir is not None:
        gen = nucleosome_generator_sprm(
            dataset_dir=dataset_dir,
            k_wrap=config.k_wrap,
            kT=1.0,
            binding_sites=config.binding_sites
        )
    else:
        if subids_range is not None:
            gen = nucleosome_generator(
                file_path=file_path,
                k_wrap=config.k_wrap,
                binding_sites=config.binding_sites,
                subids=np.arange(*subids_range).tolist()
            )
        else:
            gen = nucleosome_generator(
                file_path=file_path,
                k_wrap=config.k_wrap,
                binding_sites=config.binding_sites
            )

    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)

    batches = bk.batcher(gen, config.batch_size)

    if config.renucleation:
        logger.warning("!!! Renucleation is ON. Make sure this is intended.")
        logger.warning(f"!!! tau_min = {config.tau_min:.4f}")
    else:
        logger.warning("Renucleation is OFF.")

    temp_parquet_paths = []
    temp_tsv_paths = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.n_workers,
        initializer=bk.init_worker,
        initargs=(config.flush_every,)
    ) as pool:
        func = partial(
            run_batch_simulations,
            prot_params=config.prot_params,
            tau_points=config.tau_points,
            tau_min=config.tau_min,
            inf_protamine=config.inf_protamine,
            save_trajectories=config.save_trajectories,
            replicates=config.replicates,
            maxpoints_saved_trajectories=config.maxpoints_saved_trajectories
        )

        futures = [pool.submit(func, batch=batch) for batch in batches]

        for fut in tqdm(concurrent.futures.as_completed(futures), desc="Processing batches"):
            tsv_path, parquet_path = fut.result()
            temp_parquet_paths.append(parquet_path)
            temp_tsv_paths.append(tsv_path)

    merge_output_files(
        temp_tsv_paths=temp_tsv_paths,
        temp_parquet_paths=temp_parquet_paths,
        tsv_outfile=tsv_outfile,
        traj_outfile=traj_outfile if config.save_trajectories else None,
        save_trajectories=config.save_trajectories,
        n_workers=config.n_workers,
        logger=logger
    )
