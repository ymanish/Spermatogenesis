"""
Orchestrator Module
===================

Main orchestration function for running simulations with parallelization.

Author: MY
Date: 2025-11-16
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

from src.core.build_nucleosomes import nucleosome_generator
import src.core.helper.bkeep as bk
from src.core.helper.tau_min import _compute_tau_min
from src.config.var import create_nucleosomes_instance, create_protamines_instance
from src.config.custom_type import SimulationConfig

from .batch import run_batch_simulations
from .io import merge_output_files



def run_simulation(
    file_path: Path,
    traj_outfile: Path,
    tsv_outfile: Path,
    config: SimulationConfig,
    logger: Optional[logging.Logger] = None, 
    max_nucs: Optional[int]=None, 
    subids_range:Optional[tuple]=None   
) -> None:
    """
    Run Gillespie simulations using a SimulationConfig object.
    
    This is the recommended way to run simulations, as it encapsulates all
    configuration parameters in a single object for better maintainability.
    
    Args:
        file_path: Path to input TSV file with nucleosome data
        traj_outfile: Path for trajectory output (Parquet)
        tsv_outfile: Path for summary output (TSV)
        config: SimulationConfig object with all parameters
        logger: Logger instance (created if None)
    
    Example:
        >>> from src.simulation import run_simulation_with_config
        >>> from src.config.custom_type import SimulationConfig
        >>> 
        >>> config = SimulationConfig(
        ...     k_wrap=1.0,
        ...     prot_p_conc=100.0,
        ...     prot_cooperativity=4.5,
        ...     tau_max=10000.0,
        ...     tau_steps=1000,
        ...     replicates=20,
        ...     n_workers=10,
        ...     save_trajectories=True
        ... )
        >>> 
        >>> run_simulation_with_config(
        ...     file_path=Path("data/nucleosomes.tsv"),
        ...     traj_outfile=Path("output/trajectories.parquet"),
        ...     tsv_outfile=Path("output/summary.tsv"),
        ...     config=config
        ... )
    """
    if logger is None:
        from src.utils.logger_util import get_logger
        logger = get_logger(__name__, log_file=None, level='INFO')
    
    # Generate nucleosomes
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

    gen = itertools.islice(gen, max_nucs) if max_nucs is not None else gen

    # Create batches
    batches = bk.batcher(gen, config.batch_size)
    
    # Log configuration
    if config.renucleation:
        logger.warning("!!! Renucleation is ON. Make sure this is intended.")
        logger.warning(f"!!! tau_min = {config.tau_min:.4f}")
    else:
        logger.warning("Renucleation is OFF.")
    
    # Run parallel simulations
    temp_parquet_paths = []
    temp_tsv_paths = []
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.n_workers,
        initializer=bk.init_worker,
        initargs=(config.flush_every,)
    ) as pool:
        func = partial(
            run_batch_simulations,
            build_params=config.build_params,
            tau_points=config.tau_points,
            binding_sites=config.binding_sites,
            inf_protamine=config.inf_protamine,
            save_trajectories=config.save_trajectories,
            tau_min=config.tau_min,
            replicates=config.replicates,
            maxpoints_saved_trajectories=config.maxpoints_saved_trajectories
        )
        
        futures = [
            pool.submit(func, batch=batch, k_wrap=config.k_wrap)
            for batch in batches
        ]
        
        for fut in tqdm(concurrent.futures.as_completed(futures), desc="Processing batches"):
            tsv_path, parquet_path = fut.result()
            temp_parquet_paths.append(parquet_path)
            temp_tsv_paths.append(tsv_path)
    
    # Merge output files
    merge_output_files(
        temp_tsv_paths=temp_tsv_paths,
        temp_parquet_paths=temp_parquet_paths,
        tsv_outfile=tsv_outfile,
        traj_outfile=traj_outfile if config.save_trajectories else None,
        save_trajectories=config.save_trajectories,
        n_workers=config.n_workers,
        logger=logger
    )
