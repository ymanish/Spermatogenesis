"""
Batch Processing Module
=======================

Functions for running Markov solver on batches of nucleosomes with parallelization.

Author: MY
Date: 2025-12-11
"""

import os
import datetime as dt
import psutil
from typing import List, Tuple, Optional
from pathlib import Path
import src.core.helper.bkeep as bk
from src.core.nucleosomes import Nucleosome
# from .solver_runner import solve_single_nucleosome, compute_derived_quantities
from .solver_runner import solve_single_nucleosome
from .output import save_batch_to_temp_files
from .config import MarkovConfig


def run_batch_markov(
    batch: List[Nucleosome],
    config: MarkovConfig,
    save_survival: bool = True,
    save_states: bool = False,
    save_mfpt: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Run Markov solver for a batch of nucleosomes.
    
    Args:
        batch: List of nucleosome instances
        config: MarkovConfig instance with all parameters
        save_survival: Whether to save survival functions
        save_states: Whether to save state probabilities
        save_mfpt: Whether to save MFPT values
    
    Returns:
        Tuple of (tsv_path, parquet_path)
            - tsv_path: Path to temporary TSV file with summary data
            - parquet_path: Path to temporary Parquet file with survival/state data (None if not saved)
    """
    start_time = dt.datetime.now()
    proc = psutil.Process(os.getpid())
    
    # Initialize logger if available
    logger = bk.WORKER_LOGGER if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None else None
    
    if logger:
        logger.info(
            "Processing batch of %d nucleosomes with Markov solver",
            len(batch)
        )
    
    # Initialize data structures
    summary_data = []
    detailed_data = []  # For survival functions and state probabilities
    
    tmpfile_tsv, writer_tsv = bk.new_batch_writer(fmt="tsv", suffix=".tsv")
    if logger:
        logger.info("Temporary TSV file created: %s", tmpfile_tsv)
    
    parquet_path = None
    if save_survival or save_states:
        parquet_path = bk.new_batch_writer(fmt="parquet", suffix=".parquet")
        if logger:
            logger.info("Temporary Parquet file created: %s", parquet_path)
    
    # Process each nucleosome in the batch
    for nuc_idx, nuc in enumerate(batch):
        if logger:
            logger.info(
                f"Solving nucleosome {nuc_idx + 1}/{len(batch)}: ID={nuc.id}, SubID={nuc.subid}"
            )
        
        # Solve Markov chain
        results = solve_single_nucleosome(
            nuc,
            tau_grid=config.tau_grid,
            k_wrap=config.k_wrap,
            protamine_params=config.protamine_params,
            kT=config.kT,
            binding_sites=config.binding_sites,
            method=config.method,
            sparse=config.sparse,
            dimensionless=config.dimensionless,
            compute_states=save_states,
            start_state=(0, 0)
        )
        
        # # Compute derived quantities
        # derived = compute_derived_quantities(results)
        
        # Write summary to TSV
        writer_tsv.writerow([
            results['id'],
            results['subid'],
            results['mfpt'],
            # derived['half_life'],
            # derived['final_survival'],
            # derived['mean_survival']
        ])
        
        # Store detailed data for Parquet
        if save_survival or save_states:
            detailed_entry = {
                'id': results['id'],
                'subid': results['subid'],
                'tau_grid': results['tau_grid'],
                'survival': results['survival']
            }
            
            if save_states and 'state_probs' in results:
                detailed_entry['state_probs'] = results['state_probs']
                detailed_entry['states'] = results['states']
            
            if save_mfpt:
                detailed_entry['mfpt'] = results['mfpt']
                detailed_entry['mfpt_vec'] = results['mfpt_vec']
            
            detailed_data.append(detailed_entry)

    # Save detailed data to Parquet if requested
    if parquet_path and detailed_data:
        save_batch_to_temp_files(detailed_data, parquet_path)

    end_time = dt.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    mem_mb = proc.memory_info().rss / 1024 / 1024
    
    if logger:
        logger.info(
            f"Batch complete: {len(batch)} nucleosomes in {elapsed:.2f}s, "
            f"Memory: {mem_mb:.1f} MB"
        )
        logger.info("Temporary TSV file created: %s", tmpfile_tsv.name)
        if parquet_path:
            logger.info("Temporary Parquet file created: %s", parquet_path)

    return tmpfile_tsv.name, parquet_path
