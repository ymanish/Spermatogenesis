"""
Orchestrator Module
===================

Main orchestration function for running Markov solver with parallelization.

Author: MY
Date: 2025-12-11
"""

import os
import itertools
import concurrent.futures
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import logging

from src.core.build_nucleosomes import nucleosome_generator
import src.core.helper.bkeep as bk
from src.analysis.markov_solver import load_nucleosomes_from_file

from .config import MarkovConfig
from .batch import run_batch_markov
from .output import merge_markov_output_files, create_config_file


def run_markov_solver(
    file_path: Path,
    tsv_outfile: Path,
    survival_outfile: Path,
    config: MarkovConfig,
    logger: Optional[logging.Logger] = None,
    max_nucs: Optional[int] = None,
    subids_range: Optional[tuple] = None
) -> None:
    """
    Run Markov solver on multiple nucleosomes using parallelization.
    
    This function orchestrates the execution of Markov chain calculations
    for multiple nucleosomes, with batching and parallel processing.
    
    Args:
        file_path: Path to input TSV file with nucleosome data
        tsv_outfile: Path for summary output (TSV)
        survival_outfile: Path for survival data output (Parquet)
        config: MarkovConfig object with all parameters
        logger: Logger instance (created if None)
        max_nucs: Maximum number of nucleosomes to process (None = all)
        subids_range: Tuple (min, max) for subid filtering (None = all)
    
    Output Files:
        - Summary TSV: MFPT, half-life, final survival, mean survival
        - Survival Parquet: Detailed survival/state data (if enabled)
    
    Example:
        >>> from src.markov_execution import run_markov_solver, MarkovConfig
        >>> from pathlib import Path
        >>> 
        >>> config = MarkovConfig(
        ...     k_wrap=1.0,
        ...     prot_p_conc=10.0,
        ...     prot_cooperativity=0.0,
        ...     tau_max=1000.0,
        ...     tau_steps=500,
        ...     n_workers=10,
        ...     save_survival=True,
        ...     save_mfpt=True
        ... )
        >>> 
        >>> run_markov_solver(
        ...     file_path=Path("data/nucleosomes.tsv"),
        ...     tsv_outfile=Path("output/markov_summary.tsv"),
        ...     survival_outfile=Path("output/markov_survival.parquet"),
        ...     config=config,
        ...     max_nucs=50
        ... )
    """
    if logger is None:
        from src.utils.logger_util import get_logger
        logger = get_logger(__name__, log_file=None, level='INFO')
    
    # Define output paths
    tsv_output = Path(tsv_outfile)
    parquet_output = Path(survival_outfile) if (config.save_survival or config.save_states) else None
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("MARKOV SOLVER EXECUTION")
    logger.info("=" * 70)
    logger.info(f"Input file: {file_path}")
    logger.info(f"Summary output: {tsv_output}")
    if parquet_output:
        logger.info(f"Survival output: {parquet_output}")
    logger.info(f"k_wrap: {config.k_wrap}")
    logger.info(f"Protamine concentration: {config.prot_p_conc} μM")
    logger.info(f"Cooperativity: {config.prot_cooperativity} kT")
    logger.info(f"Time grid: {config.tau_steps} points, τ_max={config.tau_max}")        
    logger.info(f"Method: {config.method}")
    logger.info(f"Workers: {config.n_workers}")
    logger.info(f"Batch size: {config.batch_size}")
    
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
    
    # Limit number of nucleosomes if requested
    max_nucs = max_nucs if max_nucs is not None else config.max_nucs
    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)
        logger.info(f"Limiting to {max_nucs} nucleosomes")
    
    # Create batches
    batches = bk.batcher(gen, config.batch_size)
    
    # Temporary file storage
    temp_tsv_files = []
    temp_parquet_files = []

    # Parallel processing
    logger.info(f"Running in parallel mode with {config.n_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.n_workers,
                                                 initializer=bk.init_worker) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(
                                    run_batch_markov,
                                    batch,
                                    config,
                                    config.save_survival,
                                        config.save_states,
                                        config.save_mfpt
                                    )
            futures.append(future)
            
        # Collect results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing batches"
        ):
        
            tsv_path, parquet_path = future.result()
            temp_tsv_files.append(tsv_path)
            if parquet_path:
                temp_parquet_files.append(parquet_path)
            # logger.error(f"Batch processing failed:", exc_info=True)

# Merge temporary files
    logger.info("Merging output files...")
    merge_markov_output_files(
        temp_tsv_files,
        temp_parquet_files,
        tsv_output,
        parquet_output,
        config
    )
    
    # Summary
    logger.info("=" * 70)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Summary saved to: {tsv_output}")
    if parquet_output:
        logger.info(f"Detailed results saved to: {parquet_output}")
    logger.info("=" * 70)
