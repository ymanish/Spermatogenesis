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

from src.core.build_nucleosomes import nucleosome_generator, nucleosome_generator_sprm
import src.core.helper.bkeep as bk

from .config import MarkovConfig
from .batch import run_batch_markov
from .output import merge_markov_output_files, create_config_file


def run_markov_solver(
    tsv_outfile: Path,
    survival_outfile: Path,
    config: MarkovConfig,
    file_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    max_nucs: Optional[int] = None,
    subids_range: Optional[tuple] = None
) -> None:
    """
    Run Markov solver on multiple nucleosomes using parallelization.

    Provide exactly one of:
      - file_path:   path to an old-format TSV nucleosome file
      - dataset_dir: path to an SPRM dataset directory (energies.tsv + id_lookup.tsv)

    Args:
        tsv_outfile: Path for summary output (TSV)
        survival_outfile: Path for survival data output (Parquet)
        config: MarkovConfig object with all parameters
        file_path: Old-format input TSV (mutually exclusive with dataset_dir)
        dataset_dir: SPRM dataset directory (mutually exclusive with file_path)
        logger: Logger instance (created if None)
        max_nucs: Maximum number of nucleosomes to process (None = all)
        subids_range: Tuple (start, end) for subid filtering; ignored for SPRM input

    Output Files:
        - Summary TSV: MFPT, half-life, final survival, mean survival
        - Survival Parquet: Detailed survival/state data (if enabled)
    """
    if logger is None:
        from src.utils.logger_util import get_logger
        logger = get_logger(__name__, log_file=None, level='INFO')

    if (file_path is None) == (dataset_dir is None):
        raise ValueError("Provide exactly one of file_path (old format) or dataset_dir (SPRM format).")

    # Define output paths
    tsv_output = Path(tsv_outfile)
    parquet_output = Path(survival_outfile) if (config.save_survival or config.save_states) else None

    # Log configuration
    input_label = str(dataset_dir) if dataset_dir is not None else str(file_path)
    input_kind = "dataset_dir (SPRM)" if dataset_dir is not None else "infile (old TSV)"
    logger.info("=" * 70)
    logger.info("MARKOV SOLVER EXECUTION")
    logger.info("=" * 70)
    logger.info(f"Input ({input_kind}): {input_label}")
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

    # Build nucleosome generator
    if dataset_dir is not None:
        gen = nucleosome_generator_sprm(
            dataset_dir=str(dataset_dir),
            k_wrap=config.k_wrap,
            binding_sites=config.binding_sites,
        )
    else:
        if subids_range is not None:
            gen = nucleosome_generator(
                file_path=file_path,
                k_wrap=config.k_wrap,
                binding_sites=config.binding_sites,
                subids=np.arange(*subids_range).tolist(),
            )
        else:
            gen = nucleosome_generator(
                file_path=file_path,
                k_wrap=config.k_wrap,
                binding_sites=config.binding_sites,
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
