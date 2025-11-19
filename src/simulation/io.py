"""
I/O Module
==========

Functions for file input/output operations.

Author: MY
Date: 2025-11-16
"""

import os
import shutil
import polars as pl
from pathlib import Path
from typing import List, Optional
import logging


def merge_output_files(
    temp_tsv_paths: List[str],
    temp_parquet_paths: List[Optional[str]],
    tsv_outfile: Path,
    traj_outfile: Optional[Path],
    save_trajectories: bool,
    n_workers: int,
    logger: logging.Logger
) -> None:
    """
    Merge temporary output files into final output files.
    
    Args:
        temp_tsv_paths: List of temporary TSV file paths
        temp_parquet_paths: List of temporary Parquet file paths (may contain None)
        tsv_outfile: Path to final TSV output file
        traj_outfile: Path to final trajectory Parquet file (None if not saving)
        save_trajectories: Whether trajectories were saved
        n_workers: Number of workers for Polars
        logger: Logger instance
    """
    # Merge trajectory files if enabled
    if save_trajectories and traj_outfile:
        os.environ['POLARS_MAX_THREADS'] = str(n_workers)
        logger.info("All batches processed, merging trajectory files with Polars")
        
        # Merge Parquet files using Polars lazy API (no full load into memory)
        valid_parquet_paths = [p for p in temp_parquet_paths if p is not None]
        df_lazy = pl.concat(
            [pl.scan_parquet(p) for p in valid_parquet_paths],
            how='vertical'
        )
        df_lazy.sink_parquet(traj_outfile)  # Writes without full materialization
        
        logger.info(f"Merged trajectory data saved to: {traj_outfile}")
        
        # Clean up temporary parquet files
        for p in valid_parquet_paths:
            os.remove(p)
    
    # Merge TSV files
    HEADER = ['id', 'subid', 'n_replicates', 'avg_cs_total', 'avg_bprot', 'avg_detach_time']
    
    with open(tsv_outfile, "w") as final_tsv:
        final_tsv.write("\t".join(HEADER) + "\n")
        
        for path in temp_tsv_paths:
            with open(path, "r") as src:
                shutil.copyfileobj(src, final_tsv)
            os.remove(path)
    
    logger.info(f"Summary saved to: {tsv_outfile}")
    if traj_outfile:
        logger.info(f"Trajectories saved to: {traj_outfile}")
