"""
Output Module
=============

Functions for saving Markov solver results to files.

Author: MY
Date: 2025-12-11
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import src.core.helper.bkeep as bk
from .config import MarkovConfig
import pandas as pd

def save_batch_to_temp_files(detailed_data: List[Dict], parquet_path: str) -> None:
    """
    Save detailed batch data to temporary Parquet file.
    
    Args:
        detailed_data: List of dictionaries with survival/state data
        parquet_path: Path to temporary Parquet file
    """
    if not detailed_data:
        return
    
    # Limit PyArrow parallelism since we're already in a worker process
    import pyarrow as pa
    pa.set_cpu_count(1)          # Limits compute threads
    pa.set_io_thread_count(1)    # Limits I/O threads
    
    # Convert numpy arrays/lists to plain Python lists for Polars
    # This ensures proper type inference for List columns
    cleaned_data = []
    for entry in detailed_data:
        cleaned_entry = {
            'id': entry['id'],
            'subid': entry['subid'],
            'tau_grid': entry['tau_grid'].tolist() if hasattr(entry['tau_grid'], 'tolist') else list(entry['tau_grid']),
            'survival': entry['survival'].tolist() if hasattr(entry['survival'], 'tolist') else list(entry['survival'])
        }
        
        # Add optional fields if present
        if 'state_probs' in entry:
            cleaned_entry['state_probs'] = entry['state_probs'].tolist() if hasattr(entry['state_probs'], 'tolist') else list(entry['state_probs'])
        if 'states' in entry:
            cleaned_entry['states'] = entry['states'].tolist() if hasattr(entry['states'], 'tolist') else list(entry['states'])
        if 'mfpt' in entry:
            cleaned_entry['mfpt'] = float(entry['mfpt'])
        if 'mfpt_vec' in entry:
            cleaned_entry['mfpt_vec'] = entry['mfpt_vec'].tolist() if hasattr(entry['mfpt_vec'], 'tolist') else list(entry['mfpt_vec'])
        
        cleaned_data.append(cleaned_entry)
    
    # Build schema_overrides dynamically based on what fields are present
    schema_overrides = {
        'tau_grid': pl.List(pl.Float64),
        'survival': pl.List(pl.Float64)
    }
    
    # Add optional fields if they exist in the data
    if 'mfpt_vec' in cleaned_data[0]:
        schema_overrides['mfpt_vec'] = pl.List(pl.Float64)
    if 'state_probs' in cleaned_data[0]:
        schema_overrides['state_probs'] = pl.List(pl.Float64)
    if 'states' in cleaned_data[0]:
        schema_overrides['states'] = pl.List(pl.Int64)
    
    # # Convert to Polars DataFrame with explicit schema for list columns
    # df = pl.DataFrame(cleaned_data, schema_overrides=schema_overrides)
    # df.write_parquet(parquet_path, compression='snappy')
    # print("cleaned_data:", cleaned_data)

    df = pd.DataFrame(cleaned_data)
    df.to_parquet(parquet_path, engine="pyarrow", compression='snappy')


def save_markov_results_to_parquet(
    temp_files: List[str],
    output_path: Path, 
    n_workers: int
) -> None:
    """
    Merge temporary Parquet files into final output.
    
    Args:
        temp_files: List of temporary Parquet file paths
        output_path: Final output Parquet file path
        n_workers: Number of workers for Polars parallelization
    """
    import os
    
    if not temp_files:
        if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
            bk.WORKER_LOGGER.warning("No temporary Parquet files to merge")
        return
    
    # Set Polars thread limit
    os.environ['POLARS_MAX_THREADS'] = str(n_workers)
    
    # Filter valid Parquet files
    valid_parquet_paths = [p for p in temp_files if Path(p).exists()]
    
    if not valid_parquet_paths:
        if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
            bk.WORKER_LOGGER.warning("No valid Parquet files found")
        return
    
    # Merge Parquet files using Polars lazy API (no full load into memory)
    df_lazy = pl.concat(
        [pl.scan_parquet(p) for p in valid_parquet_paths],
        how='vertical'
    )
    df_lazy.sink_parquet(output_path)  # Writes without full materialization
    
    if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
        bk.WORKER_LOGGER.info(f"Merged {len(valid_parquet_paths)} Parquet files to {output_path}")
    
    # Clean up temporary parquet files
    for p in valid_parquet_paths:
        try:
            Path(p).unlink()
        except Exception as e:
            if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
                bk.WORKER_LOGGER.warning(f"Failed to delete temp file {p}: {e}")


def save_markov_summary_to_tsv(
    temp_files: List[str],
    output_path: Path,
    config: MarkovConfig
) -> None:
    """
    Merge temporary TSV files into final summary output.
    
    Args:
        temp_files: List of temporary TSV file paths
        output_path: Final output TSV file path
        config: MarkovConfig instance for header information
    """
    if not temp_files:
        if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
            bk.WORKER_LOGGER.warning("No temporary TSV files to merge")
        return
    
    with open(output_path, 'w') as outfile:
        # Write header with configuration
        # outfile.write("# Markov Solver Results\n")
        # outfile.write(f"# k_wrap: {config.k_wrap}\n")
        # outfile.write(f"# prot_p_conc: {config.prot_p_conc}\n")
        # outfile.write(f"# prot_cooperativity: {config.prot_cooperativity}\n")
        # outfile.write(f"# tau_max: {config.tau_max}\n")
        # outfile.write(f"# tau_steps: {config.tau_steps}\n")
        # outfile.write(f"# method: {config.method}\n")
        # outfile.write("#\n")
        
        # Column headers
        # outfile.write("id\tsubid\tmfpt\thalf_life\tfinal_survival\tmean_survival\n")
        outfile.write("id\tsubid\tmfpt\n")

        
        # Merge data from temp files
        for temp_file in temp_files:
            if not Path(temp_file).exists():
                continue
            with open(temp_file, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    
    if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
        bk.WORKER_LOGGER.info(f"Merged {len(temp_files)} TSV files to {output_path}")


def merge_markov_output_files(
    temp_tsv_files: List[str],
    temp_parquet_files: List[str],
    tsv_output: Path,
    parquet_output: Optional[Path],
    config: MarkovConfig
) -> None:
    """
    Merge all temporary files into final outputs.
    
    Args:
        temp_tsv_files: List of temporary TSV file paths
        temp_parquet_files: List of temporary Parquet file paths
        tsv_output: Final TSV output path
        parquet_output: Final Parquet output path (None to skip)
        config: MarkovConfig instance
    """
    # Merge TSV files (summary)
    save_markov_summary_to_tsv(temp_tsv_files, tsv_output, config)
    
    # Merge Parquet files (detailed data)
    if parquet_output and temp_parquet_files:
        save_markov_results_to_parquet(temp_parquet_files, parquet_output, config.n_workers)
    
    # Clean up temporary TSV files (Parquet files are cleaned up in save_markov_results_to_parquet)
    for temp_file in temp_tsv_files:
        try:
            Path(temp_file).unlink()
        except Exception as e:
            if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
                bk.WORKER_LOGGER.warning(f"Failed to delete temp file {temp_file}: {e}")


def create_config_file(output_dir: Path, config: 'MarkovConfig') -> None:
    """
    Save configuration to a separate file for reproducibility.
    
    Args:
        output_dir: Output directory
        config: MarkovConfig instance
    """
    config_path = output_dir / "markov_config.txt"
    
    with open(config_path, 'w') as f:
        f.write("Markov Solver Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        info = config.get_info_dict()
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    # Use bk.WORKER_LOGGER if available, otherwise just pass
    if hasattr(bk, 'WORKER_LOGGER') and bk.WORKER_LOGGER is not None:
        bk.WORKER_LOGGER.info(f"Configuration saved to {config_path}")
