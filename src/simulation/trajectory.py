"""
Trajectory Module
=================

Functions for handling trajectory data storage and conversion.

Author: MY
Date: 2025-11-16
"""

import pandas as pd
import pyarrow as pa
from typing import List, Dict
from src.core.nucleosomes import Nucleosome


def store_trajectory_data(
    traj_data: dict,
    nuc: Nucleosome,
    r: int,
    tau_times: List[float],
    cs_totals: List[int],
    bprots: List[int],
    detached_totals: List[int]
) -> None:
    """
    Store trajectory data for a single replicate using efficient nested structure.
    
    Structure: traj_data[id][subid][replicate] = {time_series_data}
    This avoids storing id/subid/replicate for every time point.
    
    Args:
        traj_data: Nested dictionary for trajectory data
        nuc: Nucleosome instance with id and subid
        r: Replicate number
        tau_times: List of tau time points
        cs_totals: List of chromatin state totals
        bprots: List of bound protamine counts
        detached_totals: List of detached nucleosome counts
    """
    # Create nested structure if not exists
    if nuc.id not in traj_data:
        traj_data[nuc.id] = {}
    if nuc.subid not in traj_data[nuc.id]:
        traj_data[nuc.id][nuc.subid] = {}
    
    # Store time series data for this replicate
    traj_data[nuc.id][nuc.subid][r] = {
        'tau_time': tau_times,
        'cs_total': cs_totals,
        'bprot': bprots,
        'detached_total': detached_totals
    }


def convert_trajectory_to_dataframe(traj_data: dict) -> pd.DataFrame:
    """
    Convert nested trajectory dict to DataFrame with list columns.
    
    This format is Polars-compatible and allows lazy loading:
    - Each row = one trajectory (one nucleosome replicate)
    - Time series stored as lists in single cells
    - id/subid/replicate stored once per trajectory
    
    Args:
        traj_data: Nested dictionary with trajectory data
    
    Returns:
        DataFrame with efficient list columns
    
    Example:
        >>> df = convert_trajectory_to_dataframe(traj_data)
        >>> df.columns
        Index(['id', 'subid', 'replicate', 'tau_time', 'cs_total', 'bprot', 'detached_total'])
    """
    rows = []
    
    for nuc_id in traj_data:
        for subid in traj_data[nuc_id]:
            for replicate in traj_data[nuc_id][subid]:
                data = traj_data[nuc_id][subid][replicate]
                
                # Each row is one complete trajectory with list columns
                rows.append({
                    'id': nuc_id,
                    'subid': subid,
                    'replicate': replicate,
                    'tau_time': data['tau_time'],
                    'cs_total': data['cs_total'],
                    'bprot': data['bprot'],
                    'detached_total': data['detached_total']
                })
    
    return pd.DataFrame(rows)


def save_trajectories_to_parquet(traj_data: dict, parquet_path: str) -> None:
    """
    Save trajectory data to Parquet format.
    
    Parquet efficiently stores list columns, so we get:
    - Compact storage (id/subid/replicate stored once per trajectory)
    - Polars lazy loading support
    - Ability to filter before loading
    - Columnar compression
    
    Args:
        traj_data: Nested dictionary with trajectory data
        parquet_path: Path to output Parquet file
    
    Example:
        >>> save_trajectories_to_parquet(traj_data, "output/trajectories.parquet")
    """
    pa.set_cpu_count(1)  # Limits compute threads
    pa.set_io_thread_count(1)  # Limits I/O threads
    
    df = convert_trajectory_to_dataframe(traj_data)
    df.to_parquet(parquet_path, engine="pyarrow", compression='snappy')
