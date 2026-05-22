"""
Utility to load and query trajectory data from Parquet files using Polars.

The trajectory data is stored in Parquet format with list columns:
- Each row = one trajectory (one nucleosome replicate)
- id, subid, replicate stored once per row
- Time series (tau_time, cs_total, bprot, detached_total) stored as lists

This format allows:
- Lazy loading (don't load everything into memory)
- Filtering before loading (e.g., only specific nucleosomes)
- Efficient columnar storage with compression
- Polars parallel operations
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def load_trajectory_data(parquet_path: str, lazy: bool = True) -> pl.LazyFrame:
    """
    Load trajectory data from Parquet file using Polars.
    
    Args:
        parquet_path: Path to the Parquet file
        lazy: If True, returns LazyFrame (doesn't load into memory). 
              If False, returns DataFrame (loads all data).
    
    Returns:
        Polars LazyFrame or DataFrame with trajectory data
    """
    if lazy:
        return pl.scan_parquet(parquet_path)
    else:
        return pl.read_parquet(parquet_path)


def get_trajectory(parquet_path: str, nuc_id: int, subid: int, replicate: Optional[int] = None) -> Dict:
    """
    Get trajectory for a specific nucleosome using lazy loading.
    
    Only loads the specific rows needed, very memory efficient!
    
    Args:
        parquet_path: Path to the Parquet file
        nuc_id: Nucleosome ID
        subid: Nucleosome subid
        replicate: Specific replicate (if None, returns all replicates)
    
    Returns:
        If replicate specified: {'tau_time': [...], 'cs_total': [...], ...}
        If replicate is None: {0: {...}, 1: {...}, ...} for all replicates
    """
    # Build filter
    if replicate is not None:
        df = (pl.scan_parquet(parquet_path)
              .filter((pl.col('id') == nuc_id) & 
                     (pl.col('subid') == subid) & 
                     (pl.col('replicate') == replicate))
              .collect())
        
        if len(df) == 0:
            raise ValueError(f"No data found for id={nuc_id}, subid={subid}, replicate={replicate}")
        
        row = df.row(0, named=True)
        return {
            'tau_time': row['tau_time'],
            'cs_total': row['cs_total'],
            'bprot': row['bprot'],
            'detached_total': row['detached_total']
        }
    else:
        df = (pl.scan_parquet(parquet_path)
              .filter((pl.col('id') == nuc_id) & (pl.col('subid') == subid))
              .collect())
        
        if len(df) == 0:
            raise ValueError(f"No data found for id={nuc_id}, subid={subid}")
        
        result = {}
        for row in df.iter_rows(named=True):
            result[row['replicate']] = {
                'tau_time': row['tau_time'],
                'cs_total': row['cs_total'],
                'bprot': row['bprot'],
                'detached_total': row['detached_total']
            }
        return result


def list_nucleosomes(parquet_path: str) -> List[Tuple[int, int]]:
    """
    Get list of all (id, subid) pairs in the data using lazy loading.
    
    Args:
        parquet_path: Path to the Parquet file
    
    Returns:
        List of (id, subid) tuples
    """
    df = (pl.scan_parquet(parquet_path)
          .select(['id', 'subid'])
          .unique()
          .collect())
    
    return [(row['id'], row['subid']) for row in df.iter_rows(named=True)]


def get_average_trajectory(parquet_path: str, nuc_id: int, subid: int) -> Dict:
    """
    Get averaged trajectory across all replicates using lazy loading.
    
    Args:
        parquet_path: Path to the Parquet file
        nuc_id: Nucleosome ID
        subid: Nucleosome subid
    
    Returns:
        Dictionary with mean and std for each observable
    """
    replicates = get_trajectory(parquet_path, nuc_id, subid)
    
    if len(replicates) == 0:
        raise ValueError(f"No replicates found for {nuc_id}, {subid}")
    
    # Use first replicate's time points as reference
    first_rep = replicates[list(replicates.keys())[0]]
    tau_time = first_rep['tau_time']
    
    # Collect data from all replicates (interpolate if needed)
    cs_totals = []
    bprots = []
    detached_totals = []
    
    for rep_data in replicates.values():
        if len(rep_data['tau_time']) != len(tau_time):
            # Interpolate to common grid
            cs_totals.append(np.interp(tau_time, rep_data['tau_time'], rep_data['cs_total']))
            bprots.append(np.interp(tau_time, rep_data['tau_time'], rep_data['bprot']))
            detached_totals.append(np.interp(tau_time, rep_data['tau_time'], rep_data['detached_total']))
        else:
            cs_totals.append(rep_data['cs_total'])
            bprots.append(rep_data['bprot'])
            detached_totals.append(rep_data['detached_total'])
    
    cs_totals = np.array(cs_totals)
    bprots = np.array(bprots)
    detached_totals = np.array(detached_totals)
    
    return {
        'tau_time': tau_time,
        'cs_total_mean': np.mean(cs_totals, axis=0),
        'cs_total_std': np.std(cs_totals, axis=0),
        'bprot_mean': np.mean(bprots, axis=0),
        'bprot_std': np.std(bprots, axis=0),
        'detached_total_mean': np.mean(detached_totals, axis=0),
        'detached_total_std': np.std(detached_totals, axis=0),
        'n_replicates': len(replicates)
    }


def process_trajectories_in_batches(parquet_path: str, batch_size: int = 100, 
                                    operation=None) -> None:
    """
    Process trajectories in batches to avoid loading everything into memory.
    
    Example use case: Compute statistics for all nucleosomes without loading all data.
    
    Args:
        parquet_path: Path to the Parquet file
        batch_size: Number of nucleosomes to process at once
        operation: Function to apply to each batch. 
                  Takes (nuc_id, subid, trajectory_data) as arguments.
    
    Example:
        def compute_final_state(nuc_id, subid, traj_data):
            print(f"Processing {nuc_id}, {subid}: final cs = {traj_data['cs_total'][-1]}")
        
        process_trajectories_in_batches("traj.parquet", operation=compute_final_state)
    """
    # Get all unique nucleosomes
    nucleosomes = list_nucleosomes(parquet_path)
    
    # Process in batches
    for i in range(0, len(nucleosomes), batch_size):
        batch = nucleosomes[i:i + batch_size]
        
        for nuc_id, subid in batch:
            # Load only this nucleosome's data
            traj_data = get_trajectory(parquet_path, nuc_id, subid)
            
            if operation:
                operation(nuc_id, subid, traj_data)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trajectory_loader.py <parquet_file>")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    
    # Show available nucleosomes (lazy - only loads id/subid columns)
    print("Listing nucleosomes (lazy loading)...")
    nucleosomes = list_nucleosomes(parquet_path)
    print(f"\nFound {len(nucleosomes)} nucleosomes")
    print(f"First 10: {nucleosomes[:10]}")
    
    # Example: Get specific trajectory (lazy - only loads specific rows)
    if nucleosomes:
        nuc_id, subid = nucleosomes[0]
        print(f"\nExample: Getting trajectory for id={nuc_id}, subid={subid}")
        
        # Get all replicates
        all_reps = get_trajectory(parquet_path, nuc_id, subid)
        print(f"  Number of replicates: {len(all_reps)}")
        
        # Get specific replicate
        traj = get_trajectory(parquet_path, nuc_id, subid, replicate=0)
        print(f"  Replicate 0 has {len(traj['tau_time'])} time points")
        print(f"  Final cs_total: {traj['cs_total'][-1]}")
        print(f"  Final bprot: {traj['bprot'][-1]}")
        
        # Get average
        avg = get_average_trajectory(parquet_path, nuc_id, subid)
        print(f"\n  Average across {avg['n_replicates']} replicates:")
        print(f"    Final cs_total: {avg['cs_total_mean'][-1]:.2f} ± {avg['cs_total_std'][-1]:.2f}")
        print(f"    Final bprot: {avg['bprot_mean'][-1]:.2f} ± {avg['bprot_std'][-1]:.2f}")
        
    # Example: Process all trajectories without loading everything
    print("\n\nExample: Processing all trajectories in batches...")
    def example_operation(nuc_id, subid, traj_data):
        # Only loads this nucleosome's data
        n_reps = len(traj_data)
        first_rep = traj_data[list(traj_data.keys())[0]]
        print(f"  {nuc_id}/{subid}: {n_reps} reps, {len(first_rep['tau_time'])} time points")
    
    process_trajectories_in_batches(parquet_path, batch_size=10, operation=example_operation)
