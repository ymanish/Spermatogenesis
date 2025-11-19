"""
Examples of using Polars for lazy loading and efficient processing of trajectory data.

The key advantage of Parquet + Polars:
- Lazy loading: Only load data you need
- Filtering: Filter before loading (very fast!)
- Parallel operations: Polars uses all CPU cores
- Memory efficient: Process data in chunks
"""

import polars as pl
import numpy as np
from pathlib import Path


# ============================================================================
# Example 1: List all nucleosomes WITHOUT loading trajectory data
# ============================================================================
def example_list_nucleosomes(parquet_path: str):
    """Only loads the id and subid columns - very fast and memory efficient!"""
    print("Example 1: List all nucleosomes (lazy loading)")
    print("=" * 70)
    
    # Lazy scan - doesn't load data yet
    df = (pl.scan_parquet(parquet_path)
          .select(['id', 'subid'])  # Only select these columns
          .unique()                  # Get unique combinations
          .collect())                # Now load into memory
    
    print(f"Found {len(df)} unique nucleosomes")
    print(df.head(10))
    print()


# ============================================================================
# Example 2: Filter specific nucleosomes BEFORE loading
# ============================================================================
def example_filter_before_loading(parquet_path: str, target_ids: list):
    """Only load data for specific nucleosomes - very memory efficient!"""
    print("Example 2: Filter before loading (only load what you need)")
    print("=" * 70)
    
    # Lazy scan with filter - only loads matching rows!
    df = (pl.scan_parquet(parquet_path)
          .filter(pl.col('id').is_in(target_ids))  # Filter in Parquet file!
          .collect())
    
    print(f"Loaded {len(df)} trajectories (only for IDs: {target_ids})")
    print(f"Memory usage: {df.estimated_size() / 1024**2:.2f} MB")
    print(df.head())
    print()


# ============================================================================
# Example 3: Compute statistics without loading all data
# ============================================================================
def example_compute_stats_lazy(parquet_path: str):
    """Compute final state statistics without loading trajectory arrays!"""
    print("Example 3: Compute statistics lazily")
    print("=" * 70)
    
    # Use Polars expressions to compute on list columns
    stats = (pl.scan_parquet(parquet_path)
             .with_columns([
                 # Extract last element from each list (final state)
                 pl.col('cs_total').list.last().alias('final_cs'),
                 pl.col('bprot').list.last().alias('final_bprot'),
                 # Find first non-zero detachment
                 pl.col('detached_total').list.eval(
                     pl.element().filter(pl.element() > 0).first()
                 ).list.first().alias('detached')
             ])
             .select(['id', 'subid', 'replicate', 'final_cs', 'final_bprot', 'detached'])
             .collect())
    
    print(stats.head(10))
    print()
    print("Summary statistics:")
    print(stats.describe())
    print()


# ============================================================================
# Example 4: Group by nucleosome and compute averages
# ============================================================================
def example_group_and_aggregate(parquet_path: str):
    """Compute average across replicates for each nucleosome."""
    print("Example 4: Group by nucleosome and aggregate")
    print("=" * 70)
    
    summary = (pl.scan_parquet(parquet_path)
               .with_columns([
                   pl.col('cs_total').list.last().alias('final_cs'),
                   pl.col('bprot').list.last().alias('final_bprot'),
               ])
               .group_by(['id', 'subid'])
               .agg([
                   pl.count().alias('n_replicates'),
                   pl.col('final_cs').mean().alias('avg_final_cs'),
                   pl.col('final_cs').std().alias('std_final_cs'),
                   pl.col('final_bprot').mean().alias('avg_final_bprot'),
                   pl.col('final_bprot').std().alias('std_final_bprot'),
               ])
               .collect())
    
    print(summary.head(10))
    print()


# ============================================================================
# Example 5: Process in batches for very large files
# ============================================================================
def example_process_in_batches(parquet_path: str, batch_size: int = 100):
    """Process trajectories in batches to control memory usage."""
    print("Example 5: Process in batches")
    print("=" * 70)
    
    # Get all unique nucleosomes
    nucleosomes = (pl.scan_parquet(parquet_path)
                   .select(['id', 'subid'])
                   .unique()
                   .collect())
    
    print(f"Processing {len(nucleosomes)} nucleosomes in batches of {batch_size}")
    
    results = []
    for i in range(0, len(nucleosomes), batch_size):
        batch = nucleosomes[i:i + batch_size]
        
        # Get IDs and subids for this batch
        ids = batch['id'].to_list()
        subids = batch['subid'].to_list()
        
        # Load only this batch
        batch_data = (pl.scan_parquet(parquet_path)
                     .filter(pl.col('id').is_in(ids))
                     .collect())
        
        print(f"  Batch {i//batch_size + 1}: Loaded {len(batch_data)} trajectories, "
              f"{batch_data.estimated_size() / 1024**2:.2f} MB")
        
        # Process this batch (example: extract final states)
        # ... your processing here ...
        
        # Clear memory
        del batch_data
    
    print()


# ============================================================================
# Example 6: Find specific trajectories matching criteria
# ============================================================================
def example_find_matching_trajectories(parquet_path: str):
    """Find trajectories where nucleosomes fully detached."""
    print("Example 6: Find trajectories matching criteria")
    print("=" * 70)
    
    # Find trajectories where final detachment > 0
    detached = (pl.scan_parquet(parquet_path)
                .filter(pl.col('detached_total').list.last() > 0)
                .select(['id', 'subid', 'replicate'])
                .collect())
    
    print(f"Found {len(detached)} trajectories with complete detachment")
    print(detached.head(10))
    print()


# ============================================================================
# Example 7: Advanced: Compute time to 50% chromatin state loss
# ============================================================================
def example_compute_time_to_half(parquet_path: str, nuc_id: int, subid: int):
    """Compute time when chromatin state drops to 50% of initial."""
    print("Example 7: Compute time to 50% CS loss")
    print("=" * 70)
    
    # Load specific nucleosome
    df = (pl.scan_parquet(parquet_path)
          .filter((pl.col('id') == nuc_id) & (pl.col('subid') == subid))
          .collect())
    
    for row in df.iter_rows(named=True):
        cs_total = np.array(row['cs_total'])
        tau_time = np.array(row['tau_time'])
        
        initial_cs = cs_total[0]
        half_cs = initial_cs / 2
        
        # Find when it crosses 50%
        idx = np.where(cs_total <= half_cs)[0]
        if len(idx) > 0:
            time_to_half = tau_time[idx[0]]
            print(f"  Replicate {row['replicate']}: t_half = {time_to_half:.2f}")
        else:
            print(f"  Replicate {row['replicate']}: Never reached 50%")
    
    print()


# ============================================================================
# Example 8: Export subset to CSV for external analysis
# ============================================================================
def example_export_subset(parquet_path: str, output_csv: str, target_ids: list):
    """Export only specific nucleosomes to CSV."""
    print("Example 8: Export filtered subset")
    print("=" * 70)
    
    # Filter and export
    (pl.scan_parquet(parquet_path)
     .filter(pl.col('id').is_in(target_ids))
     .with_columns([
         pl.col('cs_total').list.last().alias('final_cs'),
         pl.col('bprot').list.last().alias('final_bprot'),
     ])
     .select(['id', 'subid', 'replicate', 'final_cs', 'final_bprot'])
     .sink_csv(output_csv))
    
    print(f"Exported filtered data to {output_csv}")
    print()


# ============================================================================
# Example 9: Parallel processing with Polars
# ============================================================================
def example_parallel_operations(parquet_path: str):
    """Polars automatically uses all CPU cores for operations."""
    print("Example 9: Parallel operations")
    print("=" * 70)
    
    import time
    
    start = time.time()
    
    # This runs in parallel automatically!
    result = (pl.scan_parquet(parquet_path)
              .with_columns([
                  # These operations run in parallel across rows
                  pl.col('cs_total').list.mean().alias('mean_cs'),
                  pl.col('bprot').list.mean().alias('mean_bprot'),
                  pl.col('cs_total').list.std().alias('std_cs'),
                  pl.col('bprot').list.std().alias('std_bprot'),
              ])
              .collect())
    
    elapsed = time.time() - start
    
    print(f"Processed {len(result)} trajectories in {elapsed:.2f} seconds")
    print(f"Using {pl.thread_pool_size()} threads")
    print()


# ============================================================================
# Example 10: Memory-efficient histogram computation
# ============================================================================
def example_compute_histogram(parquet_path: str):
    """Compute histogram of final states without loading all data."""
    print("Example 10: Compute histogram lazily")
    print("=" * 70)
    
    # Extract final cs_total values
    final_cs = (pl.scan_parquet(parquet_path)
                .select(pl.col('cs_total').list.last().alias('final_cs'))
                .collect()
                ['final_cs'])
    
    # Compute histogram
    hist = np.histogram(final_cs, bins=15)
    
    print("Histogram of final chromatin states:")
    for i, (count, edge) in enumerate(zip(hist[0], hist[1])):
        print(f"  [{edge:.1f}, {hist[1][i+1]:.1f}): {count}")
    
    print()


# ============================================================================
# Main: Run all examples
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python polars_examples.py <parquet_file>")
        print()
        print("This script demonstrates efficient Polars operations for trajectory analysis.")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    
    # Run examples
    example_list_nucleosomes(parquet_path)
    example_filter_before_loading(parquet_path, target_ids=[1234, 1235, 1236])
    example_compute_stats_lazy(parquet_path)
    example_group_and_aggregate(parquet_path)
    example_find_matching_trajectories(parquet_path)
    example_parallel_operations(parquet_path)
    example_compute_histogram(parquet_path)
    
    # Examples that need specific parameters
    nucleosomes = (pl.scan_parquet(parquet_path)
                   .select(['id', 'subid'])
                   .unique()
                   .collect())
    
    if len(nucleosomes) > 0:
        nuc_id = nucleosomes[0, 'id']
        subid = nucleosomes[0, 'subid']
        example_compute_time_to_half(parquet_path, nuc_id, subid)
    
    print("=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Use .scan_parquet() for lazy loading (doesn't load into memory)")
    print("2. Filter BEFORE .collect() to load only what you need")
    print("3. Use list operations (.list.last(), .list.mean(), etc.)")
    print("4. Process in batches for very large files")
    print("5. Polars automatically uses all CPU cores")
    print("6. Much more memory efficient than loading everything with pickle!")
