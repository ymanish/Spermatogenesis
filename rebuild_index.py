#!/usr/bin/env python3
"""
Rebuild the simulation index after parallel jobs complete.

This script scans all parameter directories and creates/updates the 
simulation_index.csv file based on the parameters.json files.

Best Practice:
  - Run simulations with use_index=False (default) to avoid race conditions
  - Run this script AFTER all parallel jobs complete
  - The index is only for convenience - all data is stored per-directory

Usage:
    # Local (requires environment with pandas)
    python3 rebuild_index.py
    
    # On cluster with singularity
    singularity exec nucleosome.sif python3 rebuild_index.py
    
    # With custom results directory
    python3 rebuild_index.py --results-dir /path/to/results
"""

import argparse
from pathlib import Path
from src.config.storage import SimulationStorage
from src.config.path import RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description="Rebuild simulation index from parameters.json files")
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        default=RESULTS_DIR,
        help=f"Results directory (default: {RESULTS_DIR})"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Rebuilding Simulation Index")
    print("=" * 70)
    print(f"Scanning: {args.results_dir}")
    print()
    
    # Create storage without index initially, then rebuild
    storage = SimulationStorage(base_dir=args.results_dir, use_index=False)
    index = storage.rebuild_index()
    
    print("\n" + "=" * 70)
    print("Index Summary")
    print("=" * 70)
    print(f"Total parameter sets: {len(index)}")
    print(f"Index file: {storage.index_file}")
    
    if len(index) > 0:
        print("\nParameter ranges found:")
        for col in ['k_wrap', 'k_unbind', 'k_bind', 'p_conc', 'cooperativity', 'inf_protamine']:
            if col in index.columns:
                unique_vals = index[col].dropna().unique()
                if len(unique_vals) > 0:
                    print(f"  {col:15s}: {sorted(unique_vals)}")
        
        # Show sample of directories
        print(f"\nParameter directories found:")
        for i, hash_dir in enumerate(index['param_hash'].head(5)):
            print(f"  {hash_dir}")
        if len(index) > 5:
            print(f"  ... and {len(index) - 5} more")
    else:
        print("\n⚠ Warning: No parameter sets found!")
        print("  Make sure your simulations have completed and created parameters.json files.")
    
    print("\n" + "=" * 70)
    print("✓ Index rebuild complete!")
    print("=" * 70)
    print("\nYou can now use the index to query simulations:")
    print("  storage = SimulationStorage(RESULTS_DIR, use_index=True)")
    print("  matches = storage.find_simulations(k_wrap=22.0, inf_protamine=True)")
    print()

if __name__ == "__main__":
    main()
