#!/bin/bash
################################################################################
# Rebuild Markov Solver Index
################################################################################
# This script rebuilds the CSV index for Markov solver results after
# array jobs complete. Use this when running array jobs with use_index=False
# to avoid index corruption from concurrent writes.
#
# Usage:
#   ./rebuild_markov_index.sh [storage_dir]
#
# Example:
#   ./rebuild_markov_index.sh /group/pol_schiessel/Manish/Spermatogensis/output/markov_parameter_grid
#
# Author: MY
# Date: 2025-12-11
################################################################################

# Default storage directory
DEFAULT_STORAGE_DIR="/group/pol_schiessel/Manish/Spermatogensis/output/markov_parameter_grid"

# Use provided directory or default
STORAGE_DIR="${1:-$DEFAULT_STORAGE_DIR}"

echo "=========================================="
echo "Rebuilding Markov Solver Index"
echo "=========================================="
echo "Storage directory: $STORAGE_DIR"
echo "Started: $(date)"
echo ""

# Check if directory exists
if [ ! -d "$STORAGE_DIR" ]; then
    echo "ERROR: Storage directory not found: $STORAGE_DIR"
    exit 1
fi

# Run the rebuild script
python3 << 'EOF'
import sys
from pathlib import Path

# Add project to path
project_dir = Path(__file__).parent if hasattr(Path(__file__), 'parent') else Path.cwd()
sys.path.insert(0, str(project_dir))

from src.markov_execution.storage import MarkovStorage

# Get storage directory from command line argument
storage_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/markov_parameter_grid")

print(f"Initializing MarkovStorage at: {storage_dir}")
storage = MarkovStorage(base_dir=storage_dir, use_index=True)

print("\nRebuilding index from existing results...")
n_results = storage.rebuild_index()

print(f"\n✓ Index rebuilt successfully!")
print(f"  Total results indexed: {n_results}")
print(f"  Index location: {storage.index_path}")

# Show summary
print("\nIndex summary:")
if n_results > 0:
    import polars as pl
    df = pl.read_csv(storage.index_path)
    print(f"  Unique parameter sets: {df['param_hash'].n_unique()}")
    print(f"  Unique file IDs: {df['file_id'].n_unique()}")
    
    # Group by key parameters
    if 'prot_p_conc' in df.columns:
        print("\nResults by protamine concentration:")
        summary = df.group_by('prot_p_conc').agg(pl.count('file_id').alias('count'))
        print(summary)
else:
    print("  No results found!")

EOF

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Index rebuilt"
else
    echo "ERROR: Index rebuild failed with exit code $EXIT_CODE"
fi
echo "Ended: $(date)"
echo "=========================================="

exit $EXIT_CODE
