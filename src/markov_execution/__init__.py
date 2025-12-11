"""
Markov Execution Package
========================

Modular package for running Markov solver calculations on multiple nucleosomes
with parallelization and output management.

Main Components:
----------------
- solver_runner: Individual nucleosome solver execution
- batch: Batch processing with parallelization
- output: Data storage and file I/O
- orchestrator: Main orchestration function
- cli: Command-line interface

Main Entry Point:
-----------------
from src.markov_execution import run_markov_solver

Author: MY
Date: 2025-12-11
"""

# Import main orchestration function
from .orchestrator import run_markov_solver

# Import configuration and storage
from .config import MarkovConfig
from .storage import MarkovStorage

# Import key components for direct access
from .solver_runner import solve_single_nucleosome
from .batch import run_batch_markov
from .output import (
    save_markov_results_to_parquet,
    save_markov_summary_to_tsv,
    merge_markov_output_files
)

__all__ = [
    # Main functions
    'run_markov_solver',
    
    # Configuration and storage
    'MarkovConfig',
    'MarkovStorage',
    
    # Components
    'solve_single_nucleosome',
    'run_batch_markov',
    'save_markov_results_to_parquet',
    'save_markov_summary_to_tsv',
    'merge_markov_output_files',
]

__version__ = '1.0.0'
