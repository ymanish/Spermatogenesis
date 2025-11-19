"""
Simulation Package
==================

Modular package for running Gillespie simulations of nucleosome dynamics.

Main Components:
----------------
- simulator: Simulator creation and configuration
- replicate: Single replicate execution
- batch: Batch processing with parallelization
- trajectory: Trajectory data handling
- io: File I/O operations
- cli: Command-line interface

Main Entry Point:
-----------------
from src.simulation import run_simulation

Author: MY
Date: 2025-11-16
"""

# Import main orchestration functions
from .orchestrator import run_simulation
# Import configuration
from src.config.custom_type import SimulationConfig

# Import key components for direct access
from .simulator import create_simulator, calculate_stride
from .replicate import run_single_replicate
from .batch import run_batch_simulations
from .trajectory import (
    store_trajectory_data,
    convert_trajectory_to_dataframe,
    save_trajectories_to_parquet
)
from .io import merge_output_files

__all__ = [
    # Main functions
    'run_simulation',
    
    # Configuration
    'SimulationConfig',
    
    # Components
    'create_simulator',
    'calculate_stride',
    'run_single_replicate',
    'run_batch_simulations',
    'store_trajectory_data',
    'convert_trajectory_to_dataframe',
    'save_trajectories_to_parquet',
    'merge_output_files',
]

__version__ = '2.0.0'
