#!/usr/bin/env python3
"""
CLI: Simulation Launcher
========================

Command-line interface for running Gillespie simulations.

Usage:
------
python -m src.simulation.cli \
    --infile data/nucleosomes.tsv \
    --storage_dir output/simulations \
    --k_wrap 1.0 \
    --prot_p_conc 100.0 \
    --prot_cooperativity 4.5 \
    --replicates 20 \
    --n_workers 10

Author: MY
Date: 2025-11-16
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import argparse
import time
import datetime as dt
import numpy as np
from pathlib import Path
from src.utils.logger_util import get_logger
from src.config.storage import SimulationStorage
from src.config.path import RESULTS_DIR
from src.config.custom_type import SimulationConfig

# Import main function
from .orchestrator import run_simulation


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run nucleosome simulations with configurable parameters."
    )
    
    # Input/output arguments
    parser.add_argument(
        "--infile", type=Path,
        help="Path to the input TSV file with nucleosome data."
    )
    parser.add_argument(
        "--storage_dir", type=Path,
        help="Directory to store simulation results."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset identifier (e.g., 'bound', 'unbound') to prefix output files and prevent overwrites."
    )
    
    # Execution parameters
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Number of sequences per batch (default: 50)."
    )
    parser.add_argument(
        "--n_workers", type=int, default=20,
        help="Number of parallel workers (default: 20)."
    )
    parser.add_argument(
        "--flush_every", type=int, default=10000,
        help="Number of rows to flush to disk per batch (default: 10000)."
    )
    
    # Simulation parameters
    parser.add_argument(
        "--k_wrap", type=float, default=1.0,
        help="Nucleosome wrapping constant (default: 1.0)."
    )
    parser.add_argument(
        "--binding_sites", type=int, default=14,
        help="Number of binding sites (default: 14)."
    )
    parser.add_argument(
        "--inf_protamine", action="store_true",
        help="Enable infinite protamine (default: False)."
    )
    
    # Protamine parameters
    parser.add_argument(
        "--prot_k_unbind", type=float, default=0.01,
        help="Protamine unbinding rate (default: 0.01)."
    )
    parser.add_argument(
        "--prot_k_bind", type=float, default=10.0,
        help="Protamine binding rate (default: 10.0)."
    )
    parser.add_argument(
        "--prot_p_conc", type=float, default=0.0,
        help="Protamine concentration (default: 0.0)."
    )
    parser.add_argument(
        "--prot_cooperativity", type=float, default=0.0,
        help="Protamine cooperativity factor (default: 0.0)."
    )
    parser.add_argument(
        "--replicates", type=int, default=20,
        help="Replicates per nucleosome (default: 20)."
    )
    
    # Time points configuration
    parser.add_argument(
        "--t_stop", type=float, default=10.0,
        help="Simulation end time in physical units (default: 10.0)."
    )
    parser.add_argument(
        "--t_num", type=int, default=1000,
        help="Number of time points in physical units (default: 1000)."
    )
    parser.add_argument(
        "--tau_stop", type=float, default=None,
        help="Dimensionless end time tau_max (overrides --t_stop if set)."
    )
    parser.add_argument(
        "--tau_num", type=int, default=None,
        help="Number of tau-sample points (defaults to --t_num if not set)."
    )
    parser.add_argument(
        "--maxpoints_saved_trajectories", type=int, default=100,
        help="Maximum trajectory datapoints to save (default: 100)."
    )
    
    # Trajectory and renucleation
    parser.add_argument(
        "--save_trajectories", action="store_true",
        help="Save trajectory data (default: False)."
    )
    parser.add_argument(
        "--renucleation", action="store_true",
        help="Enable renucleation (default: False)."
    )

    ####Testing Only Arguments####
    parser.add_argument( "--max_nucs", type=int, default=None, help="Maximum number of nucleosomes to process (for testing only)." )
    parser.add_argument( "--subids_start", type=int, default=None, help="Start subID for testing (inclusive)." )
    parser.add_argument( "--subids_end", type=int, default=None, help="End subID for testing (exclusive)." )
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level='INFO')
    
    # Setup temporary directory
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    logger.info(f"Using temporary directory: {tmp_dir}")
    
    # Parse arguments
    args = parse_args()
    
    # Validate input file
    if not args.infile:
        raise ValueError("--infile is required")
    if not args.infile.exists():
        raise FileNotFoundError(f"Input file {args.infile} does not exist.")
    
    # Setup storage directory
    if not args.storage_dir:
        raise ValueError("--storage_dir is required")
    args.storage_dir.mkdir(parents=True, exist_ok=True)
    
    ###indexing will maintain a csv file which can be problem in the array jobs. Use index=False and then rebuild the index from the rebuild_index method after all jobs are done.
    storage = SimulationStorage(base_dir=args.storage_dir, use_index=False)

    
    # Calculate tau points
    if args.tau_stop is not None:
        tau_stop = float(args.tau_stop)
        tau_num = int(args.tau_num or args.t_num)
    else:
        # Back-compat: convert t_stop to tau
        tau_stop = float(args.k_wrap) * float(args.t_stop)
        tau_num = int(args.t_num)
    
    tau_points = np.linspace(0.0, tau_stop, tau_num)
    
    # Validate maxpoints_saved_trajectories
    if args.save_trajectories and args.maxpoints_saved_trajectories is not None:
        if args.maxpoints_saved_trajectories > tau_num:
            raise ValueError(
                f"maxpoints_saved_trajectories ({args.maxpoints_saved_trajectories}) "
                f"cannot exceed tau_num ({tau_num}). "
                f"Either increase tau_num or decrease maxpoints_saved_trajectories."
            )
        stride = max(1, int(np.ceil(tau_num / args.maxpoints_saved_trajectories)))
        logger.info(
            f"Trajectory saving: {args.maxpoints_saved_trajectories} points "
            f"out of {tau_num} total (stride: {stride})"
        )
    
    # Setup protamine parameters
    logger.info(
        f"Protamine: p_conc={args.prot_p_conc}, "
        f"k_unbind={args.prot_k_unbind}, k_bind={args.prot_k_bind}"
    )
    
    #### The order of the parameters in this dictionary matters! ####
    #### because it affects the directory naming and hashing. ####
    #### if you want to make the order invariant, modify the code of storage.py ####
    prot_params = {
        'k_unbind': args.prot_k_unbind,
        'k_bind': args.prot_k_bind,
        'p_conc': args.prot_p_conc,
        'cooperativity': args.prot_cooperativity
    }
    
    params = {
        'k_wrap': args.k_wrap,
        'prot_params': prot_params,
        'binding_sites': args.binding_sites,
        'tau_max': tau_stop,
        'tau_steps': tau_num,
        'inf_protamine': args.inf_protamine,
        'replicates': args.replicates
    }
    
    # Get output paths
    file_id = args.infile.stem
    # Add dataset prefix if provided to prevent overwriting
    if args.dataset:
        file_id = f"{args.dataset}_{file_id}"
        logger.info(f"Running simulation for file: {args.infile} with dataset: {args.dataset}, ID: {file_id}")
    else:
        logger.info(f"Running simulation for file: {args.infile} with ID: {file_id}")
    
    output_paths = storage.get_output_paths(params, file_id)
    traj_outfile = output_paths['trajectory']
    tsv_outfile = output_paths['summary']
    
    # Create SimulationConfig from arguments
    config = SimulationConfig(
        # Nucleosome parameters
        k_wrap=args.k_wrap,
        binding_sites=args.binding_sites,
        
        # Protamine parameters
        prot_k_unbind=args.prot_k_unbind,
        prot_k_bind=args.prot_k_bind,
        prot_p_conc=args.prot_p_conc,
        prot_cooperativity=args.prot_cooperativity,
        
        # Simulation time parameters
        tau_max=tau_stop,
        tau_steps=tau_num,
        
        # Simulation behavior
        inf_protamine=args.inf_protamine,
        renucleation=args.renucleation,
        replicates=args.replicates,
        
        # Execution parameters
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        flush_every=args.flush_every,
        
        # Trajectory parameters
        save_trajectories=args.save_trajectories,
        maxpoints_saved_trajectories=args.maxpoints_saved_trajectories
    )
    
    logger.info(f"Configuration: {config}")
    
    # Run simulation with config
    run_simulation(
        file_path=args.infile,
        traj_outfile=traj_outfile,
        tsv_outfile=tsv_outfile,
        config=config,
        logger=logger, 
        max_nucs=args.max_nucs,
        subids_range=(args.subids_start, args.subids_end) if args.subids_start is not None and args.subids_end is not None else None
    )
    
    # Report completion
    end = time.perf_counter()
    logger.info(f"Total execution time: {dt.timedelta(seconds=end - start)}")


if __name__ == "__main__":
    main()
