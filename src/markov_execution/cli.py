"""
Command-Line Interface Module
==============================

CLI for running Markov solver from command line.

Usage:
------
python -m src.markov_execution.cli \
    --infile data/nucleosomes.tsv \
    --storage_dir output/markov \
    --k_wrap 1.0 \
    --prot_p_conc 10.0 \
    --prot_cooperativity 0.0 \
    --t_max 1000.0 \
    --n_workers 10

Author: MY
Date: 2025-12-11
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import argparse
import time
import datetime as dt
from pathlib import Path
from src.utils.logger_util import get_logger
from .config import MarkovConfig
from .orchestrator import run_markov_solver
from .storage import MarkovStorage


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Markov solver for nucleosome unwrapping dynamics"
    )
    
    # Input/Output
    parser.add_argument(
        '--infile',
        type=Path,
        required=True,
        help='Path to input TSV file with nucleosome data'
    )
    parser.add_argument(
        '--storage_dir',
        type=Path,
        required=True,
        help='Storage directory for results (organized by parameters)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset identifier (e.g., "bound", "unbound") to prefix output files'
    )
    
    # Nucleosome parameters
    parser.add_argument(
        '--k_wrap',
        type=float,
        default=1.0,
        help='Wrapping rate constant (s^-1) (default: 1.0)'
    )
    parser.add_argument(
        '--binding_sites',
        type=int,
        default=14,
        help='Number of DNA-histone binding sites (default: 14)'
    )

    # Protamine parameters
    parser.add_argument(
        '--prot_k_bind',
        type=float,
        default=1.0,
        help='Protamine binding rate ((μM·s)^-1) (default: 1.0)'
    )
    parser.add_argument(
        '--prot_k_unbind',
        type=float,
        default=89.7,
        help='Protamine unbinding rate (s^-1) (default: 89.7)'
    )
    parser.add_argument(
        '--prot_p_conc',
        type=float,
        default=0.0,
        help='Protamine concentration (μM) (default: 0.0)'
    )
    parser.add_argument(
        '--prot_cooperativity',
        type=float,
        default=0.0,
        help='Cooperativity parameter J (k_B T) (default: 0.0)'
    )
    
    # Computation parameters
    parser.add_argument(
        '--tau_max',
        type=float,
        default=1000.0,
        help='Maximum dimensionless time τ (default: 1000.0)'
    )
    parser.add_argument(
        '--tau_steps',
        type=int,
        default=500,
        help='Number of time points (default: 500)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['expm', 'ode'],
        default='expm',
        help='Solver method (default: expm)'
    )
    parser.add_argument(
        '--sparse',
        action='store_true',
        help='Use sparse matrices (default: False)'
    )
    
    # Execution parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Number of nucleosomes per batch (default: 10)'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    parser.add_argument(
        '--max_nucs',
        type=int,
        default=None,
        help='Maximum number of nucleosomes to process (default: None, process all)'
    )
    
    # Output options
    parser.add_argument(
        '--save_survival',
        action='store_true',
        default=True,
        help='Save survival functions (default: True)'
    )
    parser.add_argument(
        '--save_states',
        action='store_true',
        help='Save state probabilities (default: False)'
    )
    parser.add_argument(
        '--save_mfpt',
        action='store_true',
        default=True,
        help='Save MFPT values (default: True)'
    )
    
    # Testing arguments
    parser.add_argument(
        '--subids_start',
        type=int,
        default=None,
        help='Start subID for testing (inclusive)'
    )
    parser.add_argument(
        '--subids_end',
        type=int,
        default=None,
        help='End subID for testing (exclusive)'
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
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
    if not args.infile.exists():
        raise FileNotFoundError(f"Input file {args.infile} does not exist.")
    
    # Setup storage directory
    args.storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Indexing will maintain a CSV file which can be problem in array jobs.
    # Use index=False and then rebuild the index from the rebuild_index method after all jobs are done.
    storage = MarkovStorage(base_dir=args.storage_dir, use_index=False)
    
    # Create configuration
    config = MarkovConfig(
        k_wrap=args.k_wrap,
        binding_sites=args.binding_sites,
        prot_k_bind=args.prot_k_bind,
        prot_k_unbind=args.prot_k_unbind,
        prot_p_conc=args.prot_p_conc,
        prot_cooperativity=args.prot_cooperativity,
        tau_max=args.tau_max,
        tau_steps=args.tau_steps,
        method=args.method,
        sparse=args.sparse,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        max_nucs=args.max_nucs,
        save_survival=args.save_survival,
        save_states=args.save_states,
        save_mfpt=args.save_mfpt
    )
    
    # Prepare parameters for storage (order matters for hashing!)
    prot_params = {
        'k_unbind': args.prot_k_unbind,
        'k_bind': args.prot_k_bind,
        'p_conc': args.prot_p_conc,
        'cooperativity': args.prot_cooperativity
    }
    
    params = {
        'k_wrap': config.k_wrap,
        'prot_params': prot_params,
        'binding_sites': config.binding_sites,
        'tau_max': config.tau_max,
        'tau_steps': config.tau_steps,
        'method': config.method,
        'sparse': config.sparse,
        'dimensionless': config.dimensionless,
    }
    
    # Get output paths
    file_id = args.infile.stem
    if args.dataset:
        file_id = f"{args.dataset}_{file_id}"
        logger.info(f"Running Markov solver for file: {args.infile} with dataset: {args.dataset}, ID: {file_id}")
    else:
        logger.info(f"Running Markov solver for file: {args.infile} with ID: {file_id}")
    
    output_paths = storage.get_output_paths(params, file_id)
    tsv_outfile = output_paths['summary']
    survival_outfile = output_paths['survivals']
    
    # Save configuration text to the parameter directory
   
    logger.info(f"Configuration: {config}")
    
    # Run Markov solver
    run_markov_solver(
        file_path=args.infile,
        tsv_outfile=tsv_outfile,
        survival_outfile=survival_outfile,
        config=config,
        logger=logger,
        max_nucs=args.max_nucs,
        subids_range=(args.subids_start, args.subids_end) if args.subids_start is not None and args.subids_end is not None else None
    )
    
    # Report completion
    end = time.perf_counter()
    logger.info(f"Total execution time: {dt.timedelta(seconds=end - start)}")


if __name__ == '__main__':
    main()
