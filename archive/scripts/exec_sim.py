"""
DEPRECATED: exec_sim.py
=======================

This file is deprecated and maintained only for backward compatibility.
Please use the new modular simulation package instead:

    from src.simulation import run_simulation

Or use the CLI:

    python -m src.simulation.cli --help

Author: MY
Date: 2025-11-16
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import warnings
import time
import datetime as dt
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Import from new modular package
from src.simulation import run_simulation
from src.utils.logger_util import get_logger
from src.config.storage import SimulationStorage
from src.config.path import RESULTS_DIR

# Show deprecation warning
warnings.warn(
    "exec_sim.py is deprecated. Use 'from src.simulation import run_simulation' "
    "or 'python -m src.simulation.cli' instead.",
    DeprecationWarning,
    stacklevel=2
)


def main(file_path: Path, traj_outfile: Path, tsv_outfile: Path,
        k_wrap: float, prot_params: dict, replicates: int = 20, binding_sites: int = 14,
        batch_size: int = 10, n_workers: int = 4, flush_every: int = 10000,
            tau_points: np.ndarray = None, 
            inf_protamine: bool = True, 
            save_trajectories: bool = False,
            renucleation: bool = False, 
            maxpoints_saved_trajectories: int = 100,
            logger: Optional[logging.Logger] = None) -> None:
    """
    DEPRECATED: Use run_simulation from src.simulation instead.
    
    This function is maintained for backward compatibility only.
    """
    warnings.warn(
        "main() in exec_sim.py is deprecated. Use run_simulation() from src.simulation",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Call the new modular function
    run_simulation(
        file_path=file_path,
        traj_outfile=traj_outfile,
        tsv_outfile=tsv_outfile,
        k_wrap=k_wrap,
        prot_params=prot_params,
        replicates=replicates,
        binding_sites=binding_sites,
        batch_size=batch_size,
        n_workers=n_workers,
        flush_every=flush_every,
        tau_points=tau_points,
        inf_protamine=inf_protamine,
        save_trajectories=save_trajectories,
        renucleation=renucleation,
        maxpoints_saved_trajectories=maxpoints_saved_trajectories,
        logger=logger
    )


def arg_parser():
    import argparse 
    parser = argparse.ArgumentParser(description="Run nucleosome simulations with configurable parameters.")
    
    # Input/output arguments
    parser.add_argument("--infile", type=Path, help="Path to the input FASTA file.")
    parser.add_argument("--storage_dir", type=Path, help="Directory to store simulation results.")
    
    # Execution parameters
    parser.add_argument("--batch_size", type=int, default=50, help="Number of sequences per batch.")
    parser.add_argument("--n_workers", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--flush_every", type=int, default=10000, help="Number of rows to flush to disk per batch.")
    
    # Simulation parameters (with sensible defaults)
    parser.add_argument("--k_wrap", type=float, default=1.0, help="Nucleosome wrapping constant.")
    parser.add_argument("--binding_sites", type=int, default=14, help="Number of binding sites.")
    parser.add_argument("--inf_protamine", action="store_true", help="Enable infinite protamine (default: False).")
    
    # Protamine parameters (as individual arguments)
    parser.add_argument("--prot_k_unbind", type=float, default=0.01, help="Protamine unbinding rate.")
    parser.add_argument("--prot_k_bind", type=float, default=10.0, help="Protamine binding rate.")
    parser.add_argument("--prot_p_conc", type=float, default=0.0, help="Protamine concentration.")
    parser.add_argument("--prot_cooperativity", type=float, default=0.0, help="Protamine cooperativity factor.")
    parser.add_argument("--replicates", type=int, default=20, help="Replicates per nucleosome")

    
    # Time points configuration
    parser.add_argument("--t_stop", type=float, default=10.0, help="Simulation end time in physical units (e.g seconds).")
    parser.add_argument("--t_num", type=int, default=1000, help="Number of time points to sample in physical units. " \
                                                                "Evaluate the state of system at these points in simulator.")
    parser.add_argument("--tau_stop", type=float, default=None, help="Dimensionless end time tau_max (overrides --t_stop if set).")
    parser.add_argument("--tau_num", type=int, default=None, help="Number of tau-sample points (defaults to --t_num if not set)." \
                                                                  "Evaluate the state of system at these points in simulator.")
    parser.add_argument("--maxpoints_saved_trajectories", type=int, default=100, help="Maximum number of trajectory datapoints to save out of the total datapoints." \
                                                                                        "For example, if tau_num=1000 and maxpoints_saved_trajectories=100, " \
                                                                                        "then every 10th point will be saved as trajectory data.")

    parser.add_argument("--save_trajectories", action="store_true", help="Save trajectory data (default: False).")
    parser.add_argument("--renucleation", action="store_true", help="Enable renucleation (default: False).")

    return parser.parse_args()

if __name__ == "__main__":

    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level='INFO')


    from src.config.path import RESULTS_DIR
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    logger.info(f"Using temporary directory: {tmp_dir}")


    # TSV_INFILE = Path("/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/boundprom/breath_energy/001.tsv") 
    # STORAGE_DIR = RESULTS_DIR /"boundprom/GSim"

    args = arg_parser()
    if args.infile:
        TSV_INFILE = args.infile
    if args.storage_dir:
        STORAGE_DIR = args.storage_dir


    if not TSV_INFILE.exists():
        raise FileNotFoundError(f"FASTA file {TSV_INFILE} does not exist. Please check the path.")

    if not STORAGE_DIR.exists():
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)


    storage = SimulationStorage(base_dir=STORAGE_DIR)
    # t_points = np.linspace(0, t_stop, t_num)

    #### Decide sampling grid in tau (dimensionless)
    if args.tau_stop is not None:
        tau_stop = float(args.tau_stop)
        tau_num = int(args.tau_num or args.t_num)
    else:
        ### Back-compat: user provided t_stop; convert to tau via tau = k_wrap * t
        tau_stop = float(args.k_wrap) * float(args.t_stop)
        tau_num = int(args.t_num)

    tau_points = np.linspace(0.0, tau_stop, tau_num) ### dimensionless time points
    ### Convert back to physical time points via t = tau / k_wrap
    
    # Validate maxpoints_saved_trajectories
    if args.save_trajectories and args.maxpoints_saved_trajectories is not None:
        if args.maxpoints_saved_trajectories > tau_num:
            raise ValueError(
                f"maxpoints_saved_trajectories ({args.maxpoints_saved_trajectories}) cannot be greater than "
                f"tau_num ({tau_num}). maxpoints_saved_trajectories determines how many points to save "
                f"from the total tau_num points. Either increase tau_num or decrease maxpoints_saved_trajectories."
            )
        logger.info(f"Trajectory saving: will save {args.maxpoints_saved_trajectories} points out of {tau_num} total points "
                   f"(stride: {max(1, int(np.ceil(tau_num / args.maxpoints_saved_trajectories)))})")

    logger.info(f"prot_p_conc: {args.prot_p_conc}, prot_k_unbind: {args.prot_k_unbind}, prot_k_bind: {args.prot_k_bind}")

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
        'tau_max': args.tau_stop,
        'tau_steps': args.tau_num,
        'inf_protamine': args.inf_protamine
    }


    file_id = TSV_INFILE.stem
    logger.info(f"Running simulation for file: {TSV_INFILE} with ID: {file_id}")
    output_paths = storage.get_output_paths(params, file_id)
    traj_outfile = output_paths['trajectory']
    tsv_outfile = output_paths['summary']

    main(file_path=TSV_INFILE,
         traj_outfile=traj_outfile,
         tsv_outfile=tsv_outfile,
            k_wrap=args.k_wrap,
            replicates=args.replicates,
            prot_params=prot_params,
            binding_sites=args.binding_sites,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            tau_points=tau_points,
            inf_protamine=args.inf_protamine, 
            save_trajectories=args.save_trajectories,
            maxpoints_saved_trajectories=args.maxpoints_saved_trajectories,
            renucleation=args.renucleation,
            flush_every=args.flush_every,
            logger=logger,
    )


    end = time.perf_counter()
    logger.info(f"Total execution time: {dt.timedelta(seconds=end - start)}")
