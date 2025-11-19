#!/usr/bin/env python3
"""
CLI: RMST-Based Replicate Estimation Launcher
==============================================

Command-line interface for launching RMST-based replicate estimation jobs.
Designed for HPC cluster execution with SLURM job arrays.

Usage:
------
python -m src.analysis.rmst_estimator.cli \
    --dataset bound \
    --prot-p-conc 100.0 \
    --prot-cooperativity 4.5 \
    --n-nucs 50 \
    --n-reps 20 \
    --n-workers 10 \
    --tolerance 0.05 \
    --seed 42

Author: MY
Date: 2025-11-14
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.config.custom_type import PilotConfig
from src.config.path import RESULTS_DIR, HAMNUCRET_DATA_DIR

# Import main API
from . import estimate_replicates_rmst
from .io import create_output_directory, save_run_metadata


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    'bound': {
        'name': 'Bound Promoter (RET)',
        'path': HAMNUCRET_DATA_DIR / 'exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv',
        'short_name': 'RET'
    },
    'unbound': {
        'name': 'Unbound Promoter (EVI)',
        'path': HAMNUCRET_DATA_DIR / 'exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv',
        'short_name': 'EVI'
    }
}


# =============================================================================
# LAUNCHER FUNCTION
# =============================================================================

def launch_rmst_estimation(
    dataset: str,
    k_wrap: float,
    prot_k_unbind: float,
    prot_k_bind: float,
    prot_p_conc: float,
    prot_cooperativity: float,
    n_nucs: int,
    n_reps: int,
    n_workers: int,
    tolerance: float,
    seed: int,
    base_output_dir: Path,
    plot: bool = True,
    tau_max: float = 10000.0,
    tau_steps: int = 1000,
    batch_size: int = 10,
    random_sample: bool = True
):
    """
    Launch RMST-based replicate estimation for a specific configuration.
    
    Args:
        dataset: Dataset type ('bound' or 'unbound')
        k_wrap: Wrapping energy
        prot_k_unbind: Protamine unbinding rate
        prot_k_bind: Protamine binding rate
        prot_p_conc: Protamine concentration (μM)
        prot_cooperativity: Cooperativity parameter (k_B T)
        n_nucs: Number of nucleosomes to sample
        n_reps: Number of replicates per nucleosome
        n_workers: Number of parallel workers
        tolerance: Tolerance parameter ε
        seed: Random seed for reproducibility
        base_output_dir: Base directory for outputs
        plot: Whether to generate plots
        tau_max: Maximum dimensionless time
        tau_steps: Number of time steps
        batch_size: Nucleosomes per batch
        random_sample: Use random sampling
    
    Returns:
        RMSTAnalysis object
    """
    # Validate dataset
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_CONFIGS.keys())}")
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[dataset]
    file_path = dataset_config['path']
    dataset_name = dataset_config['name']
    
    # Verify file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Create output directory
    output_dir = create_output_directory(
        base_output_dir, dataset, k_wrap, prot_p_conc, prot_cooperativity,
        n_nucs, n_reps, seed
    )
    
    print(f"\n{'='*70}")
    print(f"LAUNCHING RMST-BASED REPLICATE ESTIMATION")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Create configuration
    config = PilotConfig(
        k_wrap=k_wrap,
        prot_k_unbind=prot_k_unbind,
        prot_k_bind=prot_k_bind,
        prot_p_conc=prot_p_conc,
        prot_cooperativity=prot_cooperativity,
        n_pilot_nucleosomes=n_nucs,
        n_pilot_replicates=n_reps,
        tau_max=tau_max,
        tau_steps=tau_steps,
        start_idx=0  # Random sampling doesn't use this
    )
    
    # Create condition label
    if prot_p_conc > 0:
        condition_label = f"{dataset_config['short_name']}_prot{prot_p_conc:.0f}_coop{prot_cooperativity:.1f}"
    else:
        condition_label = f"{dataset_config['short_name']}_noprot"
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset,
        'dataset_name': dataset_name,
        'file_path': str(file_path),
        'metric': 'RMST',
        'parameters': {
            'prot_k_unbind': prot_k_unbind,
            'prot_k_bind': prot_k_bind,
            'prot_p_conc': prot_p_conc,
            'prot_cooperativity': prot_cooperativity,
            'k_wrap': k_wrap,
            'binding_sites': config.binding_sites,
            'tau_max': tau_max,
            'tau_steps': tau_steps,
            'n_nucleosomes': n_nucs,
            'n_replicates': n_reps,
            'seed': seed,
            'batch_size': batch_size,
            'random_sample': random_sample
        },
        'analysis': {
            'tolerance': tolerance,
            'n_workers': n_workers
        }
    }
    save_run_metadata(output_dir, metadata)
    
    # Run RMST-based estimation
    analysis = estimate_replicates_rmst(
        file_path=file_path,
        config=config,
        condition_label=condition_label,
        save_path=output_dir,
        plot=plot,
        n_workers=n_workers,
        tolerance=tolerance,
        batch_size=batch_size,
        random_sample=random_sample,
        seed=seed
    )
    
    # Save summary
    import numpy as np
    summary = {
        'dataset': dataset,
        'condition': condition_label,
        'metric': 'RMST',
        'R': float(analysis.R),
        'sigma_within_sq': float(analysis.sigma_within_sq),
        'sigma_between_sq': float(analysis.sigma_between_sq),
        'n_reps_required': int(analysis.n_reps_required) if analysis.n_reps_required and not np.isinf(analysis.n_reps_required) else None,
        'recommendation': analysis.recommended_replicates,
        'mean_rmst': float(analysis.mean_rmst),
        'std_rmst': float(analysis.std_rmst),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"COMPLETED: {condition_label}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return analysis


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Launch RMST-based replicate estimation for nucleosome simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------

# Bound promoter, no protamine
python -m src.analysis.rmst_estimator.cli \\
    --dataset bound --prot-p-conc 0.0 --prot-cooperativity 0.0 \\
    --n-nucs 50 --n-reps 20 --seed 42

# Unbound promoter, with protamine
python -m src.analysis.rmst_estimator.cli \\
    --dataset unbound --prot-p-conc 1000.0 --prot-cooperativity 4.5 \\
    --n-nucs 100 --n-reps 50 --n-workers 20 --tolerance 0.05 --seed 123

# Low protamine (good for RMST - many stable nucleosomes)
python -m src.analysis.rmst_estimator.cli \\
    --dataset bound --prot-p-conc 100.0 --prot-cooperativity 4.5 \\
    --n-nucs 30 --n-reps 30 --n-workers 10 --seed 456
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['bound', 'unbound'],
        help='Dataset type: bound (RET) or unbound (EVI)'
    )
    parser.add_argument(
        '--k-wrap', type=float, default=1.0,
        help='Nucleosome wrapping energy (k_B T) (default: 1.0)'
    )
    
    # Protamine parameters
    parser.add_argument(
        '--prot-k-unbind', type=float, default=89.7,
        help='Protamine unbinding rate (default: 89.7)'
    )
    parser.add_argument(
        '--prot-k-bind', type=float, default=1.0,
        help='Protamine binding rate (default: 1.0)'
    )
    parser.add_argument(
        '--prot-p-conc', type=float, default=0.0,
        help='Protamine concentration (μM) (default: 0.0)'
    )
    parser.add_argument(
        '--prot-cooperativity', type=float, default=0.0,
        help='Protamine cooperativity factor (k_B T) (default: 0.0)'
    )
    
    # Pilot study parameters
    parser.add_argument(
        '--n-nucs', type=int, required=True,
        help='Number of nucleosomes to sample'
    )
    parser.add_argument(
        '--n-reps', type=int, required=True,
        help='Number of replicates per nucleosome'
    )
    parser.add_argument(
        '--seed', type=int, required=True,
        help='Random seed for reproducible nucleosome sampling'
    )
    parser.add_argument(
        '--random-sample', type=bool, default=True,
        help='Use random sampling (reservoir) vs sequential (default: True)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Batch size for parallel processing (default: 10)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--tolerance', type=float, default=0.05,
        help='Tolerance ε for n_rep calculation (default: 0.05)'
    )
    
    # Execution parameters
    parser.add_argument(
        '--n-workers', type=int, default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--plot', action='store_true', default=False,
        help='Enable plotting (generates PNG files, disable on cluster)'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--tau-max', type=float, default=10000.0,
        help='Maximum dimensionless time (default: 10000.0)'
    )
    parser.add_argument(
        '--tau-steps', type=int, default=1000,
        help='Number of time steps for RMST integration (default: 1000)'
    )
    
    # Output directory
    parser.add_argument(
        '--output-dir', type=str,
        default=str(RESULTS_DIR / 'rmst_replicate_estimation'),
        help='Base output directory (default: RESULTS_DIR/rmst_replicate_estimation)'
    )
    
    args = parser.parse_args()
    
    # Import numpy here (after argument parsing for faster --help)
    import numpy as np
    
    # Launch RMST estimation
    analysis = launch_rmst_estimation(
        dataset=args.dataset,
        k_wrap=args.k_wrap,
        prot_k_unbind=args.prot_k_unbind,
        prot_k_bind=args.prot_k_bind,
        prot_p_conc=args.prot_p_conc,
        prot_cooperativity=args.prot_cooperativity,
        n_nucs=args.n_nucs,
        n_reps=args.n_reps,
        n_workers=args.n_workers,
        tolerance=args.tolerance,
        seed=args.seed,
        base_output_dir=Path(args.output_dir),
        plot=args.plot,
        tau_max=args.tau_max,
        tau_steps=args.tau_steps,
        batch_size=args.batch_size,
        random_sample=args.random_sample
    )
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print(f"  R value: {analysis.R:.4f}")
    if analysis.n_reps_required:
        if np.isinf(analysis.n_reps_required):
            print(f"  Required replicates: ∞")
        else:
            print(f"  Required replicates (ε={args.tolerance}): {analysis.n_reps_required}")
    print(f"  Mean RMST: {analysis.mean_rmst:.2f}")
    print(f"  Recommendation: {analysis.recommended_replicates}")


if __name__ == "__main__":
    main()
