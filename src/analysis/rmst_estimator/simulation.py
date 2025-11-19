"""
Simulation Execution Functions
===============================

Orchestrates simulation runs for RMST analysis, including parallelization.

Author: MY
Date: 2025-11-14
"""

import time
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from src.core.nucleosomes import Nucleosome
from src.core.build_nucleosomes import nucleosome_generator
from src.config.custom_type import PilotConfig

from .core import compute_rmst_from_survival, extract_survival_curve_from_timeseries
from .sampling import sample_nucleosomes, batcher


def run_single_replicate_with_timeseries(
    nuc: Nucleosome,
    replicate_num: int,
    build_params: Dict,
    tau_points: np.ndarray,
    inf_protamine: bool,
    tau_min: Optional[float]
) -> Tuple[float, float, float, Dict]:
    """
    Run single replicate and return timeseries for RMST calculation.
    
    Args:
        nuc: Nucleosome instance
        replicate_num: Replicate number
        build_params: Dictionary with factory functions
        tau_points: Array of dimensionless time points
        inf_protamine: Whether to use infinite protamine
        tau_min: Minimum tau value for renucleation
    
    Returns:
        Tuple of (final_cs, final_bprot, detach_time, timeseries_dict)
        where timeseries_dict = {'tau': [...], 'cs': [...], 'bprot': [...], 'detached': [...]}
    """
    total_steps = len(tau_points)
    
    # Create simulator
    from src.simulation import create_simulator 
    from src.simulation.replicate import process_simulation_states
    sim = create_simulator(nuc, build_params, tau_points, inf_protamine, tau_min, replicate_num)
    
    # Process with save_trajectories=True to get timeseries
    tau_times, cs_totals, bprots, detached_totals, final_cs, final_bprot, detach_time = \
        process_simulation_states(sim, save_trajectories=True, eff_stride=1, total_steps=total_steps)
    
    timeseries = {
        'tau': tau_times,
        'cs': cs_totals,
        'bprot': bprots,
        'detached': detached_totals
    }
    
    return final_cs, final_bprot, detach_time, timeseries


def _simulate_single_nucleosome_rmst(
    build_params: Dict,
    nuc: Nucleosome,
    config_dict: Dict,
    n_replicates: int,
    tau_grid: np.ndarray
) -> Tuple[str, List[float]]:
    """
    Simulate replicates for one nucleosome and compute RMST for each.
    
    Args:
        build_params: Dictionary with factory functions (nucs_factory, prot_factory)
        nuc: Nucleosome instance
        config_dict: Configuration dictionary with simulation parameters
        n_replicates: Number of replicates to run
        tau_grid: Time grid for RMST integration
    
    Returns:
        Tuple of (nucleosome_key, [rmst_1, rmst_2, ..., rmst_n])
    """
    nuc_key = f"nuc_{nuc.id}_{nuc.subid}"
    rmst_values = []
    
    for rep in range(n_replicates):
        # Run simulation WITH timeseries
        final_cs, final_bprot, detach_time, timeseries = run_single_replicate_with_timeseries(
            nuc=nuc,
            replicate_num=rep,
            build_params=build_params,
            tau_points=tau_grid,
            inf_protamine=config_dict['inf_protamine'],
            tau_min=config_dict.get('tau_min')
        )
        
        # Extract survival curve from timeseries
        survival = extract_survival_curve_from_timeseries(
            timeseries,
            tau_grid,
            n_binding_sites=config_dict['binding_sites']
        )
        
        # Compute RMST
        rmst = compute_rmst_from_survival(survival, tau_grid)
        rmst_values.append(rmst)
    
    return nuc_key, rmst_values


def _simulate_batch_nucleosome(
    build_params: Dict,
    nuc_batch: List[Nucleosome],
    config_dict: Dict,
    n_replicates: int,
    tau_grid: np.ndarray
) -> Dict[str, List[float]]:
    """
    Simulate multiple nucleosomes and compute RMST for each.
    
    This function is designed to be called in parallel by ProcessPoolExecutor.
    
    Args:
        build_params: Dictionary with factory functions
        nuc_batch: List of nucleosome instances to process
        config_dict: Configuration dictionary
        n_replicates: Number of replicates to run per nucleosome
        tau_grid: Time grid for RMST integration
    
    Returns:
        Dict mapping nucleosome_key -> [rmst_1, rmst_2, ..., rmst_R]
    """
    rmst_results = {}
    for nuc in nuc_batch:
        nuc_key, rmst_values = _simulate_single_nucleosome_rmst(
            build_params=build_params,
            nuc=nuc,
            config_dict=config_dict,
            n_replicates=n_replicates,
            tau_grid=tau_grid
        )
        rmst_results[nuc_key] = rmst_values
    
    return rmst_results


def run_rmst_pilot_study(
    file_path: Path,
    config: PilotConfig,
    n_workers: int = 1,
    verbose: bool = True,
    random_sample: bool = True,
    seed: Optional[int] = None,
    batch_size: int = 1
) -> Dict[str, List[float]]:
    """
    Run pilot study computing RMST for each nucleosome × replicate.
    
    This is the main entry point for executing the pilot study with parallelization.
    
    Args:
        file_path: Path to nucleosome data file
        config: PilotConfig object with all simulation parameters
        n_workers: Number of parallel workers (default: 1)
        verbose: Print progress information (default: True)
        random_sample: Use random sampling vs sequential (default: True)
        seed: Random seed for reproducibility (optional)
        batch_size: Number of nucleosomes per batch for parallel processing (default: 1)
    
    Returns:
        Dict mapping nucleosome_key -> [rmst_1, rmst_2, ..., rmst_R]
    
    Examples:
        >>> config = PilotConfig(
        ...     k_wrap=1.0,
        ...     prot_k_unbind=89.7,
        ...     prot_k_bind=1.0,
        ...     prot_p_conc=100.0,
        ...     prot_cooperativity=4.5,
        ...     n_pilot_nucleosomes=50,
        ...     n_pilot_replicates=20,
        ...     tau_max=10000.0,
        ...     tau_steps=1000
        ... )
        >>> results = run_rmst_pilot_study(
        ...     file_path="data.tsv",
        ...     config=config,
        ...     n_workers=10,
        ...     random_sample=True,
        ...     seed=42
        ... )
        >>> print(f"Analyzed {len(results)} nucleosomes")
    
    Notes:
        - Uses ProcessPoolExecutor for true parallelism
        - Memory-efficient: never loads full dataset
        - Progress bar shows completion status
        - Returns dict can be passed directly to analyze_rmst_replicates()
    """
    # Create tau grid for RMST integration
    tau_grid = np.linspace(0, config.tau_max, config.tau_steps + 1)
    
    # Load nucleosomes using generator
    gen = nucleosome_generator(file_path, k_wrap=config.k_wrap)
    
    # Sample nucleosomes (MEMORY-EFFICIENT)
    nucleosomes = sample_nucleosomes(
        generator=gen,
        n_nucleosomes=config.n_pilot_nucleosomes,
        random_sample=random_sample,
        start_idx=config.start_idx,
        seed=seed,
        verbose=verbose
    )
    
    # Create batches for parallel processing
    nuc_batches = list(batcher(nucleosomes, size=batch_size))
    
    # Prepare config dict (must be picklable for multiprocessing)
    config_dict = {
        'k_wrap': config.k_wrap,
        'binding_sites': config.binding_sites,
        'tau_max': config.tau_max,
        'tau_steps': config.tau_steps,
        'renucleation': config.renucleation,
        'prot_params': config.prot_params,
        'inf_protamine': config.inf_protamine
    }
    
    results = {}
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"Running RMST pilot study (workers={n_workers}, batch_size={batch_size})...")
        print(f"{'─'*70}\n")
    
    start_time = time.time()
    
    # Parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batch jobs
        futures = [
            executor.submit(
                _simulate_batch_nucleosome,
                config.build_params,
                nuc_batch,
                config_dict,
                config.n_pilot_replicates,
                tau_grid
            )
            for nuc_batch in nuc_batches
        ]
        
        # Collect results with progress bar
        if verbose:
            iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(nuc_batches),
                desc="Computing RMST",
                ncols=70
            )
        else:
            iterator = concurrent.futures.as_completed(futures)
        
        for future in iterator:
            rmst_results = future.result()
            results.update(rmst_results)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"✓ RMST pilot study complete in {elapsed:.1f}s")
        print(f"  Average time per nucleosome: {elapsed/len(nucleosomes):.1f}s")
        if n_workers > 1:
            sequential_estimate = elapsed * n_workers
            print(f"  Estimated speedup: ~{sequential_estimate/elapsed:.1f}x")
        print()
    
    return results
