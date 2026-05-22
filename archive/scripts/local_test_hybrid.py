"""
Local Testing Script for Hybrid Rejection Simulator
===================================================

Script for comparing Gillespie vs Hybrid Rejection simulators using the new
modular simulation package. This script provides:

1. Single simulator runs (Gillespie or Hybrid Rejection)
2. Side-by-side comparison of both methods
3. Performance metrics and acceptance statistics
4. Trajectory plotting and analysis

The Hybrid Rejection algorithm uses:
- Outer loop: SSA for slow nucleosome reactions (wrap/unwrap)
- Inner loop: Tau-leaping for fast protamine dynamics
- Rejection sampling: Rewrapping rejected if protamine is bound at site

QUICK START:
-----------
1. Uncomment an example in __main__
2. Run: python src/scripts/local_test_hybrid.py

Author: MY
Last Updated: 2025-11-16
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from pathlib import Path
from typing import Optional, Dict, Literal, List
import polars as pl
import time
import itertools
import concurrent.futures
from functools import partial
from tqdm import tqdm

# Import simulators
from src.core.gillespie_simulator import GillespieSimulator
from src.core.hybrid_rejection_simulator import HybridRejectionSimulator
from src.core.nucleosomes import Nucleosome
from src.core.build_nucleosomes import nucleosome_generator
from src.config.path import SRC_DIR, RESULTS_DIR, HAMNUCRET_DATA_DIR

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

def setup_matplotlib_style():
    """Configure matplotlib for publication-quality plots."""
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'svg.fonttype': 'none',
        'axes.labelpad': 1,
        'axes.titlepad': 2,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'xtick.minor.pad': 2,
        'ytick.minor.pad': 2,
    })


# =============================================================================
# SIMULATOR FACTORIES
# =============================================================================

def create_simulator_instance(
    nuc: Nucleosome,
    tau_points: np.ndarray,
    build_params: dict,
    simulator_type: Literal['gillespie', 'hybrid'],
    inf_protamine: bool = True,
    tau_min: Optional[float] = None,
    seed: Optional[int] = None,
    delta_tau_slow: float = 0.1
) -> GillespieSimulator | HybridRejectionSimulator:
    """
    Create a simulator instance of the specified type.
    
    Args:
        nuc: Nucleosome instance
        tau_points: Array of dimensionless time points
        build_params: Dictionary with 'nucs_factory' and 'prot_factory'
        simulator_type: 'gillespie' or 'hybrid'
        inf_protamine: Whether to use infinite protamine
        tau_min: Minimum tau for renucleation (None to disable)
        seed: Random seed
        delta_tau_slow: Time step for slow reactions (hybrid only)
    
    Returns:
        Simulator instance
    """
    # Create fresh instances
    nucs = build_params['nucs_factory'](nuc)
    prots = build_params['prot_factory']()
    
    if simulator_type == 'gillespie':
        return GillespieSimulator(
            nuc_inst=nucs,
            prot_inst=prots,
            t_points=None,
            max_steps=None,
            inf_protamine=inf_protamine,
            seed=seed,
            tau_min=tau_min,
            tau_points=tau_points
        )
    elif simulator_type == 'hybrid':
        return HybridRejectionSimulator(
            nuc_inst=nucs,
            prot_inst=prots,
            tau_points=tau_points,
            epsilon=delta_tau_slow,
            inf_protamine=inf_protamine,
            seed=seed,
            tau_min=tau_min
        )
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")


def create_build_params(
    k_wrap: float = 1.0,
    k_unwrap_factor: float = 1.0,
    prot_p_conc: float = 100.0,
    prot_cooperativity: float = 0.0,
    prot_k_unbind: float = 100.0,
    prot_k_bind: float = 1.0,
    binding_sites: int = 147
) -> dict:
    """
    Create build parameters for nucleosome and protamine factories.
    
    Args:
        k_wrap: Wrapping rate
        k_unwrap_factor: Unwrapping factor
        prot_p_conc: Protamine concentration (μM)
        prot_cooperativity: Cooperativity parameter
        prot_k_unbind: Unbinding rate
        prot_k_bind: Binding rate
        binding_sites: Number of binding sites
    
    Returns:
        Dictionary with factory functions
    """
    from functools import partial
    from src.core.nucleosomes import Nucleosomes
    from src.core.protamine import protamines
    
    def nucs_factory(nuc):
        return Nucleosomes(
            nucleosomes=[nuc],
            k_wrap=k_wrap,
            kT=1.0,
            binding_sites=binding_sites
        )
    
    def prot_factory():
        return protamines(
            k_unbind=prot_k_unbind,
            k_bind=prot_k_bind,
            p_conc=prot_p_conc,
            cooperativity=prot_cooperativity
        )
    
    return {
        'nucs_factory': nucs_factory,
        'prot_factory': prot_factory
    }


# =============================================================================
# SINGLE SIMULATION RUNNER
# =============================================================================

def run_single_simulation(
    nuc: Nucleosome,
    simulator_type: Literal['gillespie', 'hybrid'],
    tau_max: float = 100.0,
    tau_steps: int = 100,
    replicates: int = 5,
    build_params: Optional[dict] = None,
    delta_tau_slow: float = 0.1,
    seed_offset: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Run simulation for a single nucleosome with specified simulator.
    
    Args:
        nuc: Nucleosome instance
        simulator_type: 'gillespie' or 'hybrid'
        tau_max: Maximum dimensionless time
        tau_steps: Number of time steps
        replicates: Number of replicates
        build_params: Build parameters (uses defaults if None)
        delta_tau_slow: Time step for slow reactions (hybrid only)
        seed_offset: Offset for random seeds
        verbose: Print detailed output
    
    Returns:
        Dictionary with trajectories, statistics, and timing
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running {simulator_type.upper()} Simulation")
        print(f"{'='*70}")
        print(f"Nucleosome: {nuc.id}, subid: {nuc.subid}")
        print(f"Tau range: [0, {tau_max}], Steps: {tau_steps}")
        print(f"Replicates: {replicates}")
        if simulator_type == 'hybrid':
            print(f"Delta tau slow: {delta_tau_slow}")
        print(f"{'='*70}\n")
    
    # Create build parameters
    if build_params is None:
        build_params = create_build_params()
    
    # Create time points
    tau_points = np.linspace(0, tau_max, tau_steps + 1)
    
    # Run replicates
    trajectories = []
    replicate_times = []
    
    for rep in range(replicates):
        seed = seed_offset + rep
        
        if verbose:
            print(f"Running replicate {rep+1}/{replicates}...")
        
        # Create simulator
        sim = create_simulator_instance(
            nuc=nuc,
            tau_points=tau_points,
            build_params=build_params,
            simulator_type=simulator_type,
            seed=seed,
            delta_tau_slow=delta_tau_slow if simulator_type == 'hybrid' else None
        )
        
        # Run simulation with timing
        start_time = time.perf_counter()
        trajectory = []
        for state in sim.run():
            trajectory.append({
                'tau': state.tau,
                'cs_total': state.cs_total
            })
        elapsed = time.perf_counter() - start_time
        
        trajectories.append(trajectory)
        replicate_times.append(elapsed)
        
        if verbose:
            print(f"  Completed in {elapsed:.4f}s")
            if simulator_type == 'hybrid' and hasattr(sim, 'print_stats'):
                sim.print_stats()
    
    # Calculate statistics
    avg_time = np.mean(replicate_times)
    std_time = np.std(replicate_times)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Average time per replicate: {avg_time:.4f} ± {std_time:.4f}s")
        print(f"{'='*70}\n")
    
    return {
        'simulator_type': simulator_type,
        'trajectories': trajectories,
        'tau_points': tau_points,
        'replicate_times': replicate_times,
        'avg_time': avg_time,
        'std_time': std_time,
        'nuc': nuc
    }


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison(
    nuc: Nucleosome,
    tau_max: float = 100.0,
    tau_steps: int = 100,
    replicates: int = 5,
    build_params: Optional[dict] = None,
    delta_tau_slow: float = 0.1,
    seed_offset: int = 0,
    plot: bool = True,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Compare Gillespie and Hybrid Rejection simulators.
    
    Args:
        nuc: Nucleosome instance
        tau_max: Maximum dimensionless time
        tau_steps: Number of time steps
        replicates: Number of replicates
        build_params: Build parameters (uses defaults if None)
        delta_tau_slow: Time step for slow reactions (hybrid only)
        seed_offset: Offset for random seeds
        plot: Whether to plot results
        save_path: Path to save plot (None = don't save)
    
    Returns:
        Dictionary with results from both simulators
    """
    print(f"\n{'#'*70}")
    print("GILLESPIE vs HYBRID REJECTION COMPARISON")
    print(f"{'#'*70}\n")
    
    # Run Gillespie
    gillespie_results = run_single_simulation(
        nuc=nuc,
        simulator_type='gillespie',
        tau_max=tau_max,
        tau_steps=tau_steps,
        replicates=replicates,
        build_params=build_params,
        seed_offset=seed_offset,
        verbose=True
    )
    
    # Run Hybrid Rejection
    hybrid_results = run_single_simulation(
        nuc=nuc,
        simulator_type='hybrid',
        tau_max=tau_max,
        tau_steps=tau_steps,
        replicates=replicates,
        build_params=build_params,
        delta_tau_slow=delta_tau_slow,
        seed_offset=seed_offset,
        verbose=True
    )
    
    # Print comparison
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"Gillespie:        {gillespie_results['avg_time']:.4f} ± {gillespie_results['std_time']:.4f}s")
    print(f"Hybrid Rejection: {hybrid_results['avg_time']:.4f} ± {hybrid_results['std_time']:.4f}s")
    speedup = gillespie_results['avg_time'] / hybrid_results['avg_time']
    print(f"Speedup:          {speedup:.2f}x")
    print(f"{'='*70}\n")
    
    results = {
        'gillespie': gillespie_results,
        'hybrid': hybrid_results,
        'speedup': speedup
    }
    
    # Plot if requested
    if plot:
        plot_comparison(results, save_path=save_path)
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_comparison(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot comparison of Gillespie vs Hybrid Rejection trajectories.
    
    Args:
        results: Dictionary from run_comparison()
        save_path: Path to save figure (None = display only)
    """
    setup_matplotlib_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    colors = {'gillespie': 'tab:blue', 'hybrid': 'tab:orange'}
    labels = {'gillespie': 'Gillespie', 'hybrid': 'Hybrid Rejection'}
    
    for sim_type, ax in zip(['gillespie', 'hybrid'], axes):
        sim_results = results[sim_type]
        tau_points = sim_results['tau_points']
        trajectories = sim_results['trajectories']
        
        # Convert to arrays for averaging
        cs_arrays = []
        for traj in trajectories:
            cs_vals = [state['cs_total'] for state in traj]
            cs_arrays.append(cs_vals)
        
        cs_arrays = np.array(cs_arrays)
        avg_cs = np.mean(cs_arrays, axis=0)
        std_cs = np.std(cs_arrays, axis=0)
        
        # Plot individual trajectories (lighter)
        for cs_vals in cs_arrays:
            ax.plot(tau_points, cs_vals, color=colors[sim_type], alpha=0.2, lw=0.8)
        
        # Plot average (bold)
        ax.plot(tau_points, avg_cs, color=colors[sim_type], lw=2.5, label='Average')
        
        # Add uncertainty band
        ax.fill_between(
            tau_points,
            np.clip(avg_cs - std_cs, 0, 147),
            np.clip(avg_cs + std_cs, 0, 147),
            color=colors[sim_type],
            alpha=0.2
        )
        
        # Formatting
        ax.set_xlabel(r'Dimensionless Time ($\tau$)')
        ax.set_ylabel('Wrapped Nucleosome Sites')
        ax.set_title(labels[sim_type])
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='best')
    
    # Add overall title with performance info
    speedup = results['speedup']
    fig.suptitle(
        f'Gillespie vs Hybrid Rejection (Speedup: {speedup:.2f}×)',
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        # Don't print for batch mode to avoid clutter
    else:
        # Only show plot if not saving (interactive mode)
        plt.show()


def plot_trajectory_single(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot trajectories from a single simulator run.
    
    Args:
        results: Dictionary from run_single_simulation()
        save_path: Path to save figure (None = display only)
    """
    setup_matplotlib_style()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sim_type = results['simulator_type']
    tau_points = results['tau_points']
    trajectories = results['trajectories']
    
    # Convert to arrays
    cs_arrays = []
    for traj in trajectories:
        cs_vals = [state['cs_total'] for state in traj]
        cs_arrays.append(cs_vals)
    
    cs_arrays = np.array(cs_arrays)
    avg_cs = np.mean(cs_arrays, axis=0)
    std_cs = np.std(cs_arrays, axis=0)
    
    color = 'tab:blue' if sim_type == 'gillespie' else 'tab:orange'
    
    # Plot individual trajectories
    for cs_vals in cs_arrays:
        ax.plot(tau_points, cs_vals, color=color, alpha=0.2, lw=0.8)
    
    # Plot average
    ax.plot(tau_points, avg_cs, color=color, lw=2.5, label='Average')
    
    # Uncertainty band
    ax.fill_between(
        tau_points,
        np.clip(avg_cs - std_cs, 0, 147),
        np.clip(avg_cs + std_cs, 0, 147),
        color=color,
        alpha=0.2
    )
    
    # Formatting
    ax.set_xlabel(r'Dimensionless Time ($\tau$)')
    ax.set_ylabel('Wrapped Nucleosome Sites')
    ax.set_title(f'{sim_type.upper()} Simulator')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def plot_survival_curves(
    trajectory_results: List[Dict],
    save_path: Optional[Path] = None
):
    """
    Plot survival curves showing all nucleosomes' trajectories together.
    
    Args:
        trajectory_results: List of result dictionaries with gillespie and hybrid trajectories
        save_path: Path to save figure (None = display only)
    """
    setup_matplotlib_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'gillespie': 'tab:blue', 'hybrid': 'tab:orange'}
    labels = {'gillespie': 'Gillespie', 'hybrid': 'Hybrid Rejection'}
    
    for sim_type, ax in zip(['gillespie', 'hybrid'], axes):
        all_trajectories = []
        tau_points = None
        
        # Collect all trajectories from all nucleosomes
        for result in trajectory_results:
            key = 'gillespie_results' if sim_type == 'gillespie' else 'hybrid_results'
            sim_results = result[key]
            
            if tau_points is None:
                tau_points = sim_results['tau_points']
            
            # Get all replicates for this nucleosome
            for traj in sim_results['trajectories']:
                cs_vals = [state['cs_total'] for state in traj]
                all_trajectories.append(cs_vals)
        
        # Convert to array
        all_trajectories = np.array(all_trajectories)
        
        # Plot each trajectory with low alpha
        for traj in all_trajectories:
            ax.plot(tau_points, traj, color=colors[sim_type], alpha=0.1, lw=0.5)
        
        # Calculate and plot overall average
        avg_trajectory = np.mean(all_trajectories, axis=0)
        std_trajectory = np.std(all_trajectories, axis=0)
        
        ax.plot(tau_points, avg_trajectory, color=colors[sim_type], lw=3, 
                label=f'Average (n={len(all_trajectories)})', zorder=10)
        
        # Add uncertainty band
        ax.fill_between(
            tau_points,
            np.clip(avg_trajectory - std_trajectory, 0, 14),
            np.clip(avg_trajectory + std_trajectory, 0, 14),
            color=colors[sim_type],
            alpha=0.3,
            zorder=5
        )
        
        # Formatting
        ax.set_xlabel(r'Dimensionless Time ($\tau$)')
        ax.set_ylabel('Wrapped Nucleosome Sites')
        ax.set_title(f'{labels[sim_type]} - All Nucleosomes')
        ax.set_ylim(0, 14)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='best')
    
    fig.suptitle(
        f'Survival Curves: {len(trajectory_results)} Nucleosomes',
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()


# =============================================================================
# BATCH COMPARISON FOR MULTIPLE NUCLEOSOMES
# =============================================================================

def _run_comparison_worker(
    nuc: Nucleosome,
    idx: int,
    tau_max: float,
    tau_steps: int,
    replicates: int,
    k_wrap: float,
    prot_p_conc: float,
    prot_cooperativity: float,
    prot_k_unbind: float,
    prot_k_bind: float,
    binding_sites: int,
    delta_tau_slow: float,
    return_trajectories: bool = False
) -> Dict:
    """
    Worker function to run comparison for a single nucleosome.
    This runs in a separate process.
    """
    # Create build_params fresh in this worker process
    build_params = create_build_params(
        k_wrap=k_wrap,
        prot_p_conc=prot_p_conc,
        prot_cooperativity=prot_cooperativity,
        prot_k_unbind=prot_k_unbind,
        prot_k_bind=prot_k_bind,
        binding_sites=binding_sites
    )
    
    result = run_comparison(
        nuc=nuc,
        tau_max=tau_max,
        tau_steps=tau_steps,
        replicates=replicates,
        build_params=build_params,
        delta_tau_slow=delta_tau_slow,
        seed_offset=idx * 1000,
        plot=False
    )
    
    output = {
        'nuc_id': nuc.id,
        'subid': nuc.subid,
        'gillespie_time': result['gillespie']['avg_time'],
        'hybrid_time': result['hybrid']['avg_time'],
        'speedup': result['speedup']
    }
    
    if return_trajectories:
        output['gillespie_results'] = result['gillespie']
        output['hybrid_results'] = result['hybrid']
    
    return output


def run_batch_comparison(
    file_path: Path,
    max_nucs: int = 10,
    tau_max: float = 100.0,
    tau_steps: int = 100,
    replicates: int = 3,
    build_params: Optional[dict] = None,
    delta_tau_slow: float = 0.1,
    subids_range: Optional[tuple] = None,
    output_dir: Optional[Path] = None,
    n_workers: int = 4,
    k_wrap: float = 21.0,
    kT: float = 1.0,
    binding_sites: int = 14,
    prot_p_conc: float = 100.0,
    prot_cooperativity: float = 0.0,
    prot_k_unbind: float = 100.0,
    prot_k_bind: float = 1.0,
    plot_trajectories: bool = True
) -> Dict:
    """
    Run comparison for multiple nucleosomes in parallel and aggregate statistics.
    
    Args:
        file_path: Path to input TSV file
        max_nucs: Maximum number of nucleosomes to process
        tau_max: Maximum dimensionless time
        tau_steps: Number of time steps
        replicates: Number of replicates per nucleosome
        build_params: Build parameters (ignored, parameters specified directly)
        delta_tau_slow: Time step for slow reactions (hybrid only)
        subids_range: Tuple (start, end) for subid range
        output_dir: Directory to save results
        n_workers: Number of parallel workers
        k_wrap: Wrapping rate for nucleosome loading and simulation
        kT: Temperature parameter for nucleosome loading
        binding_sites: Number of binding sites
        prot_p_conc: Protamine concentration (μM)
        prot_cooperativity: Cooperativity parameter
        prot_k_unbind: Protamine unbinding rate
        prot_k_bind: Protamine binding rate
        plot_trajectories: Whether to save trajectory comparison plots for each nucleosome
    
    Returns:
        Dictionary with aggregated statistics
    """
    print(f"\n{'#'*70}")
    print(f"BATCH COMPARISON: {max_nucs} Nucleosomes")
    print(f"{'#'*70}\n")
    
    # Generate nucleosomes using nucleosome_generator
    if subids_range is not None:
        gen = nucleosome_generator(
            file_path=str(file_path),
            k_wrap=k_wrap,
            kT=kT,
            binding_sites=binding_sites,
            subids=list(range(*subids_range))
        )
    else:
        gen = nucleosome_generator(
            file_path=str(file_path),
            k_wrap=k_wrap,
            kT=kT,
            binding_sites=binding_sites
        )
    
    # Limit to max_nucs
    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)
    
    # Convert to list for parallel processing
    nucs_list = list(gen)
    
    print(f"Processing {len(nucs_list)} nucleosomes with {n_workers} workers...")
    
    # Run comparisons in parallel
    gillespie_times = []
    hybrid_times = []
    speedups = []
    trajectory_results = []  # Store results with trajectories for plotting
    
    worker_func = partial(
        _run_comparison_worker,
        tau_max=tau_max,
        tau_steps=tau_steps,
        replicates=replicates,
        k_wrap=k_wrap,
        prot_p_conc=prot_p_conc,
        prot_cooperativity=prot_cooperativity,
        prot_k_unbind=prot_k_unbind,
        prot_k_bind=prot_k_bind,
        binding_sites=binding_sites,
        delta_tau_slow=delta_tau_slow,
        return_trajectories=plot_trajectories
    )
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(worker_func, nuc, idx)
            for idx, nuc in enumerate(nucs_list)
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Processing nucleosomes"):
            result = future.result()
            gillespie_times.append(result['gillespie_time'])
            hybrid_times.append(result['hybrid_time'])
            speedups.append(result['speedup'])
            
            if plot_trajectories:
                trajectory_results.append(result)
    
    # Aggregate statistics
    stats = {
        'n_nucleosomes': len(nucs_list),
        'gillespie_avg_time': np.mean(gillespie_times),
        'gillespie_std_time': np.std(gillespie_times),
        'hybrid_avg_time': np.mean(hybrid_times),
        'hybrid_std_time': np.std(hybrid_times),
        'avg_speedup': np.mean(speedups),
        'std_speedup': np.std(speedups),
        'gillespie_times': gillespie_times,
        'hybrid_times': hybrid_times,
        'speedups': speedups
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Nucleosomes processed: {stats['n_nucleosomes']}")
    print(f"Replicates per nucleosome: {replicates}")
    print(f"Gillespie avg:        {stats['gillespie_avg_time']:.4f} ± {stats['gillespie_std_time']:.4f}s")
    print(f"Hybrid Rejection avg: {stats['hybrid_avg_time']:.4f} ± {stats['hybrid_std_time']:.4f}s")
    print(f"Average speedup:      {stats['avg_speedup']:.2f}× ± {stats['std_speedup']:.2f}×")
    print(f"{'='*70}\n")
    
    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        stats_df = pl.DataFrame({
            'gillespie_time': gillespie_times,
            'hybrid_time': hybrid_times,
            'speedup': speedups
        })
        
        csv_path = output_dir / 'batch_comparison_stats.csv'
        stats_df.write_csv(csv_path)
        print(f"✓ Saved statistics to {csv_path}")
        
        # Plot trajectories if requested
        if plot_trajectories and trajectory_results:
            plots_dir = output_dir / 'trajectory_plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Generate individual trajectory plots
            print(f"\nGenerating individual trajectory plots...")
            for result in tqdm(trajectory_results, desc="Plotting individual trajectories"):
                nuc_id = result['nuc_id']
                subid = result['subid']
                
                # Create a comparison dictionary for plotting
                comparison_result = {
                    'gillespie': result['gillespie_results'],
                    'hybrid': result['hybrid_results'],
                    'speedup': result['speedup']
                }
                
                # Create filename-safe version of nuc_id
                safe_nuc_id = nuc_id.replace('/', '_').replace(':', '_')
                plot_path = plots_dir / f"{safe_nuc_id}_subid_{subid}.png"
                
                # Plot and save
                plot_comparison(comparison_result, save_path=plot_path)
                plt.close('all')  # Close figures to free memory
            
            print(f"✓ Saved {len(trajectory_results)} individual trajectory plots to {plots_dir}")
            
            # Generate survival curve with all nucleosomes
            print(f"\nGenerating survival curve...")
            survival_plot_path = output_dir / 'survival_curves_all_nucleosomes.png'
            plot_survival_curves(trajectory_results, save_path=survival_plot_path)
            plt.close('all')
            print(f"✓ Saved survival curve to {survival_plot_path}")
    
    return stats


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default file paths
DEFAULT_FILE = HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath" / "breath_energy" / "001.tsv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "hybrid_tests"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("HYBRID REJECTION SIMULATOR - LOCAL TESTING")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # EXAMPLE 1: Single Nucleosome - Gillespie vs Hybrid Comparison
    # =========================================================================
    
    # build_params = create_build_params(
    #     prot_p_conc=100.0,
    #     prot_cooperativity=0.0
    # )
    # 
    # # Load a single nucleosome for testing
    # gen = nucleosome_generator(
    #     file_path=str(DEFAULT_FILE),
    #     k_wrap=21.0,
    #     kT=1.0,
    #     binding_sites=14,
    #     subids=[2014]  # Specify a particular subid
    # )
    # nuc = next(gen)
    # 
    # results = run_comparison(
    #     nuc=nuc,
    #     tau_max=100.0,
    #     tau_steps=100,
    #     replicates=5,
    #     build_params=build_params,
    #     delta_tau_slow=0.1,
    #     plot=True,
    #     save_path=OUTPUT_DIR / "single_comparison.png"
    # )
    
    # =========================================================================
    # EXAMPLE 2: Single Nucleosome - Hybrid Only
    # =========================================================================
    
    # build_params = create_build_params(
    #     prot_p_conc=100.0,
    #     prot_cooperativity=0.0
    # )
    # 
    # # Load a single nucleosome
    # gen = nucleosome_generator(
    #     file_path=str(DEFAULT_FILE),
    #     k_wrap=21.0,
    #     kT=1.0,
    #     binding_sites=14,
    #     subids=[2014]
    # )
    # nuc = next(gen)
    # 
    # results = run_single_simulation(
    #     nuc=nuc,
    #     simulator_type='hybrid',
    #     tau_max=100.0,
    #     tau_steps=100,
    #     replicates=5,
    #     build_params=build_params,
    #     delta_tau_slow=0.1,
    #     verbose=True
    # )
    # 
    # plot_trajectory_single(results, save_path=OUTPUT_DIR / "hybrid_only.png")
    
    # =========================================================================
    # EXAMPLE 3: Batch Comparison - Multiple Nucleosomes (Parallel)
    # =========================================================================
    
    stats = run_batch_comparison(
        file_path=DEFAULT_FILE,
        max_nucs=20,
        tau_max=5000.0,
        tau_steps=500,
        replicates=5,
        delta_tau_slow=0.3,
        subids_range=None,
        output_dir=OUTPUT_DIR / "batch_comparison",
        n_workers=20,
        k_wrap=1.0,
        kT=1.0,
        binding_sites=14,
        prot_p_conc=100.0,
        prot_cooperativity=0.0,
        prot_k_unbind=89.7,
        prot_k_bind=1.0,
        plot_trajectories=True  # Enable trajectory plotting
    )
    
    # =========================================================================
    # EXAMPLE 4: Parameter Exploration - Different delta_tau_slow
    # =========================================================================
    
    # build_params = create_build_params(
    #     prot_p_conc=100.0,
    #     prot_cooperativity=0.0
    # )
    # 
    # delta_tau_values = [0.05, 0.1, 0.2, 0.5]
    # 
    # for delta_tau in delta_tau_values:
    #     print(f"\n{'='*70}")
    #     print(f"Testing delta_tau_slow = {delta_tau}")
    #     print(f"{'='*70}\n")
    #     
    #     results = run_single_simulation(
    #         file_path=DEFAULT_FILE,
    #         simulator_type='hybrid',
    #         tau_max=100.0,
    #         tau_steps=100,
    #         replicates=5,
    #         build_params=build_params,
    #         delta_tau_slow=delta_tau,
    #         verbose=True
    #     )
    #     
    #     plot_trajectory_single(
    #         results,
    #         save_path=OUTPUT_DIR / f"hybrid_delta_{delta_tau:.2f}.png"
    #     )
