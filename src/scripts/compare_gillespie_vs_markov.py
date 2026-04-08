"""
Gillespie vs Markov Solver Comparison Script
=============================================

This script compares survival functions from:
1. Gillespie simulations (stochastic trajectories)
2. Markov solver (analytical solution)

The survival function S(t) = P(nucleosome NOT fully detached by time t).

For Gillespie:
- Run multiple replicates and compute empirical S(t)
- S(t) = fraction of trajectories still attached at time t

For Markov:
- Compute exact survival function using matrix exponential
- Uses fast protamine limit (coarse-grained process)

Author: MY
Date: 2025-12-10
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
from typing import Optional

# Gillespie simulation
from src.simulation import run_simulation
from src.config.custom_type import SimulationConfig
from src.config.path import HAMNUCRET_DATA_DIR, RESULTS_DIR

# Markov solver
from src.analysis.markov_solver import (
    load_nucleosomes_from_file,
    build_full_Q_from_nucleosome,
    compute_survival,
    compute_mfpt_from_Q_TT,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ComparisonConfig:
    """
    Configuration for comparison study.
    
    Time Convention:
    ---------------
    All time parameters use dimensionless time τ = k_wrap × t_physical.
    Both Gillespie and Markov use the same τ-grid for direct comparison.
    """
    
    def __init__(
        self,
        # Nucleosome selection
        file_path: Path,
        max_nucs: int = 2,
        subids_range: Optional[tuple] = None,
        
        # # Simulation parameters
        k_wrap: float = 1.0,
        prot_k_bind: float = 1.0,
        prot_k_unbind: float = 89.7,
        prot_p_conc: float = 100.0,
        prot_cooperativity: float = 0.0,
        replicates: int = 100,
        n_workers: int = 10,
        batch_size: int = 1,

        # Time grid (dimensionless τ = k_wrap × t_physical)
        tau_max: float = 5000.0,
        tau_steps: int = 200, ##n_steps 
        method: str = "ode",  # Markov solver method options: "expm", "ode"
        
        # Output
        output_dir: Path = RESULTS_DIR / "gillespie_vs_markov",
    ):
        self.file_path = file_path
        self.max_nucs = max_nucs
        self.subids_range = subids_range
        
        self.k_wrap = k_wrap
        self.prot_k_bind = prot_k_bind
        self.prot_k_unbind = prot_k_unbind
        self.prot_p_conc = prot_p_conc
        self.prot_cooperativity = prot_cooperativity
        self.replicates = replicates
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.method = method
        
        self.tau_max = tau_max
        self.tau_steps = tau_steps
        self.t_grid = np.linspace(0, tau_max, tau_steps) ###dimensionless time grid
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GILLESPIE SIMULATION
# =============================================================================

def run_gillespie_simulation(config: ComparisonConfig) -> Path:
    """
    Run Gillespie simulation and return path to trajectory file.
    
    Args:
        config: ComparisonConfig instance
    
    Returns:
        Path to trajectory parquet file
    """
    print("\n" + "="*70)
    print("RUNNING GILLESPIE SIMULATION")
    print("="*70)
    
    # Create simulation config
    sim_config = SimulationConfig(
        k_wrap=config.k_wrap,
        prot_p_conc=config.prot_p_conc,
        prot_cooperativity=config.prot_cooperativity,
        replicates=config.replicates,
        n_workers=config.n_workers,
        batch_size=config.batch_size,
        tau_max=config.tau_max,
        tau_steps=config.tau_steps,
        save_trajectories=True,
        maxpoints_saved_trajectories=config.tau_steps
    )
    
    # Output files
    file_id = config.file_path.stem
    traj_outfile = config.output_dir / f"gillespie_{file_id}_trajectories.parquet"
    tsv_outfile = config.output_dir / f"gillespie_{file_id}_summary.tsv"
    
    print(f"File: {config.file_path.name}")
    print(f"Nucleosomes: {config.max_nucs}")
    print(f"Replicates: {config.replicates}")
    print(f"Protamine: c={config.prot_p_conc} μM, J={config.prot_cooperativity} kT")
    print(f"Time grid: {config.tau_steps} points, τ_max={config.tau_max}")
    
    # Run simulation
    run_simulation(
        file_path=config.file_path,
        traj_outfile=traj_outfile,
        tsv_outfile=tsv_outfile,
        config=sim_config,
        max_nucs=config.max_nucs,
    )

    print(f"\n✓ Gillespie simulation complete")
    print(f"  Trajectories: {traj_outfile}")
    
    return traj_outfile


def compute_gillespie_survival(
    traj_path: Path,
    t_grid: np.ndarray,
    binding_sites: int = 14
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Compute survival function from Gillespie trajectories.
    
    Both `t_grid` and the stored `tau_time` in the trajectories are
    dimensionless times τ = k_wrap * t_phys.
    
    The nucleosome is considered "detached" when cs_total reaches 0
    (all sites unwrapped). The survival function S(t) is the fraction
    of trajectories that remain attached at time t.
    
    Args:
        traj_path: Path to trajectory parquet file
        t_grid: Dimensionless time grid τ for interpolation
        binding_sites: Number of binding sites (N)
    
    Returns:
        S_mean: Mean survival probability (averaged over all nucleosomes)
        S_std: Standard deviation
        info: Dictionary with overall statistics
        per_nuc_survival: Dictionary mapping (id, subid) -> survival array
    """     
    print("\n" + "="*70)
    print("COMPUTING GILLESPIE SURVIVAL FUNCTION")
    print("="*70)
    
    # Load trajectories
    df = pl.read_parquet(traj_path)
    print(f"Loaded {len(df)} trajectories")
    
    # Get unique nucleosomes
    unique_nucs = df.select(['id', 'subid']).unique().to_dicts()
    print(f"Found {len(unique_nucs)} unique nucleosomes")
    
    # Extract time series
    tau_times = df['tau_time'][0]  # Same for all trajectories, dimensionless
    print(f"Dimensionless Time points in trajectories: {len(tau_times)}")
    
    # Diagnostic: check that tau grid from simulation matches t_grid
    if len(tau_times) == len(t_grid):
        max_diff = float(np.max(np.abs(np.array(tau_times) - t_grid)))
        print(f"  Checking τ-grid alignment: max |tau_times - t_grid| = {max_diff:.3e}")
        if max_diff > 1e-8:
            print("  ⚠ tau_times and t_grid differ slightly; using interpolation")
    else:
        print(f"  ⚠ tau_times length ({len(tau_times)}) != t_grid length ({len(t_grid)}); using interpolation")
    
    n_times = len(t_grid)
    all_survival_curves = []
    per_nuc_survival = {}
    all_detachment_times = []
    
    # Process each nucleosome separately
    for nuc_info in unique_nucs:
        nuc_id = nuc_info['id']
        nuc_subid = nuc_info['subid']
        
        # Filter trajectories for this nucleosome
        nuc_df = df.filter((pl.col('id') == nuc_id) & (pl.col('subid') == nuc_subid))
        cs_arrays = nuc_df['cs_total'].to_list()
        n_replicates = len(cs_arrays)
        
        # Storage for this nucleosome's survival
        survival_matrix = np.zeros((n_replicates, n_times))
        print(survival_matrix.shape)
        
        # Process each replicate
        for i, cs_traj in enumerate(cs_arrays):
            # Interpolate cs_total onto common time grid
            cs_interp = np.interp(t_grid, tau_times, cs_traj)
            # print("\nReplicate", i)
            # print("Time (τ) | cs_total")
            # print("-" * 30)
            # for t_val, cs_val in zip(t_grid, cs_interp):
            #     print(f"{t_val:8.3f} | {int(cs_val):2d}")
            # Find detachment time (when cs_total first reaches 0)
            detached_idx = np.where(cs_interp <= 0.5)[0]
            
            if len(detached_idx) > 0:
                t_detach = t_grid[detached_idx[0]]
                all_detachment_times.append(t_detach)
                survival_matrix[i, :] = (t_grid < t_detach).astype(float)
            else:
                survival_matrix[i, :] = 1.0
                all_detachment_times.append(np.inf)
        
        # Average over replicates for this nucleosome
        nuc_survival = survival_matrix.mean(axis=0)
        per_nuc_survival[(nuc_id, nuc_subid)] = nuc_survival
        all_survival_curves.append(nuc_survival)
        
        print(f"  Nucleosome {nuc_id}-{nuc_subid}: {n_replicates} replicates")
    
    # Average over all nucleosomes
    all_survival_curves = np.array(all_survival_curves)
    S_mean = all_survival_curves.mean(axis=0)
    S_std = all_survival_curves.std(axis=0)
    
    # Statistics
    detachment_times = np.array(all_detachment_times)
    finite_detach = detachment_times[np.isfinite(detachment_times)]
    
    info = {
        'n_trajectories': len(df),
        'n_nucleosomes': len(unique_nucs),
        'n_detached': len(finite_detach),
        'n_survived': len(detachment_times) - len(finite_detach),
        'detachment_times': detachment_times,
        'mean_detachment_time': np.mean(finite_detach) if len(finite_detach) > 0 else np.inf,
        'median_detachment_time': np.median(finite_detach) if len(finite_detach) > 0 else np.inf,
    }
    
    print(f"\nStatistics:")
    print(f"  Total trajectories: {info['n_trajectories']}")
    print(f"  Unique nucleosomes: {info['n_nucleosomes']}")
    print(f"  Detached: {info['n_detached']} ({100*info['n_detached']/info['n_trajectories']:.1f}%)")
    print(f"  Survived: {info['n_survived']} ({100*info['n_survived']/info['n_trajectories']:.1f}%)")
    if len(finite_detach) > 0:
        print(f"  Mean detachment time: {info['mean_detachment_time']:.2f} τ")
        print(f"  Median detachment time: {info['median_detachment_time']:.2f} τ")
    
    return S_mean, S_std, info, per_nuc_survival


# =============================================================================
# MARKOV SOLVER
# =============================================================================

def compute_markov_survival(
    config: ComparisonConfig,
    t_grid: np.ndarray
) -> tuple[np.ndarray, dict, dict]:
    """
    Compute exact survival function using Markov solver for all nucleosomes.
    
    Args:
        config: ComparisonConfig instance
        t_grid: Dimensionless time grid τ = k_wrap * t_phys. This should be the 
                same τ-grid used for the Gillespie simulation (config.t_grid).
    
    Returns:
        S_mean: Mean survival function (averaged over all nucleosomes)
        info: Dictionary with overall information
        per_nuc_survival: Dictionary mapping (id, subid) -> survival array
        
    Notes:
        The generator matrix Q_TT is constructed in units of k_wrap, making it
        effectively dimensionless. We evaluate exp(Q_TT * τ) where τ is the
        dimensionless time relative to k_wrap.
    """
    print("\n" + "="*70)
    print("COMPUTING MARKOV SURVIVAL FUNCTION")
    print("="*70)
    print(f"Using dimensionless τ-grid with respect to k_wrap = {config.k_wrap}")
    
    # Load nucleosomes
    nucs = load_nucleosomes_from_file(
        config.file_path,
        k_wrap=config.k_wrap,
        max_nucs=config.max_nucs
    )
    
    print(f"Loaded {len(nucs)} nucleosomes")
    
    # Build generator matrix parameters
    protamine_params = {
        'k_bind': config.prot_k_bind,
        'k_unbind': config.prot_k_unbind,
        'p_conc': config.prot_p_conc,
        'cooperativity': config.prot_cooperativity,
    }
    
    print(f"τ_grid: {len(t_grid)} points from 0 to {t_grid[-1]:.1f}")
    print(f"  → Physical time: 0 to {t_grid[-1]/config.k_wrap:.4f} seconds")
    
    # Storage for survival curves
    all_survival_curves = []
    per_nuc_survival = {}
    all_mfpts = []
    all_n_states = []
    
    # Compute survival for each nucleosome
    for nuc in nucs:
        print(f"\n  Processing nucleosome: id={nuc.id}, subid={nuc.subid}, N={nuc.binding_sites}")
        
        # Build generator matrix
        Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
            nuc,
            k_wrap=config.k_wrap,  # Sets the timescale; Q_TT is in units of k_wrap
            protamine_params=protamine_params,
            sparse=False, 
            dimensionless=True
        )
        print(f"    State space: {len(states)} transient states")

        # Compute survival function
        start_state = (0, 0)  # Fully wrapped
        S = compute_survival(
            Q_TT,
            state_index,
            start_state,
            t_grid,  # τ-grid, so we evaluate exp(Q_TT * τ)
            method=config.method  
        )
        
        # Compute MFPT
        mfpt, _ = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state)
        print(f"    MFPT: {mfpt:.2f} τ ({mfpt/nuc.k_wrap:.4f} sec)")
        
        # Store results
        per_nuc_survival[(nuc.id, nuc.subid)] = S
        all_survival_curves.append(S)
        all_mfpts.append(mfpt)
        all_n_states.append(len(states))
    
    # Average over all nucleosomes
    all_survival_curves = np.array(all_survival_curves)
    S_mean = all_survival_curves.mean(axis=0)
    S_std = all_survival_curves.std(axis=0)
    
    info = {
        'n_nucleosomes': len(nucs),
        'n_states': all_n_states,
        'mfpt': np.mean(all_mfpts),
        'mfpt_std': np.std(all_mfpts),
        'all_mfpts': all_mfpts,
        'binding_sites': nucs[0].binding_sites,  # Assume same for all
        'S_std': S_std,
    }
    
    print(f"\nOverall Results:")
    print(f"  Nucleosomes processed: {info['n_nucleosomes']}")
    print(f"  Mean MFPT: {info['mfpt']:.2f} ± {info['mfpt_std']:.2f} τ")
    print(f"  Mean state space: {np.mean(all_n_states):.0f} states")
    
    return S_mean, info, per_nuc_survival


# =============================================================================
# COMPARISON AND PLOTTING
# =============================================================================

def plot_comparison(
    t_grid: np.ndarray,
    S_gillespie: np.ndarray,
    S_gillespie_std: np.ndarray,
    S_markov: np.ndarray,
    config: ComparisonConfig,
    gillespie_info: dict,
    markov_info: dict,
    save_path: Optional[Path] = None
):
    """
    Plot comparison of Gillespie vs Markov survival functions.
    
    Args:
        t_grid: Time grid
        S_gillespie: Gillespie survival (mean)
        S_gillespie_std: Gillespie survival (std)
        S_markov: Markov survival
        config: ComparisonConfig
        gillespie_info: Dictionary with Gillespie statistics
        markov_info: Dictionary with Markov statistics
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # -------------------------------------------------------------------------
    # Top panel: Survival functions
    # -------------------------------------------------------------------------
    
    # Markov (analytical)
    ax1.plot(t_grid, S_markov, 'b-', lw=2.5, label='Markov (analytical)', zorder=3)
    
    # Gillespie (simulation)
    ax1.fill_between(
        t_grid,
        np.clip(S_gillespie - S_gillespie_std, 0, 1),
        np.clip(S_gillespie + S_gillespie_std, 0, 1),
        color='red', alpha=0.2, linewidth=0, label='Gillespie ± 1σ'
    )
    ax1.plot(t_grid, S_gillespie, 'r--', lw=2, label='Gillespie (mean)', zorder=2)
    
    # Formatting
    ax1.set_ylabel('Survival Probability S(t)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    
    # Title with parameters
    title = f"Nucleosome Detachment: Gillespie vs Markov\n"
    title += f"c={config.prot_p_conc} μM, J={config.prot_cooperativity} kT, "
    title += f"N={markov_info['binding_sites']}, "
    title += f"nucs={gillespie_info['n_nucleosomes']}, replicates/nuc={config.replicates}"
    ax1.set_title(title, fontsize=12, fontweight='bold')
    
    # Add MFPT lines
    mfpt_markov = markov_info['mfpt']/config.k_wrap
    # ax1.axvline(mfpt_markov, color='blue', ls=':', lw=1.5, alpha=0.7)
    
    if gillespie_info['mean_detachment_time'] < np.inf:
        mfpt_gillespie = gillespie_info['mean_detachment_time']
        # ax1.axvline(mfpt_gillespie, color='red', ls=':', lw=1.5, alpha=0.7)
    
    # -------------------------------------------------------------------------
    # Bottom panel: Difference
    # -------------------------------------------------------------------------
    
    diff = S_gillespie - S_markov
    
    ax2.fill_between(
        t_grid,
        diff - S_gillespie_std,
        diff + S_gillespie_std,
        color='gray', alpha=0.3, linewidth=0
    )
    ax2.plot(t_grid, diff, 'k-', lw=1.5, label='Gillespie - Markov')
    ax2.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
    
    # Statistics
    max_abs_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    
    ax2.set_xlabel('Dimensionless Time (τ)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Difference', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Max |diff| = {max_abs_diff:.4f}\nRMS diff = {rms_diff:.4f}"
    ax2.text(0.98, 0.95, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(save_path / f'gillespie_vs_markov.{ext}', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figures to {save_path}/")
    
    plt.show()


def plot_per_nucleosome_comparison(
    t_grid: np.ndarray,
    gillespie_per_nuc: dict,
    markov_per_nuc: dict,
    config: ComparisonConfig,
    save_path: Optional[Path] = None
):
    """
    Plot per-nucleosome comparison of survival functions.
    
    Args:
        t_grid: Time grid
        gillespie_per_nuc: Dict mapping (id, subid) -> Gillespie survival
        markov_per_nuc: Dict mapping (id, subid) -> Markov survival
        config: ComparisonConfig
        save_path: Optional path to save figure
    """
    n_nucs = len(gillespie_per_nuc)
    
    # Create subplot grid
    ncols = min(3, n_nucs)
    nrows = (n_nucs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()
    
    for idx, (nuc_key, S_gill) in enumerate(gillespie_per_nuc.items()):
        ax = axes[idx]
        
        # Get corresponding Markov survival
        S_mark = markov_per_nuc[nuc_key]
        
        # Plot
        ax.plot(t_grid, S_mark, 'b-', lw=2, label='Markov', alpha=0.8)
        ax.plot(t_grid, S_gill, 'r--', lw=2, label='Gillespie', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('τ')
        ax.set_ylabel('S(t)')
        ax.set_title(f'Nucleosome {nuc_key[0]}-{nuc_key[1]}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    
    # Hide unused subplots
    for idx in range(n_nucs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Per-Nucleosome Comparison (c={config.prot_p_conc} μM, J={config.prot_cooperativity} kT)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path / f'per_nucleosome_comparison.{ext}', dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved per-nucleosome figures to {save_path}/")
    
    plt.show()


def print_summary_statistics(
    S_gillespie: np.ndarray,
    S_markov: np.ndarray,
    gillespie_info: dict,
    markov_info: dict
):
    """Print summary statistics comparing both methods."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # MFPT comparison
    print("\nMean First Passage Time (MFPT):")
    print(f"  Markov:    {markov_info['mfpt']:.2f} ± {markov_info['mfpt_std']:.2f} τ")
    if gillespie_info['mean_detachment_time'] < np.inf:
        print(f"  Gillespie: {gillespie_info['mean_detachment_time']:.2f} τ")
        rel_error = abs(gillespie_info['mean_detachment_time'] - markov_info['mfpt']) / markov_info['mfpt']
        print(f"  Rel. error: {100*rel_error:.2f}%")
    
    # Survival function comparison
    diff = S_gillespie - S_markov
    print(f"\nSurvival Function Agreement:")
    print(f"  Max absolute difference: {np.max(np.abs(diff)):.6f}")
    print(f"  RMS difference:          {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"  Mean difference:         {np.mean(diff):.6f}")
    
    # Gillespie statistics
    print(f"\nGillespie Statistics:")
    print(f"  Total trajectories: {gillespie_info['n_trajectories']}")
    print(f"  Nucleosomes: {gillespie_info['n_nucleosomes']}")
    print(f"  Detached: {gillespie_info['n_detached']} ({100*gillespie_info['n_detached']/gillespie_info['n_trajectories']:.1f}%)")
    
    # Markov statistics
    print(f"\nMarkov Statistics:")
    print(f"  Nucleosomes processed: {markov_info['n_nucleosomes']}")
    print(f"  Mean state space: {np.mean(markov_info['n_states']):.0f} states")
    print(f"  Binding sites: {markov_info['binding_sites']}")
    print(f"  MFPT range: [{np.min(markov_info['all_mfpts']):.2f}, {np.max(markov_info['all_mfpts']):.2f}] τ")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_comparison(config: ComparisonConfig):
    """
    Run complete comparison workflow.
    
    Args:
        config: ComparisonConfig instance
    """
    print("\n" + "="*70)
    print("GILLESPIE VS MARKOV COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  File: {config.file_path.name}")
    print(f"  Nucleosomes: {config.max_nucs}")
    print(f"  Protamine: c={config.prot_p_conc} μM, J={config.prot_cooperativity} kT")
    print(f"  Replicates: {config.replicates}")
    print(f"  Time grid: {config.tau_steps} points, τ_max={config.tau_max}")
    print(f"  Output: {config.output_dir}")
    
    # Step 1: Run Gillespie simulation
    traj_path = run_gillespie_simulation(config)
    
    # Step 2: Compute Gillespie survival function
    S_gillespie, S_gillespie_std, gillespie_info, gillespie_per_nuc = compute_gillespie_survival(
        traj_path,
        config.t_grid,
        binding_sites=14  # Will be extracted from data in future
    )
    
    # Step 3: Compute Markov survival function
    S_markov, markov_info, markov_per_nuc = compute_markov_survival(config, config.t_grid)
    
    # Step 4: Compare and plot
    plot_comparison(
        config.t_grid,
        S_gillespie,
        S_gillespie_std,
        S_markov,
        config,
        gillespie_info,
        markov_info,
        save_path=config.output_dir / "figures"
    )
    
    # Step 5: Plot per-nucleosome comparison (if not too many)
    if gillespie_info['n_nucleosomes'] <= 20:
        plot_per_nucleosome_comparison(
            config.t_grid,
            gillespie_per_nuc,
            markov_per_nuc,
            config,
            save_path=config.output_dir / "figures"
        )
    else:
        print(f"\n⚠ Skipping per-nucleosome plot (too many: {gillespie_info['n_nucleosomes']} nucleosomes)")
    
    # Step 6: Print summary
    print_summary_statistics(S_gillespie, S_markov, gillespie_info, markov_info)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import os
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    
    # -------------------------------------------------------------------------
    # Example 1: Single nucleosome, no protamine
    # -------------------------------------------------------------------------
    
    # config = ComparisonConfig(
    #     file_path=HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv",
    #     max_nucs=1,
    #     prot_p_conc=0.0,
    #     prot_cooperativity=0.0,
    #     replicates=200,
    #     n_workers=10,
    #     tau_max=5000.0,
    #     tau_steps=200,
    #     output_dir=RESULTS_DIR / "gillespie_vs_markov" / "no_protamine"
    # )
    # 
    # run_comparison(config)
    
    # -------------------------------------------------------------------------
    # Example 2: Two nucleosomes, with protamine
    # -------------------------------------------------------------------------
    
    config = ComparisonConfig(
        file_path=HAMNUCRET_DATA_DIR / "exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv",
        max_nucs=20,
        prot_k_bind=1.0,
        prot_k_unbind=89.7,
        k_wrap=1.0,
        prot_p_conc=0.0,
        prot_cooperativity=0.0,
        replicates=50,
        n_workers=20,
        tau_max=10000.0,
        tau_steps=800,
        method="ode",
        output_dir=RESULTS_DIR / "gillespie_vs_markov" / "with_protamine_kwrap1"
    )
    
    run_comparison(config)
    
    # -------------------------------------------------------------------------
    # Example 3: Parameter scan
    # -------------------------------------------------------------------------
    
    # for p_conc in [0.0, 10.0, 100.0, 1000.0]:
    #     config = ComparisonConfig(
    #         file_path=HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv",
    #         max_nucs=1,
    #         prot_p_conc=p_conc,
    #         prot_cooperativity=0.0,
    #         replicates=300,
    #         n_workers=10,
    #         tau_max=5000.0,
    #         tau_steps=200,
    #         output_dir=RESULTS_DIR / "gillespie_vs_markov" / f"scan_c{p_conc:.0f}"
    #     )
    #     
    #     run_comparison(config)
