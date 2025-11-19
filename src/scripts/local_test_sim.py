"""
Local Testing Script for Nucleosome Simulations
================================================

Simplified script for testing nucleosome simulations locally using the new
modular simulation package and SimulationConfig class.

This script uses the production `run_simulation()` function which:
1. Creates temporary files for each parallel worker
2. Merges them into final output files
3. Provides the final results for plotting and analysis

QUICK START:
-----------
1. Uncomment an example in __main__
2. Run: python src/scripts/local_test_sim.py

Author: MY  
Last Updated: 2025-11-16
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator, MultipleLocator
from pathlib import Path
from typing import Optional, Dict
import polars as pl

# Use the new modular simulation package
from src.simulation import run_simulation
from src.config.custom_type import SimulationConfig
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
# SIMULATION RUNNER
# =============================================================================

def run_local_simulation(
    file_path: Path,
    config: SimulationConfig,
    output_dir: Path,
    label: str = "Simulation",
    max_nucs: Optional[int] = None,
    subids_range: Optional[tuple] = None
) -> Dict:
    """
    Run simulations using the production simulation module.
    
    The production run_simulation() function:
    - Creates temporary files for each parallel worker
    - Merges them into final output files
    - Cleans up temporary files automatically
    
    Args:
        file_path: Path to input TSV file
        config: SimulationConfig instance
        output_dir: Directory for output files
        label: Label for this simulation run
    
    Returns:
        Dictionary with trajectory and summary file paths
    """
    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"{'='*70}")
    print(f"File: {file_path.name}")
    print(f"Config: {config}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define final output file paths (not temporary!)
    file_id = file_path.stem
    tsv_outfile = output_dir / f"{label}_{file_id}_summary.tsv"
    traj_outfile = output_dir / f"{label}_{file_id}_trajectories.parquet"
    
    # Run simulation using production code
    # This creates worker-specific temporary files internally and merges them
    run_simulation(
        file_path=file_path,
        traj_outfile=traj_outfile,
        tsv_outfile=tsv_outfile,
        config=config, 
        max_nucs=max_nucs,
        subids_range=subids_range
    )
    
    return {
        'label': label,
        'trajectory_path': traj_outfile if config.save_trajectories else None,
        'summary_path': tsv_outfile
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_trajectory_comparison(
    result_paths: Dict[str, Dict],
    config: SimulationConfig,
    save_path: Optional[Path] = None
):
    """
    Plot averaged trajectories from parquet files.
    
    Args:
        result_paths: Dict mapping labels to result dictionaries
        config: SimulationConfig instance
        save_path: Optional path to save figure
    """
    if not result_paths:
        print("No results to plot")
        return
    
    if not config.save_trajectories:
        print("⚠ Trajectories not saved. Set save_trajectories=True in config.")
        return
    
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    
    colors = {'RET': 'tab:green', 'EVI': 'tab:orange'}
    linestyles = {'RET': '-', 'EVI': '--'}
    
    for label, result_dict in result_paths.items():
        traj_path = result_dict.get('trajectory_path')
        if not traj_path or not traj_path.exists():
            print(f"⚠ No trajectory file for {label}")
            continue
        
        # Load trajectories using Polars
        print(f"Loading trajectories for {label}...")
        df = pl.read_parquet(traj_path)
        
        if 'tau_time' not in df.columns or 'cs_total' not in df.columns:
            print(f"⚠ Required columns not found in {label}")
            continue
        
        # Extract time series data (list columns)
        tau_times = df['tau_time'][0]  # Same for all rows
        cs_arrays = df['cs_total'].to_list()
        
        # Calculate average and std across all nucleosomes/replicates
        avg_cs = np.mean(cs_arrays, axis=0)
        std_cs = np.std(cs_arrays, axis=0)
        
        color = colors.get(label, f'C{len(ax.lines)}')
        ls = linestyles.get(label, '-')
        
        # Plot with uncertainty band
        ax.fill_between(
            tau_times,
            np.clip(avg_cs - std_cs, 0, config.binding_sites),
            np.clip(avg_cs + std_cs, 0, config.binding_sites),
            color=color, alpha=0.12, linewidth=0
        )
        ax.plot(tau_times, avg_cs, label=label, color=color, ls=ls, lw=2.2)
    
    # Formatting
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='x', which='major', length=6)
    ax.tick_params(axis='x', which='minor', length=3, color='0.3')
    
    ax.set_ylim(0, config.binding_sites + 0.2)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='major', length=6)
    ax.tick_params(axis='y', which='minor', length=3, labelleft=False, color='0.4')
    
    ax.set_xlabel("Dimensionless Time (τ)")
    ax.set_ylabel("Wrapped Sites")
    
    # Title
    p_conc = config.prot_params['p_conc']
    coop = config.prot_params['cooperativity']
    if p_conc > 0:
        ax.set_title(f"c = {p_conc} μM, J = {coop} $k_B T$")
    else:
        ax.set_title("No Protamine")
    
    ax.grid(which='major', alpha=0.35)
    ax.grid(which='minor', alpha=0.12)
    ax.legend(framealpha=0.85, loc='best')
    
    plt.tight_layout()
    plt.show()
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(save_path / f"comparison.{ext}", dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}/")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_summary_stats(result_paths: Dict[str, Dict]):
    """Analyze and print summary statistics from TSV files."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    for label, result_dict in result_paths.items():
        tsv_path = result_dict.get('summary_path')
        if not tsv_path or not tsv_path.exists():
            continue
        
        df = pl.read_csv(tsv_path, separator='\t')
        
        print(f"{label}:")
        print(f"  Nucleosomes: {len(df)}")
        if 'avg_cs_total' in df.columns:
            print(f"  Mean wrapped sites: {df['avg_cs_total'].mean():.2f} ± {df['avg_cs_total'].std():.2f}")
        if 'avg_bprot' in df.columns:
            print(f"  Mean bound protamines: {df['avg_bprot'].mean():.2f} ± {df['avg_bprot'].std():.2f}")
        print()


# =============================================================================
# MAIN WORKFLOW FUNCTIONS
# =============================================================================

def run_comparison_analysis(
    config: SimulationConfig,
    file_bound: Path,
    file_unbound: Path,
    output_dir: Path,
    save_plot: Optional[Path] = None, 
    max_nucs: Optional[int]=None, 
    subids_range:Optional[tuple]=None
):
    """
    Compare bound (RET) vs unbound (EVI) promoter regions.
    
    Args:
        config: SimulationConfig with save_trajectories=True for plotting
        file_bound: Path to bound promoter data
        file_unbound: Path to unbound promoter data
        output_dir: Directory for output files
        save_plot: Optional path to save figures
        max_nucs: Number of Nucleosomes simulation to run Optional[int] = None
        subids_range: Optional[tuple] = None    
    """
    # Run simulations
    results_bound = run_local_simulation(file_bound, config, output_dir, "RET", max_nucs=max_nucs, subids_range=subids_range)
    results_unbound = run_local_simulation(file_unbound, config, output_dir, "EVI", max_nucs=max_nucs, subids_range=subids_range)

    # Plot and analyze
    plot_trajectory_comparison(
        {'RET': results_bound, 'EVI': results_unbound},
        config,
        save_plot
    )
    analyze_summary_stats({'RET': results_bound, 'EVI': results_unbound})
    
    return results_bound, results_unbound


def run_parameter_scan(
    file_path: Path,
    configs: Dict[str, SimulationConfig],
    output_dir: Path,
    save_plot: Optional[Path] = None
):
    """
    Scan multiple parameter combinations.
    
    Args:
        file_path: Path to data file
        configs: Dict mapping labels to SimulationConfig instances
        output_dir: Directory for output files
        save_plot: Optional path to save figure
    """
    all_results = {}
    
    for label, config in configs.items():
        result = run_local_simulation(file_path, config, output_dir, label)
        all_results[label] = result
    
    # Use last config for plotting parameters (binding_sites, etc.)
    last_config = list(configs.values())[-1]
    plot_trajectory_comparison(all_results, last_config, save_plot)
    analyze_summary_stats(all_results)
    
    return all_results


# =============================================================================
# DEFAULT FILE PATHS
# =============================================================================

DEFAULT_FILE_BOUND = Path(
    HAMNUCRET_DATA_DIR / "exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
)
DEFAULT_FILE_UNBOUND = Path(
    HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"
)

# Local output directory for test runs
OUTPUT_DIR = RESULTS_DIR / "local_tests"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("NUCLEOSOME SIMULATION - LOCAL TESTING")
    print("="*70)

    # Setup temporary directory
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    
    # =========================================================================
    # EXAMPLE 1: No Protamine - RET vs EVI
    # =========================================================================
    
    # config_no_prot = SimulationConfig(
    #     prot_p_conc=0.0,
    #     prot_cooperativity=0.0,
    #     replicates=10,
    #     n_workers=4,
    #     batch_size=10,
    #     save_trajectories=True,     # Required for plotting
    #     maxpoints_saved_trajectories=100
    # )
    # 
    # run_comparison_analysis(
    #     config=config_no_prot,
    #     file_bound=DEFAULT_FILE_BOUND,
    #     file_unbound=DEFAULT_FILE_UNBOUND,
    #     output_dir=OUTPUT_DIR / "no_protamine",
    #     save_plot=OUTPUT_DIR / "no_protamine" / "figures"
    # )
    
    # =========================================================================
    # EXAMPLE 2: With Protamine - RET vs EVI
    # =========================================================================
    
    config_with_prot = SimulationConfig(
        prot_p_conc=100.0,
        prot_cooperativity=0.0,
        replicates=10,
        n_workers=20,
        batch_size=1,
        tau_max=1000.0,
        tau_steps=100,
        save_trajectories=True,
        maxpoints_saved_trajectories=100
    )
    
    run_comparison_analysis(
        config=config_with_prot,
        file_bound=DEFAULT_FILE_BOUND,
        file_unbound=DEFAULT_FILE_UNBOUND,
        output_dir=OUTPUT_DIR / "with_protamine",
        save_plot=OUTPUT_DIR / "with_protamine" / "figures", 
        max_nucs=20, ##### Number of Nucleosomes simulation to run
        subids_range=(2000, 2050)
    )
    
    # =========================================================================
    # EXAMPLE 3: Parameter Scan
    # =========================================================================
    
    # configs = {
    #     "No Prot": SimulationConfig(
    #         prot_p_conc=0.0,
    #         replicates=10,
    #         n_workers=8,
    #         save_trajectories=True
    #     ),
    #     "100 μM": SimulationConfig(
    #         prot_p_conc=100.0,
    #         prot_cooperativity=4.5,
    #         replicates=10,
    #         n_workers=8,
    #         save_trajectories=True
    #     ),
    #     "1000 μM": SimulationConfig(
    #         prot_p_conc=1000.0,
    #         prot_cooperativity=4.5,
    #         replicates=10,
    #         n_workers=8,
    #         save_trajectories=True
    #     ),
    # }
    # 
    # run_parameter_scan(
    #     file_path=DEFAULT_FILE_BOUND,
    #     configs=configs,
    #     output_dir=OUTPUT_DIR / "parameter_scan",
    #     save_plot=OUTPUT_DIR / "parameter_scan" / "figures"
    # )
