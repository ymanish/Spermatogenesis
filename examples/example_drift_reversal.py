#!/usr/bin/env python3
"""
Drift Reversal Analysis from File

This script loads nucleosomes from a data file and runs drift reversal analysis.
Demonstrates:
1. Loading nucleosomes from breath energy file
2. Running drift reversal analysis on each nucleosome
3. Visualizing results (drift, rates, quasi-potential, survival curves)
4. Comparing results across multiple nucleosomes
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.path import HAMNUCRET_DATA_DIR
from src.analysis.markov_solver import load_nucleosomes_from_file
from src.analysis.markov_solver.drift_reversal import DriftReversalAnalyzer
from src.analysis.markov_solver.drift_reversal_plots import (
    plot_full_analysis,
    plot_drift_and_rates,
    plot_quasi_potential,
    plot_survival_comparison,
    plot_committor
)


def analyze_single_nucleosome(nuc, protamine_params, k_wrap=1.0):
    """
    Analyze drift reversal for a single nucleosome.
    
    Parameters
    ----------
    nuc : Nucleosome
        The nucleosome to analyze
    protamine_params : dict
        Protamine binding parameters
    k_wrap : float
        Wrapping rate constant
    kT : float
        Thermal energy
    
    Returns
    -------
    results : DriftReversalResults
        Analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing nucleosome: {nuc.id}, SubID: {nuc.subid}")
    print(f"{'='*70}")
    print(f"Binding sites N = {nuc.binding_sites}")
    
    # Create analyzer
    analyzer = DriftReversalAnalyzer(
        nucleosome=nuc,
        k_wrap=k_wrap,
        protamine_params=protamine_params )
    
    # Run analysis
    print("Running drift reversal analysis...")
    results = analyzer.analyze()
    
    # Print key results
    print("\nRESULTS:")
    print(f"  Critical nucleus: n* = {results.n_star}")
    if results.n_star_refined is not None:
        print(f"    (refined: {results.n_star_refined:.2f})")
    print(f"  Barrier height: ΔΦ = {results.delta_phi:.3f}")
    print(f"  1D MFPT: {results.mfpt_1d / k_wrap:.3e} s")
    
    # Show drift at key points
    print(f"\n  Drift v(n) at key points:")
    for n in [1, 2, results.n_star if results.n_star else 5, results.N - 1]:
        if n < len(results.drift):
            print(f"    v({n}) = {results.drift[n]:.3e}")
    
    return results


def analyze_multiple_nucleosomes(file_path, protamine_params:dict, k_wrap=1.0, max_nucs=20 
                                 ):
    """
    Load nucleosomes from file and analyze each one.
    
    Parameters
    ----------
    file_path : Path or str
        Path to the breath energy file
    k_wrap : float
        Wrapping rate constant
    max_nucs : int
        Maximum number of nucleosomes to load
    protamine_params : dict
        Protamine binding parameters
    
    Returns
    -------
    results_list : list of DriftReversalResults
        Analysis results for each nucleosome
    nucs : list of Nucleosome
        The nucleosomes that were analyzed
    """
    print(f"\n{'='*70}")
    print(f"LOADING NUCLEOSOMES FROM FILE")
    print(f"{'='*70}")
    print(f"File: {file_path}")
    print(f"Max nucleosomes: {max_nucs}")
    
    # Load nucleosomes
    nucs = load_nucleosomes_from_file(
        file_path=file_path,
        k_wrap=k_wrap,
        max_nucs=max_nucs
    )
    
    print(f"\nLoaded {len(nucs)} nucleosomes")
    
    
    print(f"\nProtamine parameters:")
    for key, val in protamine_params.items():
        print(f"  {key}: {val}")
    
    # Analyze each nucleosome
    results_list = []
    for i, nuc in enumerate(nucs):
        print(f"\n[{i+1}/{len(nucs)}]")
        results = analyze_single_nucleosome(
            nuc=nuc,
            protamine_params=protamine_params,
            k_wrap=k_wrap
        )
        results_list.append(results)
    
    return results_list, nucs


def plot_results(results_list, nucs, output_dir):
    """
    Create plots for all analyzed nucleosomes.
    
    Parameters
    ----------
    results_list : list of DriftReversalResults
        Analysis results for each nucleosome
    nucs : list of Nucleosome
        The nucleosomes that were analyzed
    output_dir : Path
        Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print(f"GENERATING PLOTS")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    # Plot full analysis for each nucleosome
    for i, (results, nuc) in enumerate(zip(results_list, nucs)):
        print(f"\nPlotting nucleosome {i+1}/{len(nucs)}: {nuc.id}, SubID: {nuc.subid}")
        
        # Full 4-panel plot
        fig = plot_full_analysis(results)
        fig.suptitle(f"Drift Reversal Analysis: {nuc.id}, SubID: {nuc.subid}", fontsize=16, y=0.995)
        
        output_file = output_dir / f"drift_analysis_{nuc.id}_{nuc.subid}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
        
        # Separate committor plot for clarity
        fig_committor, ax = plt.subplots(figsize=(8, 6))
        plot_committor(results, ax=ax)
        ax.set_title(f"Committor Function: {nuc.id}, SubID: {nuc.subid}", fontsize=13)
        output_file_committor = output_dir / f"committor_{nuc.id}_{nuc.subid}.png"
        plt.savefig(output_file_committor, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file_committor}")
    
    # Compare critical nucleus across nucleosomes
    if len(results_list) > 1:
        print("\nComparing critical nucleus across nucleosomes...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        n_stars = [r.n_star for r in results_list]
        n_stars_refined = [r.n_star_refined if r.n_star_refined else r.n_star 
                          for r in results_list]
        delta_phis = [r.delta_phi for r in results_list]
        mfpts = [r.mfpt_1d / r.k_wrap for r in results_list]
        binding_sites = [nuc.binding_sites for nuc in nucs]
        
        # Plot 1: Critical nucleus
        ax = axes[0, 0]
        ax.plot(range(1, len(n_stars)+1), n_stars, 'o-', label='Integer n*')
        ax.plot(range(1, len(n_stars_refined)+1), n_stars_refined, 's--', 
                label='Refined n*', alpha=0.7)
        ax.set_xlabel('Nucleosome index')
        ax.set_ylabel('Critical nucleus n*')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Barrier height
        ax = axes[0, 1]
        ax.plot(range(1, len(delta_phis)+1), delta_phis, 'o-')
        ax.set_xlabel('Nucleosome index')
        ax.set_ylabel('Barrier height ΔΦ')
        ax.grid(alpha=0.3)
        
        # Plot 3: MFPT
        ax = axes[1, 0]
        ax.semilogy(range(1, len(mfpts)+1), mfpts, 'o-')
        ax.set_xlabel('Nucleosome index')
        ax.set_ylabel('MFPT (s)')
        ax.grid(alpha=0.3)
        
        # Plot 4: n* vs N
        ax = axes[1, 1]
        ax.plot(binding_sites, n_stars_refined, 'o')
        ax.set_xlabel('Binding sites N')
        ax.set_ylabel('Critical nucleus n*')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / "drift_comparison_across_nucleosomes.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    print(f"\n✓ All plots saved to: {output_dir}")


def main():
    """Main execution function."""
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # File path
    if DATASET == "unbound":
        dataset = "exactpoint_unboundpromoter_regions_breath"
    else:
        DATASET = "bound"
        dataset = "exactpoint_boundpromoter_regions_breath"
    
    file_path = HAMNUCRET_DATA_DIR / f"{dataset}/breath_energy/001.tsv"
    
    # Parameters
    k_wrap = 1.0
    max_nucs = 20
    
    # Protamine parameters
    protamine_params = {
        'k_bind': 1.0,
        'k_unbind': 89.7,
        'p_conc': 10.0,        # Protamine concentration
        'cooperativity': 4.5  # Cooperativity parameter J
    }
    
    # Output directory
    output_dir = project_root / "output" / "drift_reversal_analysis"
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    # Analyze nucleosomes
    results_list, nucs = analyze_multiple_nucleosomes(
        file_path=file_path,
        protamine_params=protamine_params,
        k_wrap=k_wrap,
        max_nucs=max_nucs,
    )
    
    # Generate plots
    plot_results(results_list, nucs, output_dir)
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    n_stars = [r.n_star for r in results_list]
    delta_phis = [r.delta_phi for r in results_list]
    mfpts = [r.mfpt_1d / r.k_wrap for r in results_list]
    
    print(f"\nCritical nucleus n*:")
    print(f"  Mean: {np.mean(n_stars):.1f}")
    print(f"  Std:  {np.std(n_stars):.1f}")
    print(f"  Range: [{np.min(n_stars)}, {np.max(n_stars)}]")
    
    print(f"\nBarrier height ΔΦ:")
    print(f"  Mean: {np.mean(delta_phis):.2f}")
    print(f"  Std:  {np.std(delta_phis):.2f}")
    print(f"  Range: [{np.min(delta_phis):.2f}, {np.max(delta_phis):.2f}]")
    
    print(f"\nMFPT (seconds):")
    print(f"  Mean: {np.mean(mfpts):.2e}")
    print(f"  Std:  {np.std(mfpts):.2e}")
    print(f"  Range: [{np.min(mfpts):.2e}, {np.max(mfpts):.2e}]")
    
    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
