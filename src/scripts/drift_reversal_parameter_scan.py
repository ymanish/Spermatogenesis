#!/usr/bin/env python3
"""
Drift Reversal Parameter Space Analysis

This script performs drift reversal analysis across:
- Multiple nucleosomes from bound and unbound datasets
- Different protamine parameter sets
- Computes and plots averaged committor and quasi-potential landscapes

Directory structure:
output/drift_parameter_scan/
    bound/
        param_set_1/
            individual/  # Individual nucleosome plots
            averaged/    # Averaged quantities
        param_set_2/
            ...
    unbound/
        param_set_1/
            ...
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.path import HAMNUCRET_DATA_DIR
from src.analysis.markov_solver import load_nucleosomes_from_file
from src.analysis.markov_solver.drift_reversal import DriftReversalAnalyzer, DriftReversalResults
from src.analysis.markov_solver.drift_reversal_plots import (
    plot_committor,
    plot_quasi_potential
)


@dataclass
class ParameterSet:
    """Container for protamine parameters."""
    name: str
    k_bind: float
    k_unbind: float
    p_conc: float
    cooperativity: float
    
    def to_dict(self):
        return {
            'k_bind': self.k_bind,
            'k_unbind': self.k_unbind,
            'p_conc': self.p_conc,
            'cooperativity': self.cooperativity
        }
    
    def __str__(self):
        return f"pconc{self.p_conc}_kunb{self.k_unbind}_coop{self.cooperativity}"


def analyze_nucleosome(nuc, protamine_params: Dict, k_wrap: float = 1.0) -> DriftReversalResults:
    """
    Run drift reversal analysis on a single nucleosome.
    
    Parameters
    ----------
    nuc : Nucleosome
        The nucleosome to analyze
    protamine_params : dict
        Protamine parameters
    k_wrap : float
        Wrapping rate
        
    Returns
    -------
    DriftReversalResults
        Analysis results
    """
    analyzer = DriftReversalAnalyzer(
        nucleosome=nuc,
        k_wrap=k_wrap,
        protamine_params=protamine_params    )
    
    results = analyzer.analyze()
    return results


def compute_averaged_quantities(results_list: List[DriftReversalResults]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute averaged committor and quasi-potential across nucleosomes.
    
    Parameters
    ----------
    results_list : list of DriftReversalResults
        Results from multiple nucleosomes
        
    Returns
    -------
    n_avg : np.ndarray
        Average n values (assume all have same N)
    committor_avg : np.ndarray
        Average committor q(n)
    phi_avg : np.ndarray
        Average quasi-potential Φ(n)
    """
    # Assume all nucleosomes have the same number of binding sites
    N = results_list[0].N
    
    # Collect all committors and potentials
    committors = []
    phis = []
    
    for res in results_list:
        if res.N == N:  # Only include nucleosomes with matching N
            committors.append(res.committor)
            phis.append(res.phi)
    
    committors = np.array(committors)
    phis = np.array(phis)
    
    # Compute averages
    committor_avg = np.mean(committors, axis=0)
    committor_std = np.std(committors, axis=0)
    phi_avg = np.mean(phis, axis=0)
    phi_std = np.std(phis, axis=0)
    
    n_avg = np.arange(N + 1)
    
    return n_avg, committor_avg, committor_std, phi_avg, phi_std


def plot_averaged_committor(n_values: np.ndarray, 
                            committor_avg: np.ndarray, 
                            committor_std: np.ndarray,
                            param_set: ParameterSet,
                            dataset_name: str,
                            output_file: Path):
    """
    Plot averaged committor with standard deviation.
    
    Parameters
    ----------
    n_values : np.ndarray
        n coordinate values
    committor_avg : np.ndarray
        Average committor
    committor_std : np.ndarray
        Standard deviation
    param_set : ParameterSet
        Parameter set info
    dataset_name : str
        'bound' or 'unbound'
    output_file : Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot average with shaded error region
    ax.plot(n_values, committor_avg, 'o-', linewidth=2, color='purple', markersize=6, label='Average')
    ax.fill_between(n_values, 
                     committor_avg - committor_std, 
                     committor_avg + committor_std,
                     alpha=0.3, color='purple', label='± 1 std')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='q = 0.5')
    ax.set_xlabel('Reduced coordinate $n = l + r$', fontsize=13)
    ax.set_ylabel('Committor $q(n)$', fontsize=13)
    ax.set_title(f'Averaged Committor Function\n{dataset_name.capitalize()} Dataset - {param_set.name}', 
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved averaged committor: {output_file}")


def plot_averaged_quasi_potential(n_values: np.ndarray,
                                  phi_avg: np.ndarray,
                                  phi_std: np.ndarray,
                                  param_set: ParameterSet,
                                  dataset_name: str,
                                  output_file: Path):
    """
    Plot averaged quasi-potential with standard deviation.
    
    Parameters
    ----------
    n_values : np.ndarray
        n coordinate values
    phi_avg : np.ndarray
        Average quasi-potential
    phi_std : np.ndarray
        Standard deviation
    param_set : ParameterSet
        Parameter set info
    dataset_name : str
        'bound' or 'unbound'
    output_file : Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot average with shaded error region
    ax.plot(n_values, phi_avg, 'o-', linewidth=2, color='darkblue', markersize=6, label='Average')
    ax.fill_between(n_values,
                     phi_avg - phi_std,
                     phi_avg + phi_std,
                     alpha=0.3, color='darkblue', label='± 1 std')
    
    # Mark barrier top (maximum of average)
    n_star_avg = np.argmax(phi_avg)
    ax.plot(n_star_avg, phi_avg[n_star_avg], 'r*', markersize=15, label=f'Avg barrier top (n={n_star_avg})')
    
    ax.set_xlabel('Reduced coordinate $n = l + r$', fontsize=13)
    ax.set_ylabel('Quasi-potential $\\Phi(n)$', fontsize=13)
    ax.set_title(f'Averaged Quasi-Potential Landscape\n{dataset_name.capitalize()} Dataset - {param_set.name}',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved averaged quasi-potential: {output_file}")


def plot_comparison_across_parameters(all_results: Dict[str, Dict[str, List[DriftReversalResults]]],
                                      output_dir: Path):
    """
    Plot comparison of averaged quantities across different parameter sets.
    
    Parameters
    ----------
    all_results : dict
        Nested dict: {dataset_name: {param_set_name: [results_list]}}
    output_dir : Path
        Output directory
    """
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS ACROSS PARAMETERS")
    print("="*70)
    
    # Define colors and markers
    dataset_colors = {'bound': 'tab:green', 'unbound': 'tab:orange'}
    markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    
    # For each dataset, compare parameter sets
    for dataset_name, param_results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        
        color = dataset_colors.get(dataset_name, 'blue')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Averaged committor for all parameter sets
        ax = axes[0]
        for idx, (param_name, results_list) in enumerate(param_results.items()):
            n_avg, q_avg, q_std, phi_avg, phi_std = compute_averaged_quantities(results_list)
            marker = markers[idx % len(markers)]
            ax.plot(n_avg, q_avg, linestyle='-', marker=marker, color=color, 
                   label=param_name, linewidth=2.5, markersize=8)
        
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Reduced coordinate $n = l + r$', fontsize=30, fontweight='bold')
        ax.set_ylabel('Committor $q(n)$', fontsize=30, fontweight='bold')
        ax.set_title(f'Averaged Committor - {dataset_name.capitalize()}', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=2)
        ax.legend(fontsize=12, loc='best')
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Plot 2: Averaged quasi-potential for all parameter sets
        ax = axes[1]
        for idx, (param_name, results_list) in enumerate(param_results.items()):
            n_avg, q_avg, q_std, phi_avg, phi_std = compute_averaged_quantities(results_list)
            marker = markers[idx % len(markers)]
            ax.plot(n_avg, phi_avg, linestyle='-', marker=marker, color=color, 
                   label=param_name, linewidth=2.5, markersize=8)

        ax.set_xlabel('Reduced coordinate $n = l + r$', fontsize=30, fontweight='bold')
        ax.set_ylabel('Quasi-potential $\\Phi(n)$', fontsize=30, fontweight='bold')
        ax.set_title(f'Averaged Quasi-Potential - {dataset_name.capitalize()}', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2)
        ax.legend(fontsize=12, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        for ext in [".png", ".svg", ".pdf"]:
            output_file = output_dir / f"comparison_{dataset_name}{ext}"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / f'comparison_{dataset_name}.png'}")

def analyze_dataset_with_params(
    file_path: Path,
    param_set: ParameterSet,
    dataset_name: str,
    output_base_dir: Path,
    k_wrap: float = 1.0,
    max_nucs: int = 100
) -> List[DriftReversalResults]:
    """
    Analyze a dataset with a specific parameter set.
    
    Parameters
    ----------
    file_path : Path
        Path to the breath energy file
    param_set : ParameterSet
        Parameter set to use
    dataset_name : str
        'bound' or 'unbound'
    output_base_dir : Path
        Base output directory
    k_wrap : float
        Wrapping rate
    max_nucs : int
        Maximum number of nucleosomes to analyze
        
    Returns
    -------
    results_list : list of DriftReversalResults
        Analysis results for all nucleosomes
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {dataset_name.upper()} - {param_set.name}")
    print(f"{'='*70}")
    print(f"File: {file_path}")
    print(f"Parameters:")
    for key, val in param_set.to_dict().items():
        print(f"  {key}: {val}")
    
    # Create output directories
    param_dir = output_base_dir / dataset_name / str(param_set)
    individual_dir = param_dir / "individual"
    averaged_dir = param_dir / "averaged"
    individual_dir.mkdir(parents=True, exist_ok=True)
    averaged_dir.mkdir(parents=True, exist_ok=True)
    
    # Load nucleosomes
    print(f"\nLoading nucleosomes (max={max_nucs})...")
    nucs = load_nucleosomes_from_file(
        file_path=file_path,
        k_wrap=k_wrap,
        max_nucs=max_nucs
    )
    print(f"Loaded {len(nucs)} nucleosomes")
    
    # Analyze each nucleosome
    results_list = []
    for i, nuc in enumerate(nucs):
        print(f"  [{i+1}/{len(nucs)}] Analyzing {nuc.id} (SubID: {nuc.subid}), N={nuc.binding_sites}")
        
        try:
            results = analyze_nucleosome(nuc, param_set.to_dict(), k_wrap)
            results_list.append(results)
            
            # Save individual plots (optional - can comment out for speed)
            # Uncomment to save individual nucleosome plots:
            # fig_q, ax = plt.subplots(figsize=(8, 6))
            # plot_committor(results, ax=ax)
            # output_file = individual_dir / f"committor_{nuc.id}_{nuc.subid}.png"
            # plt.savefig(output_file, dpi=100, bbox_inches='tight')
            # plt.close()
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    print(f"\nSuccessfully analyzed {len(results_list)} nucleosomes")
    
    # Compute and plot averaged quantities
    if results_list:
        print("\nComputing averaged quantities...")
        n_avg, q_avg, q_std, phi_avg, phi_std = compute_averaged_quantities(results_list)
        
        # Plot averaged committor
        output_file_q = averaged_dir / "averaged_committor.png"
        plot_averaged_committor(n_avg, q_avg, q_std, param_set, dataset_name, output_file_q)
        
        # Plot averaged quasi-potential
        output_file_phi = averaged_dir / "averaged_quasi_potential.png"
        plot_averaged_quasi_potential(n_avg, phi_avg, phi_std, param_set, dataset_name, output_file_phi)
        
        # Save numerical data
        data_file = averaged_dir / "averaged_data.npz"
        np.savez(data_file,
                 n_values=n_avg,
                 committor_avg=q_avg,
                 committor_std=q_std,
                 phi_avg=phi_avg,
                 phi_std=phi_std,
                 n_nucleosomes=len(results_list))
        print(f"  Saved numerical data: {data_file}")
        
        # Save summary statistics
        summary_file = averaged_dir / "summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Drift Reversal Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Parameter Set: {param_set.name}\n")
            f.write(f"Number of nucleosomes: {len(results_list)}\n")
            f.write(f"\nParameters:\n")
            for key, val in param_set.to_dict().items():
                f.write(f"  {key}: {val}\n")
            f.write(f"\nCritical nucleus statistics:\n")
            n_stars = [r.n_star for r in results_list if r.n_star is not None]
            if n_stars:
                f.write(f"  Mean: {np.mean(n_stars):.2f}\n")
                f.write(f"  Std: {np.std(n_stars):.2f}\n")
                f.write(f"  Range: [{np.min(n_stars)}, {np.max(n_stars)}]\n")
            f.write(f"\nBarrier height statistics:\n")
            delta_phis = [r.delta_phi for r in results_list if not np.isnan(r.delta_phi)]
            if delta_phis:
                f.write(f"  Mean: {np.mean(delta_phis):.2f}\n")
                f.write(f"  Std: {np.std(delta_phis):.2f}\n")
                f.write(f"  Range: [{np.min(delta_phis):.2f}, {np.max(delta_phis):.2f}]\n")
            f.write(f"\nMFPT statistics (seconds):\n")
            mfpts = [r.mfpt_1d / r.k_wrap for r in results_list]
            f.write(f"  Mean: {np.mean(mfpts):.3e}\n")
            f.write(f"  Std: {np.std(mfpts):.3e}\n")
            f.write(f"  Range: [{np.min(mfpts):.3e}, {np.max(mfpts):.3e}]\n")
        print(f"  Saved summary: {summary_file}")
    
    return results_list


def main():
    """Main execution function."""
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Output directory
    output_base_dir = Path(__file__).parent.parent.parent / "output" / "drift_parameter_scan"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset paths
    datasets = {
        'bound': HAMNUCRET_DATA_DIR / "exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv",
        'unbound': HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"
    }
    
    # Parameter sets to explore
    parameter_sets = [
        ParameterSet(
            name="low_prot_low_coop",
            k_bind=1.0,
            k_unbind=89.7,
            p_conc=0.0,
            cooperativity=0.0
        ),
        ParameterSet(
            name="high_prot_low_coop",
            k_bind=1.0,
            k_unbind=89.7,
            p_conc=10.0,
            cooperativity=0.0
        ),
        ParameterSet(
            name="high_prot_high_coop",
            k_bind=1.0,
            k_unbind=89.7,
            p_conc=10.0,
            cooperativity=4.5
        ),
        # ParameterSet(
        #     name="very_high_prot",
        #     k_bind=1.0,
        #     k_unbind=89.7,
        #     p_conc=1000.0,
        #     cooperativity=0.0
        # ),
        # ParameterSet(
        #     name="very_high_prot_high_coop",
        #     k_bind=1.0,
        #     k_unbind=89.7,
        #     p_conc=500.0,
        #     cooperativity=4.5
        # )
    ]
    
    # Analysis parameters
    k_wrap = 1.0
    max_nucs = 1000  # Maximum nucleosomes per dataset
    
    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    
    print("\n" + "="*70)
    print("DRIFT REVERSAL PARAMETER SPACE ANALYSIS")
    print("="*70)
    print(f"\nDatasets: {list(datasets.keys())}")
    print(f"Parameter sets: {len(parameter_sets)}")
    print(f"Max nucleosomes per analysis: {max_nucs}")
    print(f"Output directory: {output_base_dir}")
    
    # Store all results for comparison
    all_results = {dataset_name: {} for dataset_name in datasets.keys()}
    
    # Loop over datasets and parameter sets
    for dataset_name, file_path in datasets.items():
        if not file_path.exists():
            print(f"\nWARNING: File not found: {file_path}")
            print(f"Skipping dataset: {dataset_name}")
            continue
        
        for param_set in parameter_sets:
            try:
                results_list = analyze_dataset_with_params(
                    file_path=file_path,
                    param_set=param_set,
                    dataset_name=dataset_name,
                    output_base_dir=output_base_dir,
                    k_wrap=k_wrap,
                    max_nucs=max_nucs
                )
                all_results[dataset_name][str(param_set)] = results_list
            except Exception as e:
                print(f"\nERROR analyzing {dataset_name} with {param_set.name}:")
                print(f"  {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # =========================================================================
    # COMPARISON PLOTS
    # =========================================================================
    
    plot_comparison_across_parameters(all_results, output_base_dir)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_base_dir}")
    print("\nDirectory structure:")
    print("  drift_parameter_scan/")
    print("    bound/")
    for param_set in parameter_sets:
        print(f"      {param_set}/")
        print(f"        individual/  # Individual nucleosome plots (if enabled)")
        print(f"        averaged/    # Averaged committor, quasi-potential, data")
    print("    unbound/")
    print("      ...")
    print("    comparison_bound.png")
    print("    comparison_unbound.png")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
