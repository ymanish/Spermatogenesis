"""
QSSA Visualization Functions
=============================

This module provides visualization functions for QSSA validation results:
- Epsilon heatmaps by (i,j) state
- Timescale comparison plots
- System-wide QSSA validity overview

Author: MY
Date: 2025-11-27
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, List

from .validation import QSSAValidationResult, SystemQSSAResult


def setup_matplotlib_style():
    """Configure matplotlib style for consistent plots."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'font.family': 'sans-serif',
    })


def plot_epsilon_heatmap(
    result: QSSAValidationResult,
    threshold: float = 0.1,
    save_path: Optional[Path] = None
):
    """
    Plot epsilon values as a heatmap over (i,j) state space.
    
    Args:
        result: QSSAValidationResult for a single nucleosome
        threshold: QSSA threshold line to overlay
        save_path: Path to save figure (None = display only)
        
    Notes:
        - Green indicates QSSA valid (epsilon <= threshold)
        - Red indicates QSSA invalid (epsilon > threshold)
        - Only shows valid (i,j) states (i + j <= L-1)
    """
    setup_matplotlib_style()
    
    L = 14  # Assuming binding_sites = 14
    
    # Create grid for heatmap
    epsilon_grid = np.full((L, L), np.nan)
    
    for (i, j), eps in result.epsilons.items():
        if i < L and j < L:
            epsilon_grid[i, j] = eps
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create custom colormap: green (valid) to red (invalid)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green -> yellow -> red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('qssa', colors, N=n_bins)
    
    # Plot heatmap
    vmax = max(threshold * 2, result.eps_max * 1.1)
    im = ax.imshow(epsilon_grid, origin='lower', cmap=cmap, 
                   vmin=0, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$\epsilon = \tau_{prot} / \tau_{slow}$')
    
    # Add threshold line on colorbar
    cbar.ax.axhline(threshold, color='black', linewidth=2, linestyle='--')
    cbar.ax.text(0.5, threshold, f'  threshold={threshold}', 
                 va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('j (right index)')
    ax.set_ylabel('i (left index)')
    ax.set_title(f'QSSA Epsilon Map: {result.nuc_id} (subid={result.subid})\n'
                 f'{"✓ VALID" if result.qssa_valid else "✗ INVALID"} '
                 f'(eps_max={result.eps_max:.4f})')
    
    # Grid
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved epsilon heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_system_overview(
    result: SystemQSSAResult,
    save_path: Optional[Path] = None
):
    """
    Plot system-wide QSSA validation overview.
    
    Shows:
    - Histogram of maximum epsilon values across nucleosomes
    - Fraction of nucleosomes passing QSSA
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        save_path: Path to save figure (None = display only)
    """
    setup_matplotlib_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Extract eps_max for each nucleosome
    eps_max_values = [nuc_result.eps_max for nuc_result in result.nucleosome_results]
    valid_flags = [nuc_result.qssa_valid for nuc_result in result.nucleosome_results]
    
    # Histogram of epsilon values
    ax1 = axes[0]
    ax1.hist(eps_max_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(result.nucleosome_results[0].threshold, color='red', 
                linewidth=2, linestyle='--', label=f'Threshold={result.nucleosome_results[0].threshold}')
    ax1.set_xlabel(r'Maximum $\epsilon$ per nucleosome')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Max Epsilon Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of pass/fail
    ax2 = axes[1]
    categories = ['Valid', 'Invalid']
    counts = [result.num_valid, result.num_invalid]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax2.bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
    
    # Add percentage labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / result.num_nucleosomes) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Number of Nucleosomes')
    ax2.set_title('QSSA Validation Summary')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    status = "VALID" if result.system_qssa_valid else "INVALID"
    fig.suptitle(f'System QSSA: {status} ({result.num_nucleosomes} nucleosomes)',
                 fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved system overview to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_timescale_comparison(
    result: QSSAValidationResult,
    save_path: Optional[Path] = None
):
    """
    Plot comparison of tau_prot vs tau_slow for all (i,j) states.
    
    Args:
        result: QSSAValidationResult for a single nucleosome
        save_path: Path to save figure (None = display only)
    """
    setup_matplotlib_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    n_open_values = []
    tau_slow_values = []
    epsilon_values = []
    
    L = 14  # Assuming binding_sites = 14
    for (i, j), ts in result.tau_slow.items():
        n_open = i + (L - 1 - j)
        n_open_values.append(n_open)
        tau_slow_values.append(ts)
        epsilon_values.append(result.epsilons[(i, j)])
    
    # Sort by n_open
    sorted_indices = np.argsort(n_open_values)
    n_open_values = np.array(n_open_values)[sorted_indices]
    tau_slow_values = np.array(tau_slow_values)[sorted_indices]
    epsilon_values = np.array(epsilon_values)[sorted_indices]
    
    # Color by QSSA validity
    colors = ['#2ecc71' if eps <= result.threshold else '#e74c3c' 
              for eps in epsilon_values]
    
    # Plot tau_slow
    ax.scatter(n_open_values, tau_slow_values, c=colors, s=50, 
               alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Plot tau_prot as horizontal line
    ax.axhline(result.tau_prot, color='blue', linewidth=2, 
               linestyle='--', label=r'$\tau_{prot}$', zorder=10)
    
    # Labels
    ax.set_xlabel('Number of Open Contacts (n_open)')
    ax.set_ylabel('Timescale (seconds)')
    ax.set_yscale('log')
    ax.set_title(f'Timescale Comparison: {result.nuc_id} (subid={result.subid})')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved timescale comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_visualizations(
    result: SystemQSSAResult,
    output_dir: Path,
    max_nucleosomes: int = 5
):
    """
    Generate all visualization plots and save to output directory.
    
    Args:
        result: SystemQSSAResult from validate_qssa_for_system
        output_dir: Directory to save all plots
        max_nucleosomes: Maximum number of individual nucleosome plots to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating QSSA visualization plots...")
    
    # System overview
    overview_path = output_dir / "system_overview.png"
    plot_system_overview(result, save_path=overview_path)
    
    # Individual nucleosome plots (limited to avoid too many files)
    individual_dir = output_dir / "individual_nucleosomes"
    individual_dir.mkdir(exist_ok=True)
    
    for idx, nuc_result in enumerate(result.nucleosome_results[:max_nucleosomes]):
        # Epsilon heatmap
        safe_nuc_id = nuc_result.nuc_id.replace('/', '_').replace(':', '_')
        heatmap_path = individual_dir / f"{safe_nuc_id}_subid_{nuc_result.subid}_heatmap.png"
        plot_epsilon_heatmap(nuc_result, threshold=nuc_result.threshold, save_path=heatmap_path)
        
        # Timescale comparison
        timescale_path = individual_dir / f"{safe_nuc_id}_subid_{nuc_result.subid}_timescales.png"
        plot_timescale_comparison(nuc_result, save_path=timescale_path)
    
    if len(result.nucleosome_results) > max_nucleosomes:
        print(f"  (Showing plots for first {max_nucleosomes} nucleosomes only)")
    
    print(f"✓ All visualization plots saved to {output_dir}")
