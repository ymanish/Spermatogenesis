"""
Visualization Functions
=======================

Generate publication-quality figures for RMST analysis.

Author: MY
Date: 2025-11-14
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from src.config.custom_type import RMSTAnalysis

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mpl = None


def plot_rmst_analysis(
    rmst_data: Dict[str, List[float]],
    analysis: RMSTAnalysis,
    save_path: Optional[Path] = None
) -> None:
    """
    Create visualization of RMST analysis results.
    
    Generates two plots:
    1. Per-nucleosome RMST variability (error bars)
    2. Distribution of all RMST values (histogram)
    
    Args:
        rmst_data: Dict mapping nucleosome_key -> [RMST values]
        analysis: RMSTAnalysis object with statistics
        save_path: Optional path to save figure (directory)
    
    Examples:
        >>> plot_rmst_analysis(rmst_data, analysis, Path("output"))
        ✓ Saved figure to output/rmst_plot_RET_prot100_coop4.5.png
    
    Notes:
        - Requires matplotlib
        - Saves as PNG with 300 DPI
        - Gracefully handles missing matplotlib
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available. Skipping plots.")
        return
    
    # Setup matplotlib style
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Per-nucleosome RMST variability
    ax = axes[0]
    n_nucs = len(analysis.nucleosome_mean_rmsts)
    x = np.arange(n_nucs)
    means = np.array(analysis.nucleosome_mean_rmsts)
    stds = np.array(analysis.nucleosome_std_rmsts)
    
    ax.errorbar(
        x, means, yerr=stds, 
        fmt='o', capsize=4, alpha=0.7,
        color='tab:blue', ecolor='tab:gray', markersize=5
    )
    
    overall_mean = analysis.mean_rmst
    ax.axhline(
        overall_mean, color='red', linestyle='--',
        label=f'Overall mean: {overall_mean:.3f}', linewidth=2
    )
    ax.fill_between(
        range(n_nucs),
        overall_mean - np.sqrt(analysis.sigma_between_sq),
        overall_mean + np.sqrt(analysis.sigma_between_sq),
        alpha=0.2, color='red',
        label=f'±σ_between: {np.sqrt(analysis.sigma_between_sq):.3f}'
    )
    
    ax.set_xlabel('Nucleosome Index')
    ax.set_ylabel('RMST (dimensionless τ)')
    ax.set_title('Per-Nucleosome RMST Variability', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Distribution of all RMST values
    ax = axes[1]
    all_rmst = []
    for values in rmst_data.values():
        all_rmst.extend(values)
    all_rmst = np.array(all_rmst)
    
    ax.hist(all_rmst, bins=30, alpha=0.7, color='tab:blue', edgecolor='black')
    ax.axvline(
        analysis.mean_rmst, color='red', linestyle='--',
        label=f'Mean: {analysis.mean_rmst:.3f}', linewidth=2
    )
    ax.axvline(
        analysis.mean_rmst - analysis.std_rmst, color='orange',
        linestyle=':', label=f'±1 SD: {analysis.std_rmst:.3f}',
        linewidth=1.5, alpha=0.7
    )
    ax.axvline(
        analysis.mean_rmst + analysis.std_rmst, color='orange',
        linestyle=':', linewidth=1.5, alpha=0.7
    )
    
    ax.set_xlabel('RMST (dimensionless τ)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of All RMST Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"rmst_plot_{analysis.condition_label.replace(' ', '_')}.png"
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path / filename}")
    
    plt.show()
