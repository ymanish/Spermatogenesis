"""
Visualization utilities for drift reversal analysis.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .drift_reversal import DriftReversalResults


def plot_drift_and_rates(
    results: DriftReversalResults,
    ax=None,
    show_rates: bool = True,
    show_critical: bool = True,
    dimensionless: bool = True
) -> plt.Axes:
    """
    Plot drift v(n) and optionally the underlying rates k+(n), k-(n).
    
    Args:
        results: DriftReversalResults from analyzer
        ax: Matplotlib axes (creates new if None)
        show_rates: Plot k+ and k- in addition to drift
        show_critical: Mark critical nucleus n*
        dimensionless: If True, plot in dimensionless units (multiply by k_wrap for physical)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    n = results.n_values
    scale = 1.0 if dimensionless else results.k_wrap
    
    if show_rates:
        ax.plot(n, results.k_plus * scale, 'o-', label='$k_+(n)$', alpha=0.7)
        ax.plot(n, results.k_minus * scale, 's-', label='$k_-(n)$', alpha=0.7)
    
    ax.plot(n, results.drift * scale, 'D-', label='$v(n) = k_+ - k_-$', 
            linewidth=2, color='darkred')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    if show_critical and results.n_star is not None:
        ax.axvline(results.n_star, color='red', linestyle=':', alpha=0.7,
                  label=f'$n^* = {results.n_star}$')
        if results.n_star_refined is not None:
            ax.axvline(results.n_star_refined, color='red', linestyle='--', 
                      alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('$n$ (total open sites)')
    ylabel = 'Rate (dimensionless)' if dimensionless else 'Rate (s$^{-1}$)'
    ax.set_ylabel(ylabel)
    ax.set_title('Drift reversal analysis')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_committor(
    results: DriftReversalResults,
    ax=None,
    show_critical: bool = True
) -> plt.Axes:
    """
    Plot committor function q(n) - probability to detach (reach N) before fully wrapping (reach 0).
    
    Args:
        results: DriftReversalResults from analyzer
        ax: Matplotlib axes (creates new if None)
        show_critical: Mark critical nucleus n*
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    n = results.n_values
    q = results.committor
    
    ax.plot(n, q, 'o-', linewidth=2, color='purple', markersize=4)
    
    if show_critical and results.n_star is not None:
        ax.axvline(results.n_star, color='red', linestyle=':', alpha=0.7,
                  label=f'$n^* = {results.n_star}$')
        # Mark committor value at n*
        q_star = q[results.n_star] if results.n_star < len(q) else None
        if q_star is not None:
            ax.plot(results.n_star, q_star, 'r*', markersize=15, 
                   label=f'$q(n^*) = {q_star:.3f}$')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='$q = 0.5$')
    ax.set_xlabel('Reduced coordinate $n = l + r$', fontsize=11)
    ax.set_ylabel('Committor $q(n)$', fontsize=11)
    ax.set_title('Probability to Detach vs. Rewrap', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    return ax


def plot_quasi_potential(
    results: DriftReversalResults,
    ax=None,
    show_critical: bool = True,
    show_barrier: bool = True
) -> plt.Axes:
    """
    Plot quasi-potential Φ(n) showing the barrier landscape.
    
    Args:
        results: DriftReversalResults from analyzer
        ax: Matplotlib axes (creates new if None)
        show_critical: Mark critical nucleus n*
        show_barrier: Annotate barrier height ΔΦ
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    n = results.n_values
    phi = results.phi
    
    ax.plot(n, phi, 'o-', linewidth=2, markersize=6)
    
    if show_critical and results.n_star is not None:
        ax.axvline(results.n_star, color='red', linestyle=':', alpha=0.7,
                  label=f'$n^* = {results.n_star}$')
        ax.plot(results.n_star, phi[results.n_star], 'r*', markersize=15,
               label=f'Barrier top')
    
    if show_barrier and results.n_star is not None:
        # Annotate barrier
        ax.annotate('', xy=(results.n_star, phi[results.n_star]),
                   xytext=(0, phi[0]),
                   arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
        mid_n = results.n_star / 2
        mid_phi = (phi[0] + phi[results.n_star]) / 2
        ax.text(mid_n, mid_phi, f'$\\Delta\\Phi = {results.delta_phi:.2f}$',
               fontsize=12, color='darkgreen', ha='right')
    
    ax.set_xlabel('$n$ (total open sites)')
    ax.set_ylabel('$\\Phi(n)$ (quasi-potential)')
    ax.set_title('Free energy landscape')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_survival_comparison(
    results: DriftReversalResults,
    t_grid: np.ndarray,
    ssa_survival: Optional[np.ndarray] = None,
    ax=None,
    log_scale: bool = False
) -> plt.Axes:
    """
    Plot 1D survival curve and optionally compare to SSA.
    
    Args:
        results: DriftReversalResults from analyzer
        t_grid: Time points
        ssa_survival: SSA survival curve (optional)
        ax: Matplotlib axes (creates new if None)
        log_scale: Use log scale for y-axis
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Compute 1D survival
    from .drift_reversal import DriftReversalAnalyzer
    analyzer = DriftReversalAnalyzer(
        nucleosome=None,
        k_wrap=results.k_wrap,
        protamine_params=results.protamine_params
    )
    survival_1d = analyzer.compute_1d_survival(
        results.k_plus, results.k_minus, t_grid
    )
    
    # Convert time to physical units
    t_phys = t_grid / results.k_wrap
    
    if log_scale:
        ax.semilogy(t_phys, survival_1d + 1e-12, '-', label='1D model', linewidth=2)
        if ssa_survival is not None:
            ax.semilogy(t_phys, ssa_survival + 1e-12, '--', label='SSA', linewidth=2, alpha=0.7)
    else:
        ax.plot(t_phys, survival_1d, '-', label='1D model', linewidth=2)
        if ssa_survival is not None:
            ax.plot(t_phys, ssa_survival, '--', label='SSA', linewidth=2, alpha=0.7)
    
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ylabel = '$S(t)$ (log scale)' if log_scale else '$S(t)$'
    ax.set_ylabel(ylabel)
    ax.set_title('Survival probability')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_full_analysis(
    results: DriftReversalResults,
    t_grid: Optional[np.ndarray] = None,
    ssa_survival: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (14, 10)
) -> plt.Figure:
    """
    Create comprehensive 4-panel figure with drift reversal analysis.
    
    Panels:
        1. Rates k+(n), k-(n) and drift v(n)
        2. Quasi-potential Φ(n)
        3. Survival curves (linear scale)
        4. Survival curves (log scale)
    
    Args:
        results: DriftReversalResults from analyzer
        t_grid: Time points for survival (optional)
        ssa_survival: SSA survival for comparison (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Drift and rates
    ax1 = fig.add_subplot(gs[0, 0])
    plot_drift_and_rates(results, ax=ax1)
    
    # Panel 2: Quasi-potential
    ax2 = fig.add_subplot(gs[0, 1])
    plot_quasi_potential(results, ax=ax2)
    
    # Panels 3-4: Survival curves (if t_grid provided)
    if t_grid is not None:
        ax3 = fig.add_subplot(gs[1, 0])
        plot_survival_comparison(results, t_grid, ssa_survival, ax=ax3, log_scale=False)
        
        ax4 = fig.add_subplot(gs[1, 1])
        plot_survival_comparison(results, t_grid, ssa_survival, ax=ax4, log_scale=True)
    else:
        # Show parameter info instead
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        # Format parameter info
        info_text = "Analysis Results:\n\n"
        info_text += f"Critical nucleus: n* = {results.n_star}\n"
        info_text += f"Barrier height: ΔΦ = {results.delta_phi:.3f}\n"
        info_text += f"1D MFPT: {results.mfpt_1d:.3e} (dimensionless)\n"
        info_text += f"1D MFPT: {results.mfpt_1d / results.k_wrap:.3e} s\n"
        if results.mfpt_nucleation is not None:
            info_text += f"Nucleation MFPT: {results.mfpt_nucleation:.3e} (dimensionless)\n"
        
        info_text += f"\nParameters:\n"
        info_text += f"k_wrap = {results.k_wrap:.2f} s⁻¹\n"
        info_text += f"kT = {results.kT:.2f}\n"
        info_text += f"N = {results.N}\n"
        info_text += f"p_conc = {results.protamine_params['p_conc']:.2e}\n"
        info_text += f"cooperativity = {results.protamine_params['cooperativity']:.2f}\n"
        
        ax3.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    fig.suptitle('Drift Reversal Analysis', fontsize=14, fontweight='bold')
    
    return fig


def plot_phase_diagram(
    results_grid: Dict[Tuple[float, float], DriftReversalResults],
    quantity: str = 'n_star',
    ax=None,
    cmap: str = 'RdYlBu_r'
) -> plt.Axes:
    """
    Plot phase diagram in (μ, J) space.
    
    Args:
        results_grid: Dict mapping (p_conc, cooperativity) -> DriftReversalResults
        quantity: What to plot: 'n_star', 'delta_phi', 'log_mfpt', 'drift_at_1'
        ax: Matplotlib axes (creates new if None)
        cmap: Colormap name
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract parameter grid
    params = list(results_grid.keys())
    p_vals = sorted(set(p for p, c in params))
    c_vals = sorted(set(c for p, c in params))
    
    # Build data matrix
    data = np.zeros((len(c_vals), len(p_vals)))
    data.fill(np.nan)
    
    for i, c in enumerate(c_vals):
        for j, p in enumerate(p_vals):
            if (p, c) in results_grid:
                res = results_grid[(p, c)]
                
                if quantity == 'n_star':
                    val = res.n_star if res.n_star is not None else np.nan
                elif quantity == 'delta_phi':
                    val = res.delta_phi
                elif quantity == 'log_mfpt':
                    val = np.log10(res.mfpt_1d / res.k_wrap) if np.isfinite(res.mfpt_1d) else np.nan
                elif quantity == 'drift_at_1':
                    val = res.drift[1] if len(res.drift) > 1 else np.nan
                else:
                    val = np.nan
                
                data[i, j] = val
    
    # Plot heatmap
    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                   extent=[min(p_vals), max(p_vals), min(c_vals), max(c_vals)])
    
    plt.colorbar(im, ax=ax, label=quantity)
    
    ax.set_xlabel('Protamine concentration')
    ax.set_ylabel('Cooperativity')
    ax.set_title(f'Phase diagram: {quantity}')
    
    return ax


def plot_mfpt_vs_barrier(
    results_list: List[DriftReversalResults],
    labels: Optional[List[str]] = None,
    ax=None,
    show_nucleation: bool = True
) -> plt.Axes:
    """
    Plot log(MFPT) vs barrier height ΔΦ to test nucleation theory.
    
    Should show linear relationship: log T ~ ΔΦ
    
    Args:
        results_list: List of DriftReversalResults for different parameters
        labels: Labels for different nucleosomes/conditions
        ax: Matplotlib axes (creates new if None)
        show_nucleation: Show nucleation prediction line
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if labels is None:
        labels = [f'Condition {i}' for i in range(len(results_list))]
    
    # Collect data
    delta_phi_all = []
    log_mfpt_all = []
    
    for res, label in zip(results_list, labels):
        if np.isfinite(res.delta_phi) and np.isfinite(res.mfpt_1d):
            delta_phi = res.delta_phi
            log_mfpt = np.log(res.mfpt_1d)
            
            ax.plot(delta_phi, log_mfpt, 'o', markersize=8, label=label)
            delta_phi_all.append(delta_phi)
            log_mfpt_all.append(log_mfpt)
    
    # Fit line to show correlation
    if len(delta_phi_all) > 1:
        coeffs = np.polyfit(delta_phi_all, log_mfpt_all, 1)
        phi_fit = np.linspace(min(delta_phi_all), max(delta_phi_all), 100)
        log_mfpt_fit = np.polyval(coeffs, phi_fit)
        ax.plot(phi_fit, log_mfpt_fit, 'k--', alpha=0.5,
               label=f'Fit: slope={coeffs[0]:.2f}')
    
    ax.set_xlabel('$\\Delta\\Phi$ (barrier height)')
    ax.set_ylabel('$\\ln(T)$ (log MFPT, dimensionless)')
    ax.set_title('Nucleation theory: MFPT vs barrier')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return ax


def plot_shell_composition(
    results: DriftReversalResults,
    n_shells: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (12, 4)
) -> plt.Figure:
    """
    Visualize microstate composition within shells.
    
    Shows how (l,r) states are distributed and weighted within each n-shell.
    
    Args:
        results: DriftReversalResults from analyzer
        n_shells: Which shells to show (default: [0, n*/2, n*, N-1])
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if n_shells is None:
        n_star = results.n_star if results.n_star is not None else results.N // 2
        n_shells = [0, n_star // 2, n_star, results.N - 1]
        n_shells = [n for n in n_shells if n in results.shell_data]
    
    n_panels = len(n_shells)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    for ax, n in zip(axes, n_shells):
        data = results.shell_data[n]
        states = data['states']
        weights = data['weights']
        F_values = data['F_values']
        
        # Plot weights vs F
        l_vals = [l for l, r in states]
        r_vals = [r for l, r in states]
        
        scatter = ax.scatter(l_vals, r_vals, s=weights * 500, c=F_values,
                           cmap='coolwarm', alpha=0.7, edgecolors='k')
        
        ax.set_xlabel('$l$')
        ax.set_ylabel('$r$')
        ax.set_title(f'$n = {n}$ ({len(states)} states)')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Colorbar
        plt.colorbar(scatter, ax=ax, label='$F(l,r)$')
    
    fig.suptitle('Shell composition (size ∝ weight)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_landscape_descriptor_comparison(
    df: "pd.DataFrame",
    descriptors: Tuple[str, ...] = ('dE_firstbreath', 'dE_barrier', 'sigma_dE'),
    cohort_col: str = 'cohort',
    labels: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Strip + box comparison of landscape descriptors across cohorts.

    For each descriptor, reports a Mann-Whitney U p-value between the two
    cohort distributions and annotates it in the panel title.

    Args:
        df: DataFrame produced by ``compare_cohorts()`` or
            ``compute_cohort_descriptors()``.  Must have a ``cohort`` column
            (or the column named by *cohort_col*) and one column per descriptor.
        descriptors: Tuple of column names to plot (default: the three
            landscape descriptors).
        cohort_col: Name of the cohort grouping column.
        labels: Optional dict mapping cohort names to display labels.
        figsize: Figure size; defaults to (5 * len(descriptors), 5).

    Returns:
        Matplotlib figure with one panel per descriptor.
    """
    import pandas as pd
    from scipy.stats import mannwhitneyu

    cohorts = list(df[cohort_col].unique())
    if figsize is None:
        figsize = (5 * len(descriptors), 5)

    fig, axes = plt.subplots(1, len(descriptors), figsize=figsize)
    if len(descriptors) == 1:
        axes = [axes]

    display_labels = {c: (labels[c] if labels and c in labels else c) for c in cohorts}

    descriptor_titles = {
        'dE_firstbreath': r'$\Delta E_\mathrm{firstbreath}$',
        'dE_barrier':     r'$\Delta E_\mathrm{barrier}$',
        'sigma_dE':       r'$\sigma_{\Delta E}$',
    }

    for ax, desc in zip(axes, descriptors):
        groups = [df.loc[df[cohort_col] == c, desc].dropna().values for c in cohorts]
        tick_labels = [display_labels[c] for c in cohorts]

        # Box plot
        bp = ax.boxplot(groups, labels=tick_labels, notch=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Strip plot
        for i, (g, color) in enumerate(zip(groups, colors), 1):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(g))
            ax.scatter(i + jitter, g, alpha=0.4, s=8, color=color, zorder=3)

        # Mann-Whitney U p-value (only for two-cohort comparisons)
        if len(groups) == 2 and len(groups[0]) > 0 and len(groups[1]) > 0:
            _, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            p_str = f'p = {p:.3g}' if p >= 0.001 else f'p < 0.001'
            title = descriptor_titles.get(desc, desc) + f'\n{p_str}'
        else:
            title = descriptor_titles.get(desc, desc)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel('$k_B T$', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Energy-landscape descriptors by cohort', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig
