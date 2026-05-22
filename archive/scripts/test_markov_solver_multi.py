#!/usr/bin/env python3
"""
Multi-nucleosome test script for the modular markov_solver package.

This script tests the markov_solver module on multiple nucleosomes and computes:
- Q_TT generator matrix with protamine effects (fast limit) for each nucleosome
- Mean First Passage Time (MFPT) to nucleosome detachment
- Survival function S(t)
- Statistical analysis across all nucleosomes (mean, std, median)
- Average survival curves with error bands

The module uses the fast protamine approximation:
- Opening rates: k_open = k_wrap * exp(-ΔF_nuc/kT)
- Closing rates: k_close_eff = k_wrap * P_free(n_open; β*μ, β*J)

Author: Test Script
Date: 2025-12-11
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from the modular markov_solver package
from src.analysis.markov_solver import (
    load_nucleosomes_from_file,
    solve_Q_TT_complete,
)
from src.config.path import HAMNUCRET_DATA_DIR


def main():
    """Main test function for multi-nucleosome analysis."""
    print("=" * 80)
    print("Multi-Nucleosome Markov Solver Analysis")
    print("=" * 80)
    
    # ========================================================================
    # SETUP: Parameters and configuration
    # ========================================================================
    print("\n[1] Configuration...")
    
    k_wrap = 1.0  # Wrapping rate in 1/s
    protamine_params = {
        'k_bind': 1.0,          # Binding rate (1/(μM·s))
        'k_unbind': 89.7,       # Unbinding rate (1/s)
        'p_conc': 0.0,       # Protamine concentration (μM)
        'cooperativity': 0.0    # Cooperativity parameter J (kT)
    }
    tau_max = 10000.0  # Dimensionless time τ = k_wrap * t_phys
    n_points = 500
    tau_grid = np.linspace(0, tau_max, n_points)
    
    # Configure multi-nucleosome testing
    num_nucleosomes = 20  # Number of nucleosomes to test
    start_state = (0, 0)   # Initial state (fully wrapped)
    compare_methods = False  # Compare ODE and expm methods
    
    # Compute dimensionless parameters
    mu_tilde = protamine_params['p_conc'] * protamine_params['k_bind'] / protamine_params['k_unbind']
    beta_mu = np.log(mu_tilde)
    beta_J = protamine_params['cooperativity']
    
    print(f"   Number of nucleosomes: {num_nucleosomes}")
    print(f"   Compare ODE vs expm: {compare_methods}")
    print(f"   Time grid: {n_points} points from 0 to {tau_max} (τ)")
    print(f"\n   Protamine parameters:")
    print(f"     k_bind      = {protamine_params['k_bind']:.1f} (μM·s)⁻¹")
    print(f"     k_unbind    = {protamine_params['k_unbind']:.1f} s⁻¹")
    print(f"     p_conc      = {protamine_params['p_conc']:.1f} μM")
    print(f"     J           = {protamine_params['cooperativity']:.1f} kT")
    print(f"     μ̃           = {mu_tilde:.2f}")
    print(f"     β*μ̃         = {beta_mu:.3f}")
    print(f"     β*J         = {beta_J:.3f}")
    
    # ========================================================================
    # LOAD NUCLEOSOMES
    # ========================================================================
    print("\n[2] Loading nucleosome data...")
    
    file_path = HAMNUCRET_DATA_DIR / "exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
    print(f"   Loading from: {file_path}")
    
    nucs = load_nucleosomes_from_file(
        file_path, 
        k_wrap=k_wrap, 
        max_nucs=num_nucleosomes,
    )
    
    print(f"   ✓ Loaded {len(nucs)} nucleosomes")
    
    # ========================================================================
    # COMPUTE RESULTS FOR ALL NUCLEOSOMES
    # ========================================================================
    print(f"\n[3] Computing MFPT and survival for {len(nucs)} nucleosomes...")
    
    all_mfpts = []
    all_mfpts_physical = []
    all_survivals_ode = []
    all_survivals_expm = []
    all_half_lives_ode = []
    all_half_lives_expm = []
    all_nuc_ids = []
    
    # Progress bar for processing
    for nuc in tqdm(nucs, desc="   Processing nucleosomes", unit="nuc"):
        try:
            # Compute with ODE method
            results_ode = solve_Q_TT_complete(
                nuc,
                start_state=start_state,
                tau_max=tau_max,
                n_points=n_points,
                method='ode',
                sparse=False,
                k_wrap=k_wrap,
                protamine_params=protamine_params
            )
            
            # Store ODE results
            mfpt = results_ode['mfpt']
            print(f"   Nucleosome {nuc.id}-{nuc.subid}: MFPT (dimensionless) = {mfpt:.3f}") 
            mfpt_physical = mfpt / k_wrap
            survival_ode = results_ode['survival']
            
            all_mfpts.append(mfpt)
            all_mfpts_physical.append(mfpt_physical)
            all_survivals_ode.append(survival_ode)
            all_nuc_ids.append(f"{nuc.id}-{nuc.subid}")
            
            # Compute half-life (ODE) – only if curve ever drops below 0.5
            if np.any(survival_ode <= 0.5):
                idx_half = np.argmin(np.abs(survival_ode - 0.5))
                tau_half = tau_grid[idx_half]
                print(f"      Half-life index: {idx_half}, τ_half = {tau_half:.3f}")
                print(f"      Survival_ode = {survival_ode[idx_half]:.3f} at τ = {tau_half:.3f}")
                all_half_lives_ode.append(tau_half / k_wrap)
            else:
                print("      Half-life not reached (S(t) never <= 0.5)")
                all_half_lives_ode.append(np.inf)  # or skip appending, depending on what you prefer
                        
            # If comparing methods, also compute with expm
            if compare_methods:
                results_expm = solve_Q_TT_complete(
                    nuc,
                    start_state=start_state,
                    tau_max=tau_max,
                    n_points=n_points,
                    method='expm',
                    sparse=False,
                    k_wrap=k_wrap,
                    protamine_params=protamine_params
                )
                
                survival_expm = results_expm['survival']
                all_survivals_expm.append(survival_expm)
                
                # Compute half-life (expm)
                idx_half_expm = np.argmin(np.abs(survival_expm - 0.5))
                tau_half_expm = tau_grid[idx_half_expm]
                all_half_lives_expm.append(tau_half_expm / k_wrap)
            
        except Exception as e:
            print(f"\n   Warning: Failed for nucleosome {nuc.id}-{nuc.subid}: {e}")
            continue
    
    # Convert to arrays
    all_mfpts = np.array(all_mfpts)
    all_mfpts_physical = np.array(all_mfpts_physical)
    all_survivals_ode = np.array(all_survivals_ode)  # Shape: (n_nucs, n_points)
    all_half_lives_ode = np.array(all_half_lives_ode)

    # Keep only finite half-lives for stats/plots
    hl_finite_mask = np.isfinite(all_half_lives_ode)
    half_lives_finite_ode = all_half_lives_ode[hl_finite_mask]

    n_hl_total = all_half_lives_ode.size
    n_hl_finite = half_lives_finite_ode.size
    print(f"Finite half-lives (ODE) used in stats/hist: {n_hl_finite}/{n_hl_total}")
        
    if compare_methods:
        all_survivals_expm = np.array(all_survivals_expm)
        all_half_lives_expm = np.array(all_half_lives_expm)

            # Keep only finite half-lives for stats/plots
        hl_finite_mask = np.isfinite(all_half_lives_expm)
        half_lives_finite_expm = all_half_lives_expm[hl_finite_mask]

        n_hl_total = all_half_lives_expm.size
        n_hl_finite = half_lives_finite_expm.size
        print(f"Finite half-lives (expm) used in stats/hist: {n_hl_finite}/{n_hl_total}")

    print(f"\n   ✓ Successfully processed {len(all_mfpts)} nucleosomes")
    
    # ========================================================================
    # COMPUTE STATISTICS
    # ========================================================================
    print("\n[4] Computing statistics...")
    
    # MFPT statistics
    mfpt_mean = np.mean(all_mfpts_physical)
    mfpt_std = np.std(all_mfpts_physical)
    mfpt_median = np.median(all_mfpts_physical)
    mfpt_min = np.min(all_mfpts_physical)
    mfpt_max = np.max(all_mfpts_physical)
    
    print(f"   MFPT Statistics (physical time):")
    print(f"     Mean   : {mfpt_mean:.4f} ± {mfpt_std:.4f} s")
    print(f"     Median : {mfpt_median:.4f} s")
    print(f"     Range  : [{mfpt_min:.4f}, {mfpt_max:.4f}] s")
    print(f"     CV     : {mfpt_std/mfpt_mean:.2%}")
        
    # Half-life statistics (ODE)
    if half_lives_finite_ode.size > 0:
        hl_mean_ode = np.mean(half_lives_finite_ode)
        hl_std_ode = np.std(half_lives_finite_ode)
        hl_median_ode = np.median(half_lives_finite_ode)

        print(f"\n   Half-life Statistics (ODE) [finite only]:")
        print(f"     Mean   : {hl_mean_ode:.4f} ± {hl_std_ode:.4f} s")
        print(f"     Median : {hl_median_ode:.4f} s")
        print(f"     Range  : [{np.min(half_lives_finite_ode):.4f}, {np.max(half_lives_finite_ode):.4f}] s")
    else:
        hl_mean_ode = np.nan
        hl_std_ode = np.nan
        hl_median_ode = np.nan
        print("\n   Half-life Statistics (ODE): no finite half-lives (all inf/NaN).")
        
    # Survival curve statistics (ODE)
    survival_mean_ode = np.mean(all_survivals_ode, axis=0)
    survival_std_ode = np.std(all_survivals_ode, axis=0)
    survival_median_ode = np.median(all_survivals_ode, axis=0)
    survival_q25_ode = np.percentile(all_survivals_ode, 25, axis=0)
    survival_q75_ode = np.percentile(all_survivals_ode, 75, axis=0)
    
    print(f"\n   Survival Curve Statistics (ODE):")
    print(f"     Computed mean, std, median, and quartiles across {len(nucs)} nucleosomes")
    
    # Method comparison statistics
    if compare_methods:
           # Half-life statistics (EXPM)
        if half_lives_finite_expm.size > 0:
            hl_mean_expm = np.mean(half_lives_finite_expm)
            hl_std_expm = np.std(half_lives_finite_expm)
            hl_median_expm = np.median(half_lives_finite_expm)

            print(f"\n   Half-life Statistics (EXPM) [finite only]:")
            print(f"     Mean   : {hl_mean_expm:.4f} ± {hl_std_expm:.4f} s")
            print(f"     Median : {hl_median_expm:.4f} s")
            print(f"     Range  : [{np.min(half_lives_finite_expm):.4f}, {np.max(half_lives_finite_expm):.4f}] s")
        else:
            hl_mean_expm = np.nan
            hl_std_expm = np.nan
            hl_median_expm = np.nan
            print("\n   Half-life Statistics (EXPM): no finite half-lives (all inf/NaN).")

        # Survival curve statistics (expm)
        survival_mean_expm = np.mean(all_survivals_expm, axis=0)
        survival_std_expm = np.std(all_survivals_expm, axis=0)
        
        # Method comparison
        max_diff_per_nuc = np.max(np.abs(all_survivals_ode - all_survivals_expm), axis=1)
        mean_max_diff = np.mean(max_diff_per_nuc)
        
        print(f"\n   Method Comparison (ODE vs expm):")
        print(f"     Mean of max |diff| per nucleosome: {mean_max_diff:.2e}")
        print(f"     Range of max |diff|: [{np.min(max_diff_per_nuc):.2e}, {np.max(max_diff_per_nuc):.2e}]")
        print(f"     Half-life difference (mean): {abs(hl_mean_ode - hl_mean_expm):.4f} s")
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    print("\n[5] Generating plots...")
    
    t_physical = tau_grid / k_wrap
    
    # Adjust figure layout based on whether we're comparing methods
    if compare_methods:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    all_mfpts_physical = np.array(all_mfpts_physical, dtype=float)

    # Drop non-finite values (inf, -inf, nan) for histogram
    finite_mask = np.isfinite(all_mfpts_physical)
    mfpts_finite = all_mfpts_physical[finite_mask]
     # Optionally, report how many were dropped
    n_total = all_mfpts_physical.size
    n_finite = mfpts_finite.size
    print(f"Finite MFPTs used in histogram: {n_finite}/{n_total}")

    # --- Plot 1: MFPT Distribution (Histogram) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(mfpts_finite, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(mfpt_mean, color='red', linestyle='--', lw=2, label=f'Mean = {mfpt_mean:.3f}s')
    ax1.axvline(mfpt_median, color='orange', linestyle=':', lw=2, label=f'Median = {mfpt_median:.3f}s')
    ax1.set_xlabel('MFPT (seconds)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('MFPT Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: MFPT Box Plot ---
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(mfpts_finite, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    ax2.set_ylabel('MFPT (seconds)', fontsize=11)
    ax2.set_title('MFPT Box Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"n = {len(mfpts_finite)}\n"
    stats_text += f"μ = {mfpt_mean:.3f}s\n"
    stats_text += f"σ = {mfpt_std:.3f}s\n"
    stats_text += f"CV = {mfpt_std/mfpt_mean:.1%}"
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # --- Plot 3: Half-life Distribution ---
    ax3 = fig.add_subplot(gs[0, 2])

    if half_lives_finite_ode.size > 0:
        ax3.hist(half_lives_finite_ode, bins=20, alpha=0.7,
                color='coral', edgecolor='black', label='ODE')
        if np.isfinite(hl_mean_ode):
            ax3.axvline(hl_mean_ode, color='red', linestyle='--', lw=2,
                        label=f'Mean (ODE) = {hl_mean_ode:.3f}s')
    else:
        ax3.text(0.5, 0.5, "No finite half-lives\n(all = ∞ or NaN)",
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

    if compare_methods:
        if half_lives_finite_expm.size > 0:
            ax3.hist(half_lives_finite_expm, bins=20, alpha=0.7,
                     color='lightblue', edgecolor='black', label='expm')
            if np.isfinite(hl_mean_expm):
                ax3.axvline(hl_mean_expm, color='blue', linestyle='--', lw=2,
                            label=f'Mean (expm) = {hl_mean_expm:.3f}s')
        else:
            ax3.text(0.5, 0.5, "No finite half-lives\n(all = ∞ or NaN)",
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

    ax3.set_xlabel('Half-life (seconds)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Half-life Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Average Survival Curve with Error Bands (Linear) ---
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Plot individual curves (faint) - ODE only
    for i, surv in enumerate(all_survivals_ode[:10]):  # Plot first 10 for visibility
        ax4.plot(t_physical, surv, lw=0.5, alpha=0.2, color='gray')
    
    # Plot mean with std error band (ODE)
    ax4.plot(t_physical, survival_mean_ode, lw=3, color='steelblue', 
            label=f'Mean ODE (n={len(nucs)})', zorder=10)
    ax4.fill_between(t_physical, 
                     survival_mean_ode - survival_std_ode, 
                     survival_mean_ode + survival_std_ode,
                     alpha=0.3, color='steelblue', label='± 1 σ (ODE)')
    
    # Plot median with quartile band (ODE)
    ax4.plot(t_physical, survival_median_ode, lw=2, color='orange', 
            linestyle='--', label='Median ODE', zorder=9)
    ax4.fill_between(t_physical, survival_q25_ode, survival_q75_ode,
                     alpha=0.2, color='orange', label='IQR (ODE)')
    
    # If comparing methods, overlay expm mean
    if compare_methods:
        ax4.plot(t_physical, survival_mean_expm, lw=2.5, color='darkgreen', 
                linestyle='-.', label=f'Mean expm', zorder=8, alpha=0.8)
    
    ax4.axhline(0.5, color='gray', linestyle=':', lw=1, alpha=0.5)
    ax4.axvline(mfpt_mean, color='darkred', linestyle=':', lw=2, 
               label=f'Mean MFPT = {mfpt_mean:.3f}s')
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('Survival Probability S(t)', fontsize=11)
    ax4.set_title('Average Survival Function (Linear Scale)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.set_ylim([-0.05, 1.05])
    
    # Add parameter info
    param_text = f"n = {len(nucs)} nucleosomes\n"
    param_text += f"k_wrap = {k_wrap:.1f} s⁻¹\n"
    param_text += f"P_conc = {protamine_params['p_conc']:.0f} μM\n"
    param_text += f"J = {protamine_params['cooperativity']:.1f} kT\n"
    param_text += f"μ̃ = {mu_tilde:.2f}"
    ax4.text(0.02, 0.02, param_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # --- Plot 5: Average Survival Curve (Log Scale) ---
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Plot mean with error band (log scale) - ODE
    ax5.semilogy(t_physical, survival_mean_ode + 1e-10, lw=3, color='steelblue', 
                label=f'Mean ODE', zorder=10)
    ax5.fill_between(t_physical, 
                     survival_mean_ode - survival_std_ode + 1e-10, 
                     survival_mean_ode + survival_std_ode + 1e-10,
                     alpha=0.3, color='steelblue')
    
    # If comparing methods, overlay expm
    if compare_methods:
        ax5.semilogy(t_physical, survival_mean_expm + 1e-10, lw=2.5, color='darkgreen',
                    linestyle='-.', label=f'Mean expm', zorder=8, alpha=0.8)
    
    ax5.axvline(mfpt_mean, color='darkred', linestyle=':', lw=2, alpha=0.7)
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('S(t) (log scale)', fontsize=11)
    ax5.set_title('Average Survival (Log Scale)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=9)
    
    # --- Plot 6: MFPT vs Nucleosome Index ---
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(range(len(all_mfpts_physical)), all_mfpts_physical, 'o-', 
            markersize=5, alpha=0.6, color='steelblue')
    ax6.axhline(mfpt_mean, color='red', linestyle='--', lw=2, 
               label=f'Mean = {mfpt_mean:.3f}s')
    ax6.fill_between(range(len(all_mfpts_physical)),
                     [mfpt_mean - mfpt_std] * len(all_mfpts_physical),
                     [mfpt_mean + mfpt_std] * len(all_mfpts_physical),
                     alpha=0.2, color='red')
    ax6.set_xlabel('Nucleosome Index', fontsize=11)
    ax6.set_ylabel('MFPT (seconds)', fontsize=11)
    ax6.set_title('MFPT per Nucleosome', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # --- Plot 7: Coefficient of Variation over Time ---
    ax7 = fig.add_subplot(gs[2, 1])
    cv_over_time = survival_std_ode / (survival_mean_ode + 1e-10)
    ax7.plot(t_physical, cv_over_time, lw=2, color='purple')
    ax7.set_xlabel('Time (seconds)', fontsize=11)
    ax7.set_ylabel('CV = σ/μ', fontsize=11)
    ax7.set_title('Coefficient of Variation over Time', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='CV = 1')
    ax7.legend(fontsize=9)
    
    # --- Plot 8: Cumulative Distribution of MFPTs ---
    ax8 = fig.add_subplot(gs[2, 2])
    sorted_mfpts = np.sort(all_mfpts_physical)
    cumulative = np.arange(1, len(sorted_mfpts) + 1) / len(sorted_mfpts)
    ax8.plot(sorted_mfpts, cumulative, lw=2, color='steelblue')
    ax8.axvline(mfpt_mean, color='red', linestyle='--', lw=2, label='Mean')
    ax8.axvline(mfpt_median, color='orange', linestyle=':', lw=2, label='Median')
    ax8.axhline(0.5, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax8.set_xlabel('MFPT (seconds)', fontsize=11)
    ax8.set_ylabel('Cumulative Probability', fontsize=11)
    ax8.set_title('MFPT Cumulative Distribution', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)
    
    # --- Additional plots for method comparison ---
    if compare_methods:
        # Plot 9: Method Difference over Time (average)
        ax9 = fig.add_subplot(gs[3, 0])
        diff_mean = survival_mean_ode - survival_mean_expm
        ax9.plot(t_physical, diff_mean, lw=2, color='purple')
        ax9.axhline(0, color='black', linestyle='-', lw=0.5)
        ax9.set_xlabel('Time (seconds)', fontsize=11)
        ax9.set_ylabel('S_ODE - S_expm', fontsize=11)
        ax9.set_title('Average Method Difference', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Max |diff| = {np.max(np.abs(diff_mean)):.2e}\n"
        stats_text += f"Mean |diff| = {np.mean(np.abs(diff_mean)):.2e}\n"
        stats_text += f"RMS diff = {np.sqrt(np.mean(diff_mean**2)):.2e}"
        ax9.text(0.98, 0.98, stats_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Plot 10: Distribution of max differences per nucleosome
        ax10 = fig.add_subplot(gs[3, 1])
        ax10.hist(max_diff_per_nuc, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax10.axvline(mean_max_diff, color='red', linestyle='--', lw=2, 
                    label=f'Mean = {mean_max_diff:.2e}')
        ax10.set_xlabel('Max |S_ODE - S_expm|', fontsize=11)
        ax10.set_ylabel('Count', fontsize=11)
        ax10.set_title('Max Difference per Nucleosome', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=9)
        ax10.grid(True, alpha=0.3)
        
        # Plot 11: Method comparison for a single nucleosome (example)
        ax11 = fig.add_subplot(gs[3, 2])
        idx_example = 0  # First nucleosome
        ax11.plot(t_physical, all_survivals_ode[idx_example], lw=2, 
                 color='steelblue', label='ODE', alpha=0.8)
        ax11.plot(t_physical, all_survivals_expm[idx_example], lw=2, 
                 linestyle='--', color='coral', label='expm', alpha=0.8)
        diff_example = all_survivals_ode[idx_example] - all_survivals_expm[idx_example]
        ax11_twin = ax11.twinx()
        ax11_twin.plot(t_physical, diff_example, lw=1.5, color='purple', 
                      linestyle=':', label='Difference', alpha=0.6)
        ax11_twin.axhline(0, color='black', linestyle='-', lw=0.5, alpha=0.3)
        ax11_twin.set_ylabel('Difference', fontsize=10, color='purple')
        ax11_twin.tick_params(axis='y', labelcolor='purple')
        ax11.set_xlabel('Time (seconds)', fontsize=11)
        ax11.set_ylabel('Survival S(t)', fontsize=11)
        ax11.set_title(f'Example: {all_nuc_ids[idx_example]}', fontsize=12, fontweight='bold')
        ax11.legend(loc='upper right', fontsize=9)
        ax11.grid(True, alpha=0.3)
        
        # Add max diff text
        max_diff_example = np.max(np.abs(diff_example))
        ax11.text(0.02, 0.02, f'Max |diff| = {max_diff_example:.2e}',
                 transform=ax11.transAxes, fontsize=9,
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plot_title = f'Multi-Nucleosome Markov Solver Analysis (n={len(nucs)})'
    if compare_methods:
        plot_title += ' - ODE vs expm Comparison'
    plt.suptitle(plot_title, fontsize=14, fontweight='bold', y=0.998)
    
    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'test_markov_solver_multi_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {output_path}")
    
    plt.show()
    
    # ========================================================================
    # DETAILED SUMMARY TABLE
    # ========================================================================
    print("\n[6] Detailed Results Table (first 10 nucleosomes):")
    print("   " + "-" * 90)
    if compare_methods:
        print(f"   {'Nucleosome':<15} {'MFPT (s)':<12} {'HL ODE (s)':<12} {'HL expm (s)':<12} {'S_ode(t_max)':<10}")
    else:
        print(f"   {'Nucleosome':<15} {'MFPT (s)':<12} {'Half-life (s)':<15} {'S(t_max)':<10}")
    print("   " + "-" * 90)
    for i in range(min(10, len(all_nuc_ids))):
        if compare_methods:
            print(f"   {all_nuc_ids[i]:<15} {all_mfpts_physical[i]:>11.4f} "
                  f"{all_half_lives_ode[i]:>11.4f} {all_half_lives_expm[i]:>11.4f} "
                  f"{all_survivals_ode[i, -1]:>9.3e}")
        else:
            print(f"   {all_nuc_ids[i]:<15} {all_mfpts_physical[i]:>11.4f} "
                  f"{all_half_lives_ode[i]:>14.4f} {all_survivals_ode[i, -1]:>9.3e}")
    print("   " + "-" * 90)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("MULTI-NUCLEOSOME ANALYSIS COMPLETE! ✓")
    print("=" * 80)
    print(f"\nSummary for {len(nucs)} nucleosomes:")
    print(f"  • MFPT: {mfpt_mean:.4f} ± {mfpt_std:.4f} seconds")
    print(f"  • Median MFPT: {mfpt_median:.4f} seconds")
    print(f"  • MFPT range: [{mfpt_min:.4f}, {mfpt_max:.4f}] seconds")
    print(f"  • Coefficient of variation: {mfpt_std/mfpt_mean:.2%}")
    print(f"  • Half-life (ODE): {hl_mean_ode:.4f} ± {hl_std_ode:.4f} seconds")
    if compare_methods:
        print(f"  • Half-life (expm): {hl_mean_expm:.4f} ± {hl_std_expm:.4f} seconds")
        print(f"  • Method agreement (avg max |diff|): {mean_max_diff:.2e}")
    print(f"  • Protamine: {protamine_params['p_conc']:.0f} μM, J = {protamine_params['cooperativity']:.1f} kT")
    if compare_methods:
        print(f"  • Methods compared: ODE vs expm")
    else:
        print(f"  • Method: ODE")
    print("=" * 80)


if __name__ == "__main__":
    main()
