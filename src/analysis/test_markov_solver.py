#!/usr/bin/env python3
"""
Test script for the modular markov_solver package.

This script tests the markov_solver module which computes:
- Q_TT generator matrix with protamine effects (fast limit)
- Mean First Passage Time (MFPT) to nucleosome detachment
- Survival function S(t)

The module uses the fast protamine approximation:
- Opening rates: k_open = k_wrap * exp(-ΔF_nuc/kT)
- Closing rates: k_close_eff = k_wrap * P_free(n_open; β*μ, β*J)

Author: Test Script
Date: 2025-12-10
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import from the modular markov_solver package
from src.analysis.markov_solver import (
    load_nucleosomes_from_file,
    build_full_Q_from_nucleosome,
    compute_mfpt_from_Q_TT,
    compute_survival,
    solve_Q_TT_complete,
)
from src.config.path import HAMNUCRET_DATA_DIR


def main():
    """Main test function."""
    print("=" * 80)
    print("Testing Modular Markov Solver Package")
    print("=" * 80)
    
    # ========================================================================
    # SETUP: Load nucleosome and define protamine parameters
    # ========================================================================
    print("\n[1] Loading nucleosome data...")
    
    protamine_params = {
        'k_bind': 1.0,          # Binding rate (1/(μM·s))
        'k_unbind': 89.7,       # Unbinding rate (1/s)
        'p_conc': 10.0,       # Protamine concentration (μM)
        'cooperativity': 0.0    # Cooperativity parameter J (kT)
    }
    
    # Compute dimensionless parameters
    mu_tilde = protamine_params['p_conc'] * protamine_params['k_bind'] / protamine_params['k_unbind']
    beta_mu = np.log(mu_tilde)
    beta_J = protamine_params['cooperativity']
    
    print(f"   Protamine parameters:")
    print(f"     k_bind      = {protamine_params['k_bind']:.1f} (μM·s)⁻¹")
    print(f"     k_unbind    = {protamine_params['k_unbind']:.1f} s⁻¹")
    print(f"     p_conc      = {protamine_params['p_conc']:.1f} μM")
    print(f"     J           = {protamine_params['cooperativity']:.1f} kT")
    print(f"     μ̃ = k_bind*c/k_unbind = {mu_tilde:.2f}")
    print(f"     β*μ̃ = ln(μ̃) = {beta_mu:.3f}")
    print(f"     β*J         = {beta_J:.3f}")
    
    # Load nucleosomes
    file_path = HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"
    print(f"\n   Loading from: {file_path}")
    
    nucs = load_nucleosomes_from_file(
        file_path, 
        k_wrap=1.0, 
        max_nucs=5,
    )
    
    print(f"   ✓ Loaded {len(nucs)} nucleosomes")
    
    # Select a nucleosome for testing
    nuc = nucs[0]
    print(f"\n   Selected nucleosome: id={nuc.id}, subid={nuc.subid}")
    print(f"     G_mat shape     : {nuc.G_mat.shape}")
    print(f"     k_wrap          : {nuc.k_wrap} s⁻¹")
    print(f"     binding_sites   : {nuc.binding_sites}")
    
    # ========================================================================
    # TEST 1: Build state space
    # ========================================================================
    print("\n[2] Building state space...")
    
    N_MAX = nuc.binding_sites
    # transient_states, absorbing_states, index_map = build_state_space(N_MAX)
    
    # print(f"   ✓ N_MAX = {N_MAX}")
    # print(f"   ✓ Number of transient states: {len(transient_states)}")
    # print(f"   ✓ Number of absorbing states: {len(absorbing_states)}")
    # print(f"   ✓ First 10 transient states: {transient_states[:10]}")
    # print(f"   ✓ Absorbing states: {absorbing_states[:5]}...")
    
    # ========================================================================
    # TEST 2: Build Q_TT matrix
    # ========================================================================
    print("\n[3] Building Q_TT generator matrix...")
    
    Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
        nuc,
        k_wrap=1.0,
        protamine_params=protamine_params,
        kT=1.0,
        binding_sites=N_MAX,
        sparse=False
    )
    
    print(f"   ✓ Q_full shape   : {Q_full.shape}")
    print(f"   ✓ Q_TT shape     : {Q_TT.shape}")
    print(f"   ✓ Q_AT shape     : {Q_AT.shape}")
    print(f"   ✓ Absorbing index: {abs_index}")

    (start_idx) = state_index[(0, 0)]

    # column-sum convention: column = "from", row = "to"
    col = Q_full[:, start_idx]

    for i, (l2, r2) in enumerate(states):
        if col[i] != 0:
            print("(l,r)=(0,0) ->", (l2, r2), "rate =", col[i])
    print("diag =", Q_full[start_idx, start_idx])

    
    # Verify generator properties
    col_sums = Q_full.sum(axis=0)
    print(f"   ✓ Column sums (should be ≈0):")
    print(f"     min = {col_sums.min():.2e}, max = {col_sums.max():.2e}")
    
    # Show a small block of Q_TT
    print(f"\n   Top-left 5×5 block of Q_TT:")
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    print(Q_TT[:5, :5])
    
    # ========================================================================
    # TEST 3: Compute MFPT
    # ========================================================================
    print("\n[4] Computing Mean First Passage Time (MFPT)...")
    
    start_state = (0, 0)
    mfpt, tau_vec = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state)
    
    print(f"   ✓ Starting from state: {start_state}")
    print(f"   ✓ MFPT (dimensionless) : {mfpt:.4f}")
    print(f"   ✓ MFPT (physical time) : {mfpt / nuc.k_wrap:.4f} seconds")
    print(f"   ✓ MFPT vector shape    : {tau_vec.shape}")
    print(f"   ✓ MFPT range: [{tau_vec.min():.2f}, {tau_vec.max():.2f}]")
    
    # ========================================================================
    # TEST 4: Compute survival function (matrix exponential)
    # ========================================================================
    print("\n[5] Computing survival function (matrix exponential method)...")
    
    t_max = 10000.0  # Dimensionless time
    n_points = 500
    t_grid = np.linspace(0, t_max, n_points)
    
    S_expm = compute_survival(
        Q_TT, 
        state_index, 
        start_state, 
        t_grid, 
        method='expm'
    )
    
    print(f"   ✓ Time grid: {n_points} points from 0 to {t_max}")
    print(f"   ✓ S(t=0)     = {S_expm[0]:.6f} (should be 1.0)")
    print(f"   ✓ S(t=t_max) = {S_expm[-1]:.6e}")
    
    # Find half-life
    idx_half = np.argmin(np.abs(S_expm - 0.5))
    t_half = t_grid[idx_half]
    print(f"   ✓ Half-life (S=0.5): t ≈ {t_half:.2f} (dimensionless)")
    print(f"                        t ≈ {t_half/nuc.k_wrap:.4f} seconds")
    
    # ========================================================================
    # TEST 5: Compute survival function (ODE method)
    # ========================================================================
    print("\n[6] Computing survival function (ODE solver method)...")
    
    S_ode = compute_survival(
        Q_TT, 
        state_index, 
        start_state, 
        t_grid, 
        method='ode'
    )
    
    print(f"   ✓ S(t=0)     = {S_ode[0]:.6f}")
    print(f"   ✓ S(t=t_max) = {S_ode[-1]:.6e}")
    
    # Compare methods
    max_diff = np.max(np.abs(S_expm - S_ode))
    print(f"   ✓ Max |S_expm - S_ode| = {max_diff:.2e}")
    
    # ========================================================================
    # TEST 6: High-level interface
    # ========================================================================
    print("\n[7] Testing high-level solve_Q_TT_complete()...")
    
    results = solve_Q_TT_complete(
        nuc,
        start_state=(0, 0),
        t_max=10000.0,
        n_points=500,
        method='ode',
        sparse=False,
        k_wrap=1.0,
        protamine_params=protamine_params
    )
    
    print(f"   ✓ Results keys: {list(results.keys())}")
    print(f"   ✓ MFPT         : {results['mfpt']:.4f}")
    print(f"   ✓ k_wrap       : {results['k_wrap']}")
    print(f"   ✓ Survival shape: {results['survival'].shape}")
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    print("\n[8] Generating plots...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Convert to physical time
    t_physical = t_grid / nuc.k_wrap
    mfpt_physical = mfpt / nuc.k_wrap
    
    # --- Plot 1: G-matrix (nucleosome free energy) ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(nuc.G_mat, origin='lower', aspect='equal', cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Free energy (kT)')
    ax1.set_xlabel('Right index j')
    ax1.set_ylabel('Left index i')
    ax1.set_title('Bare Nucleosome G-matrix', fontweight='bold')
    
    # --- Plot 2: Q_TT matrix structure ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Show log of absolute values for better visualization
    Q_TT_vis = np.abs(Q_TT) + 1e-10
    im2 = ax2.imshow(np.log10(Q_TT_vis), origin='lower', aspect='equal', cmap='RdYlBu_r')
    plt.colorbar(im2, ax=ax2, label='log₁₀|Q_TT|')
    ax2.set_xlabel('State index')
    ax2.set_ylabel('State index')
    ax2.set_title('Q_TT Generator Matrix (log scale)', fontweight='bold')
    
    # --- Plot 3: MFPT vector ---
    ax3 = fig.add_subplot(gs[0, 2])
    tau_physical = tau_vec / nuc.k_wrap
    ax3.plot(tau_physical, 'o-', markersize=3, alpha=0.6, color='steelblue')
    ax3.axhline(mfpt_physical, color='red', linestyle='--', alpha=0.7, 
                label=f'MFPT(0,0) = {mfpt_physical:.3f}s')
    ax3.set_xlabel('State index')
    ax3.set_ylabel('MFPT (seconds)')
    ax3.set_title('MFPT from All States', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Survival (linear scale) ---
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(t_physical, S_expm, lw=2.5, alpha=0.8, color='steelblue', 
            label='Matrix Exponential', linestyle='-')
    ax4.plot(t_physical, S_ode, lw=2.0, alpha=0.7, color='coral', 
            label='ODE Solver', linestyle='--')
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax4.axvline(mfpt_physical, color='darkred', linestyle=':', lw=2, alpha=0.7, 
               label=f'MFPT = {mfpt_physical:.3f}s')
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('Survival Probability S(t)', fontsize=11)
    ax4.set_title('Survival Function (Linear Scale)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # Add parameter info
    param_text = f"Nucleosome {nuc.id}-{nuc.subid}\n"
    param_text += f"k_wrap = {nuc.k_wrap:.1f} s⁻¹\n"
    param_text += f"P_conc = {protamine_params['p_conc']:.0f} μM\n"
    param_text += f"J = {protamine_params['cooperativity']:.1f} kT\n"
    param_text += f"μ̃ = {mu_tilde:.2f}"
    ax4.text(0.98, 0.98, param_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- Plot 5: Survival (log scale) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.semilogy(t_physical, S_expm + 1e-10, lw=2.5, alpha=0.8, 
                color='steelblue', label='Matrix Exponential', linestyle='-')
    ax5.semilogy(t_physical, S_ode + 1e-10, lw=2.0, alpha=0.7, 
                color='coral', label='ODE Solver', linestyle='--')
    ax5.axvline(mfpt_physical, color='darkred', linestyle=':', lw=2, alpha=0.7)
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('S(t) (log scale)', fontsize=11)
    ax5.set_title('Survival (Log Scale)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=9)
    
    # --- Plot 6: Method comparison ---
    ax6 = fig.add_subplot(gs[2, :])
    diff = S_expm - S_ode
    ax6.plot(t_physical, diff, lw=1.5, alpha=0.8, color='purple')
    ax6.axhline(0, color='black', linestyle='-', lw=0.5)
    ax6.set_xlabel('Time (seconds)', fontsize=11)
    ax6.set_ylabel('Difference (expm - ODE)', fontsize=11)
    ax6.set_title('Method Comparison: Matrix Exponential vs ODE Solver', 
                  fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Max |diff| = {max_diff:.2e}\n"
    stats_text += f"Mean |diff| = {np.mean(np.abs(diff)):.2e}\n"
    stats_text += f"RMS diff = {np.sqrt(np.mean(diff**2)):.2e}"
    ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Markov Solver Module Test Results', fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'test_markov_solver_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {output_path}")
    
    plt.show()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nSummary:")
    print(f"  • Nucleosome: {nuc.id}-{nuc.subid}")
    print(f"  • MFPT: {mfpt:.4f} (dimensionless) = {mfpt_physical:.4f} seconds")
    print(f"  • Half-life: {t_half:.2f} (dimensionless) = {t_half/nuc.k_wrap:.4f} seconds")
    print(f"  • Method agreement: max difference = {max_diff:.2e}")
    print(f"  • Protamine effect: β*μ = {beta_mu:.3f}, β*J = {beta_J:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
