#!/usr/bin/env python3
"""
Single-nucleosome drift-reversal analysis.
==========================================

Loads one nucleosome from an SPRM dataset (or from a TSV file), runs the
full DriftReversalAnalyzer pipeline, prints a summary, and generates the
standard four-panel figure plus a landscape-descriptor summary.

HOW TO RUN
----------
Edit the CONFIG block below, then:

    python -m src.analysis.barrier.analyse_single_nucleosome

or from the repo root:

    python src/analysis/barrier/analyse_single_nucleosome.py

ANALYSES PERFORMED
------------------
1. Effective rates k⁺(n), k⁻(n) and drift v(n) across the unwrapping coordinate
2. Critical nucleus n* (where drift changes sign)
3. Quasi-potential Φ(n) and barrier height ΔΦ
4. Exact 1D MFPT and nucleation-theory approximation
5. Committor q(n) — probability to detach before rewrapping
6. Landscape descriptors: ΔE_firstbreath, ΔE_barrier, σ_ΔE
7. Shell composition (microstate weights within each n-shell)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # change to "TkAgg" / "Qt5Agg" for interactive display
import matplotlib.pyplot as plt
from src.config.path import SPRM_DATA_DIR, SPRM_OUT_DIR

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

# SPRM directory for the nucleosome dataset
SPRM_DIR = SPRM_DATA_DIR / "ret_single_nuc"  # e.g. SPRM_DATA_DIR / "ctrl01_random_genome_safe"

# Which nucleosome to analyse (0-indexed within the loaded list)
NUC_INDEX = 0

# Physical / model parameters
K_WRAP        = 1.0    # wrapping rate (s⁻¹ in physical units, 1.0 for dimensionless)
KT            = 1.0    # thermal energy (same units as G_mat)
BINDING_SITES = 14

# Protamine parameters (set p_conc = 0.0 for no-protamine baseline)
PROTAMINE_PARAMS = {
    'k_bind':        1.0,
    'k_unbind':      89.7,
    'p_conc':        0.0,   # µM; 0 = no protamine
    'cooperativity': 0.0,
}

# Output directory for saved figures (None = don't save, just display)
OUT_DIR = SPRM_OUT_DIR / "single_nuc_analysis"  # e.g. SPRM_OUT_DIR / "single_nuc_analysis"
# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def run() -> None:
    from src.analysis.barrier.drift_reversal import DriftReversalAnalyzer
    from src.analysis.barrier.drift_reversal_plots import (
        plot_drift_and_rates,
        plot_quasi_potential,
        plot_committor,
        plot_shell_composition,
        plot_full_analysis,
    )

    from src.analysis.markov_solver.nucleosome_utils import load_nucleosomes_from_sprm
    nucs = load_nucleosomes_from_sprm(
        SPRM_DIR, k_wrap=K_WRAP, kT=KT,
        binding_sites=BINDING_SITES, max_nucs=NUC_INDEX + 1
    )

    if not nucs:
        raise RuntimeError("No nucleosomes loaded — check SPRM_DIR / TSV_FILE.")
    nuc = nucs[NUC_INDEX]
    print(f"\nNucleosome: id={nuc.id}  subid={nuc.subid}")

    # --- Run full drift-reversal analysis ---
    analyzer = DriftReversalAnalyzer(
        nuc,
        k_wrap=K_WRAP,
        kT=KT,
        binding_sites=BINDING_SITES,
        protamine_params=PROTAMINE_PARAMS,
    )
    res = analyzer.analyze()

    # --- Print summary ---
    print("\n" + "=" * 55)
    print("DRIFT-REVERSAL SUMMARY")
    print("=" * 55)
    print(f"  Critical nucleus  n*      = {res.n_star}")
    print(f"  Refined crossing  n*_ref  = {res.n_star_refined:.3f}"
          if res.n_star_refined is not None else "  Refined crossing  n*_ref  = None")
    print(f"  Barrier height    ΔΦ      = {res.delta_phi:.4f}")
    print(f"  1D MFPT                   = {res.mfpt_1d:.4e}  (×1/k_wrap for physical)")
    if res.mfpt_nucleation is not None:
        print(f"  Nucleation MFPT           = {res.mfpt_nucleation:.4e}")
    if res.n_star is not None:
        q_star = res.committor[res.n_star]
        print(f"  Committor at n*  q(n*)   = {q_star:.4f}")
    print()
    print("LANDSCAPE DESCRIPTORS (sequence-intrinsic)")
    print(f"  ΔE_firstbreath            = {res.dE_firstbreath:.4f} kT")
    print(f"  ΔE_barrier                = {res.dE_barrier:.4f} kT")
    print(f"  σ_ΔE  (ruggedness)        = {res.sigma_dE:.4f} kT")
    print("=" * 55)

    # --- Figures ---
    if OUT_DIR is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Full 4-panel summary
    fig_full = plot_full_analysis(res)
    _save_or_show(fig_full, OUT_DIR, "full_analysis.png")

    # 2. Drift & rates
    fig_drift, ax = plt.subplots(figsize=(8, 5))
    plot_drift_and_rates(res, ax=ax)
    _save_or_show(fig_drift, OUT_DIR, "drift_rates.png")

    # 3. Quasi-potential
    fig_phi, ax = plt.subplots(figsize=(8, 5))
    plot_quasi_potential(res, ax=ax)
    _save_or_show(fig_phi, OUT_DIR, "quasi_potential.png")

    # 4. Committor
    fig_q, ax = plt.subplots(figsize=(8, 5))
    plot_committor(res, ax=ax)
    _save_or_show(fig_q, OUT_DIR, "committor.png")

    # 5. Shell composition (microstates at n=0, n*/2, n*, N-1)
    fig_shell = plot_shell_composition(res)
    _save_or_show(fig_shell, OUT_DIR, "shell_composition.png")

    print("\nDone." + (f"  Figures saved to {OUT_DIR}" if OUT_DIR else ""))


def _save_or_show(fig, out_dir, filename):
    if out_dir is not None:
        fig.savefig(out_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    run()
