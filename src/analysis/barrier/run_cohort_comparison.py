#!/usr/bin/env python3
"""
Cohort comparison: energy-landscape and kinetic descriptors.
=============================================================

Loads two (or more) SPRM nucleosome cohorts, computes per-nucleosome
descriptors, and produces comparison figures.

HOW TO RUN
----------
Edit the CONFIG block, then:

    python -m src.analysis.barrier.run_cohort_comparison

or:

    python src/analysis/barrier/run_cohort_comparison.py

TWO ANALYSIS MODES
------------------
MODE 1 — Landscape descriptors only (fast, no protamine needed)
    Computes: ΔE_firstbreath, ΔE_barrier, σ_ΔE for every nucleosome.
    Uses: compute_cohort_descriptors / compare_cohorts

MODE 2 — Full drift-reversal (slower, protamine-dependent)
    Computes everything in MODE 1 PLUS:
      • 1D MFPT and nucleation-theory MFPT
      • Critical nucleus n* and refined crossing n*_refined
      • Quasi-potential barrier ΔΦ
      • Committor value at n*
    Uses: run_batch_drift_reversal

Set MODE = 1 or MODE = 2 in the CONFIG block below.
"""

from pathlib import Path
import matplotlib

from src.config.path import SPRM_DATA_DIR, SPRM_OUT_DIR
matplotlib.use("Agg")   # switch to "TkAgg" / "Qt5Agg" for interactive use
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

# Analysis mode: 1 = descriptors only, 2 = full drift reversal
MODE = 1

# Cohorts: name -> SPRM dataset directory
COHORT_DIRS = {
    'ret_single_nuc': SPRM_DATA_DIR / "ret_single_nuc",
    'ctrl04':         SPRM_DATA_DIR / "ctrl04_bound_prom_evicted",
}

# Physical / model parameters
K_WRAP        = 1.0
KT            = 1.0
BINDING_SITES = 14

# Protamine parameters — used only in MODE 2
# (descriptors in MODE 1 are protamine-independent)
PROTAMINE_PARAMS = {
    'k_bind':        1.0,
    'k_unbind':      89.7,
    'p_conc':        0.0,   # µM
    'cooperativity': 0.0,
}

# Cap the number of nucleosomes per cohort (None = all)
MAX_NUCS = None

# Output directory for CSV and figures
OUT_DIR = SPRM_OUT_DIR / "barrier" / "cohort_comparison"  # e.g. SPRM_OUT_DIR / "barrier" / "cohort_comparison"

# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if MODE == 1:
        _run_descriptor_comparison()
    elif MODE == 2:
        _run_full_drift_reversal_comparison()
    else:
        raise ValueError(f"Unknown MODE={MODE}. Choose 1 or 2.")


# ── MODE 1 ──────────────────────────────────────────────────────────────────

def _run_descriptor_comparison() -> None:
    """
    MODE 1: landscape descriptors only.

    Outputs
    -------
    cohort_descriptors.csv
        One row per nucleosome with columns:
        id, subid, cohort, dE_firstbreath, dE_barrier, sigma_dE
    descriptor_comparison.png
        Strip+box plot comparing the three descriptors across cohorts.
        Each panel is annotated with a Mann-Whitney U p-value.
    """
    from src.analysis.barrier.landscape_batch import compare_cohorts
    from src.analysis.barrier.drift_reversal_plots import plot_landscape_descriptor_comparison

    print("MODE 1 — Landscape descriptors")
    df = compare_cohorts(
        COHORT_DIRS,
        k_wrap=K_WRAP,
        kT=KT,
        binding_sites=BINDING_SITES,
        max_nucs=MAX_NUCS,
    )

    _print_summary(df, cols=['dE_firstbreath', 'dE_barrier', 'sigma_dE'])

    # Save CSV
    csv_path = OUT_DIR / "cohort_descriptors.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData saved → {csv_path}")

    # Comparison figure
    fig = plot_landscape_descriptor_comparison(df)
    fig_path = OUT_DIR / "descriptor_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved → {fig_path}")


# ── MODE 2 ──────────────────────────────────────────────────────────────────

def _run_full_drift_reversal_comparison() -> None:
    """
    MODE 2: full drift-reversal analysis per nucleosome.

    Outputs
    -------
    cohort_full_drift_reversal.csv
        One row per nucleosome with columns:
        id, subid, cohort,
        mfpt_1d, mfpt_nucleation, n_star, n_star_refined,
        delta_phi, committor_at_nstar,
        dE_firstbreath, dE_barrier, sigma_dE
    descriptor_comparison.png
        Strip+box of the three landscape descriptors.
    kinetics_comparison.png
        Strip+box of mfpt_1d, delta_phi, n_star, committor_at_nstar.
    """
    from src.analysis.barrier.landscape_batch import run_batch_drift_reversal
    from src.analysis.barrier.drift_reversal_plots import plot_landscape_descriptor_comparison

    print("MODE 2 — Full drift-reversal analysis")

    frames = []
    for name, d in COHORT_DIRS.items():
        print(f"  Processing cohort: {name}")
        df = run_batch_drift_reversal(
            Path(d),
            protamine_params=PROTAMINE_PARAMS,
            k_wrap=K_WRAP,
            kT=KT,
            binding_sites=BINDING_SITES,
            max_nucs=MAX_NUCS,
        )
        df['cohort'] = name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    all_cols = [
        'dE_firstbreath', 'dE_barrier', 'sigma_dE',
        'mfpt_1d', 'delta_phi', 'n_star', 'committor_at_nstar',
    ]
    _print_summary(df, cols=all_cols)

    # Save CSV
    csv_path = OUT_DIR / "cohort_full_drift_reversal.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData saved → {csv_path}")

    # Figure 1: landscape descriptors
    fig1 = plot_landscape_descriptor_comparison(df)
    fig1.savefig(OUT_DIR / "descriptor_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: kinetic quantities
    fig2 = plot_landscape_descriptor_comparison(
        df,
        descriptors=('mfpt_1d', 'delta_phi', 'n_star', 'committor_at_nstar'),
    )
    fig2.savefig(OUT_DIR / "kinetics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"Figures saved → {OUT_DIR}")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, cols: list) -> None:
    existing = [c for c in cols if c in df.columns]
    print("\n" + "=" * 60)
    print(df.groupby('cohort')[existing].describe().T.to_string())
    print("=" * 60)


if __name__ == "__main__":
    run()
