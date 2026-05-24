"""
Barrier analysis package for nucleosome eviction kinetics.

Provides drift-reversal analysis, energy-landscape descriptors,
and batch cohort comparison tools.
"""

from .drift_reversal import DriftReversalAnalyzer, DriftReversalResults, compare_to_ssa
from .drift_reversal_plots import (
    plot_drift_and_rates,
    plot_committor,
    plot_quasi_potential,
    plot_survival_comparison,
    plot_full_analysis,
    plot_phase_diagram,
    plot_mfpt_vs_barrier,
    plot_shell_composition,
    plot_landscape_descriptor_comparison,
)
from .landscape_batch import compute_cohort_descriptors, compare_cohorts, run_batch_drift_reversal

__all__ = [
    'DriftReversalAnalyzer',
    'DriftReversalResults',
    'compare_to_ssa',
    'plot_drift_and_rates',
    'plot_committor',
    'plot_quasi_potential',
    'plot_survival_comparison',
    'plot_full_analysis',
    'plot_phase_diagram',
    'plot_mfpt_vs_barrier',
    'plot_shell_composition',
    'plot_landscape_descriptor_comparison',
    'compute_cohort_descriptors',
    'compare_cohorts',
    'run_batch_drift_reversal',
]
