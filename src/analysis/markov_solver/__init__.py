"""
Markov Chain Solver for Nucleosome Unwrapping in (l,r) Space
==============================================================

This package provides tools for solving continuous-time Markov chains (CTMC)
for nucleosome unwrapping dynamics in (l,r) space with protamine binding effects.

Key Features:
-------------
- Fast protamine limit: closing rates gated by P_free
- Opening rates use bare nucleosome free energies
- Absorbing boundary at l+r = N_MAX (full detachment)
- MFPT and survival function computation

Author: MY
Date: 2025-12-10
"""

# Lazy imports to avoid circular dependencies

from .solver import (solve_Q_TT_complete)

from .state_space import (build_state_space)
from .generator import (build_full_Q_from_nucleosome)
from .mfpt import (compute_mfpt_from_Q_TT)
from .survival import compute_survival
from .nucleosome_utils import load_nucleosomes_from_file, load_nucleosomes_from_sprm
from .projection import (project_to_open_sites, compute_open_sites_evolution)
from .tnp2 import (
    TNP2Config,
    count_cpg,
    get_site_ranges,
    count_cpg_per_site,
    compute_tnp2_occupancy_profile,
    compute_jeff_profile,
    compute_oriented_jeff_profiles,
    parse_fasta,
)
# Barrier analysis - re-exported for backward compatibility when dependencies exist.
try:
    from src.analysis.barrier.drift_reversal import (
        DriftReversalAnalyzer,
        DriftReversalResults,
        compare_to_ssa,
    )
except ImportError:
    DriftReversalAnalyzer = None
    DriftReversalResults = None
    compare_to_ssa = None

__all__ = [
    'solve_Q_TT_complete',
    'build_state_space',
    'build_full_Q_from_nucleosome',
    'compute_mfpt_from_Q_TT',
    'compute_survival',
    'load_nucleosomes_from_file',
    'load_nucleosomes_from_sprm',
    'project_to_open_sites',
    'compute_open_sites_evolution',
    'DriftReversalAnalyzer',
    'DriftReversalResults',
    'compare_to_ssa',
    'TNP2Config',
    'count_cpg',
    'get_site_ranges',
    'count_cpg_per_site',
    'compute_tnp2_occupancy_profile',
    'compute_jeff_profile',
    'compute_oriented_jeff_profiles',
    'parse_fasta',
]
