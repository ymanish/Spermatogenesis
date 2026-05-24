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
from .nucleosome_utils import load_nucleosomes_from_file
from .projection import (project_to_open_sites, compute_open_sites_evolution)

# Drift-reversal analysis lives in src.analysis.barrier.drift_reversal now.

__all__ = [
    'solve_Q_TT_complete',
    'build_state_space',
    'build_full_Q_from_nucleosome',
    'compute_mfpt_from_Q_TT',
    'compute_survival',
    'load_nucleosomes_from_file',
    'project_to_open_sites',
    'compute_open_sites_evolution',
]
