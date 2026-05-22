from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import numpy as np

from src.config.var import create_nucleosomes_instance, create_protamines_instance
from src.core.helper.tau_min import _compute_tau_min


class PilotConfig:
    """Archived configuration for pilot replicate-estimation workflows."""

    def __init__(
        self,
        k_wrap: float = 1.0,
        binding_sites: int = 14,
        tau_max: float = 10000.0,
        tau_steps: int = 1000,
        prot_k_unbind: float = 89.7,
        prot_k_bind: float = 1.0,
        prot_p_conc: float = 0.0,
        prot_cooperativity: float = 0.0,
        inf_protamine: bool = True,
        renucleation: bool = False,
        n_pilot_nucleosomes: int = 20,
        n_pilot_replicates: int = 50,
        start_idx: int = 0,
    ):
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        self.tau_max = tau_max
        self.tau_steps = tau_steps
        self.tau_points = np.linspace(0, tau_max, tau_steps)

        self.prot_params = {
            "k_unbind": prot_k_unbind,
            "k_bind": prot_k_bind,
            "p_conc": prot_p_conc,
            "cooperativity": prot_cooperativity,
        }

        self.inf_protamine = inf_protamine
        self.renucleation = renucleation
        self.n_pilot_nucleosomes = n_pilot_nucleosomes
        self.n_pilot_replicates = n_pilot_replicates
        self.start_idx = start_idx
        self.tau_min = _compute_tau_min(k_wrap=k_wrap, ends=2, gamma=5.0) if renucleation else None
        self.build_params = {
            "nucs_factory": partial(
                create_nucleosomes_instance,
                k_wrap=k_wrap,
                binding_sites=binding_sites,
            ),
            "prot_factory": partial(
                create_protamines_instance,
                prot_params=self.prot_params,
            ),
        }


@dataclass
class RMSTAnalysis:
    """Archived RMST variance-analysis result container."""

    sigma_within_sq: float
    sigma_between_sq: float
    R: float
    recommended_replicates: str
    n_nucleosomes: int
    n_replicates: int
    condition_label: str
    mean_rmst: float
    std_rmst: float
    nucleosome_mean_rmsts: List[float]
    nucleosome_std_rmsts: List[float]
    tolerance: Optional[float] = None
    n_reps_required: Optional[int] = None
    tau_max: float = None
    delta_tau: float = None
