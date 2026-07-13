"""Configuration for the event-driven Gillespie pipeline.

Mirrors the dataclass-style pattern of src.config.custom_type.SimulationConfig
but lives in the new package so src/config/ is untouched. No sampling grid:
tau_max is a censoring boundary, n_survival_points / tau_spacing / tau_log_min
only control the empirical S(tau) grid resolution.
"""

import numpy as np


def build_tau_grid(tau_max: float, n_points: int, spacing: str = "linear",
                   log_min: float = 1e-2) -> np.ndarray:
    """Empirical-survival evaluation grid in dimensionless tau.

    'linear' — uniform spacing on [0, tau_max].
    'log'    — tau=0 followed by log-spaced points on [log_min, tau_max]; dense
               where S(tau) drops, sparse in the tail. Matches the Markov solver's
               log grid so the two survival curves compare on the same abscissa.

    Kept as a module-level function (not a config attribute) so the worker in
    batch.py can rebuild it from scalars without pickling an array across the
    process boundary.
    """
    if spacing == "linear":
        return np.linspace(0.0, tau_max, n_points)
    if spacing == "log":
        if n_points < 2:
            raise ValueError(f"log tau_spacing needs n_points >= 2, got {n_points}")
        if not (0 < log_min < tau_max):
            raise ValueError(
                f"log tau_spacing needs 0 < tau_log_min < tau_max, got "
                f"tau_log_min={log_min}, tau_max={tau_max}")
        return np.concatenate((
            [0.0],
            np.logspace(np.log10(log_min), np.log10(tau_max), n_points - 1),
        ))
    raise ValueError(f"tau_spacing must be 'linear' or 'log', got {spacing!r}")


class GillespieEventConfig:

    def __init__(
        self,
        # Nucleosome parameters
        k_wrap: float = 1.0,
        binding_sites: int = 14,

        # Protamine parameters
        prot_k_unbind: float = 89.7,
        prot_k_bind: float = 1.0,
        prot_p_conc: float = 0.0,
        prot_cooperativity: float = 0.0,

        # Time / sampling
        tau_max: float = 10000.0,
        n_survival_points: int = 1000,
        tau_spacing: str = "linear",   # 'linear' or 'log' (match the Markov grid)
        tau_log_min: float = 1e-2,     # smallest nonzero tau for the log grid

        # Simulation behavior
        inf_protamine: bool = True,
        replicates: int = 20,

        # Execution
        batch_size: int = 10,
        n_workers: int = 4,
        flush_every: int = 10000,

        # Output
        save_trajectories: bool = True,
    ):
        if tau_max <= 0:
            raise ValueError(f"tau_max must be > 0, got {tau_max}")
        if n_survival_points < 2:
            raise ValueError(f"n_survival_points must be >= 2, got {n_survival_points}")
        # Validate the grid spec eagerly (raises for bad spacing / tau_log_min).
        build_tau_grid(tau_max, n_survival_points, tau_spacing, tau_log_min)
        if replicates < 1:
            raise ValueError(f"replicates must be >= 1, got {replicates}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")

        self.k_wrap = k_wrap
        self.binding_sites = binding_sites

        self.prot_k_unbind = prot_k_unbind
        self.prot_k_bind = prot_k_bind
        self.prot_p_conc = prot_p_conc
        self.prot_cooperativity = prot_cooperativity

        self.tau_max = tau_max
        self.n_survival_points = n_survival_points
        self.tau_spacing = tau_spacing
        self.tau_log_min = tau_log_min

        self.inf_protamine = inf_protamine
        self.replicates = replicates

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.flush_every = flush_every

        self.save_trajectories = save_trajectories

        # NOTE: the survival-curve tau grid is NOT built here. The worker
        # (src/gillespie_event/batch.py) recomputes it from tau_max /
        # n_survival_points so the grid never has to cross the process boundary.
        self.prot_params = {
            "k_unbind": prot_k_unbind,
            "k_bind": prot_k_bind,
            "p_conc": prot_p_conc,
            "cooperativity": prot_cooperativity,
        }

    def __repr__(self) -> str:
        return (
            f"GillespieEventConfig(k_wrap={self.k_wrap}, "
            f"prot_p_conc={self.prot_p_conc}, "
            f"prot_cooperativity={self.prot_cooperativity}, "
            f"tau_max={self.tau_max}, replicates={self.replicates}, "
            f"n_workers={self.n_workers})"
        )

    def to_dict(self) -> dict:
        return {
            "k_wrap": self.k_wrap,
            "binding_sites": self.binding_sites,
            "prot_k_unbind": self.prot_k_unbind,
            "prot_k_bind": self.prot_k_bind,
            "prot_p_conc": self.prot_p_conc,
            "prot_cooperativity": self.prot_cooperativity,
            "tau_max": self.tau_max,
            "n_survival_points": self.n_survival_points,
            "tau_spacing": self.tau_spacing,
            "tau_log_min": self.tau_log_min,
            "inf_protamine": self.inf_protamine,
            "replicates": self.replicates,
            "batch_size": self.batch_size,
            "n_workers": self.n_workers,
            "flush_every": self.flush_every,
            "save_trajectories": self.save_trajectories,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GillespieEventConfig":
        return cls(**d)
