"""
Configuration Module
====================

Configuration class for Markov solver execution.

Author: MY
Date: 2025-12-11
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MarkovConfig:
    """
    Configuration for Markov solver execution on multiple nucleosomes.
    
    This class encapsulates all parameters needed to run Markov solver calculations,
    following the same pattern as SimulationConfig for consistency.
    
    Attributes:
        Nucleosome parameters:
            k_wrap: Wrapping rate constant (s^-1) - sets the timescale
            binding_sites: Number of DNA-histone binding sites (default: 14)
            kT: Thermal energy (default: 1.0 k_B T)
        
        Protamine parameters:
            prot_k_bind: Protamine binding rate ((μM·s)^-1)
            prot_k_unbind: Protamine unbinding rate (s^-1)
            prot_p_conc: Protamine concentration (μM)
            prot_cooperativity: Cooperativity parameter J (k_B T)
        
        Computation parameters:
            tau_max: Maximum dimensionless time τ (τ = k_wrap × t_physical)
            tau_steps: Number of time points for survival function evaluation
            method: Solver method ('expm' or 'ode')
            sparse: Whether to use sparse matrices
            compute_states: Whether to save full state probabilities P(t)
            dimensionless: Whether to return Q in dimensionless units
        
        Execution parameters:
            batch_size: Number of nucleosomes per batch
            n_workers: Number of parallel workers
            max_nucs: Maximum number of nucleosomes to process (None = all)
        
        Output parameters:
            save_survival: Whether to save survival function S(t)
            save_states: Whether to save state probabilities P(t)
            save_mfpt: Whether to save MFPT values
            save_generator: Whether to save Q matrices
    
    Computed attributes:
        t_grid: Array of dimensionless time points (computed from t_max/t_steps)
        protamine_params: Dictionary of protamine parameters
    
    Example:
        >>> config = MarkovConfig(
        ...     k_wrap=1.0,
        ...     prot_p_conc=10.0,
        ...     prot_cooperativity=0.0,
        ...     t_max=1000.0,
        ...     t_steps=500,
        ...     n_workers=10,
        ...     save_survival=True,
        ...     save_mfpt=True
        ... )
        >>> 
        >>> run_markov_solver(
        ...     file_path=Path("data/nucleosomes.tsv"),
        ...     output_dir=Path("output/markov"),
        ...     config=config
        ... )
    """
    
    # Nucleosome parameters
    k_wrap: float = 1.0
    binding_sites: int = 14
    kT: float = 1.0
    
    # Protamine parameters
    prot_k_bind: float = 1.0
    prot_k_unbind: float = 89.7
    prot_p_conc: float = 0.0
    prot_cooperativity: float = 0.0
    
    # Computation parameters
    tau_max: float = 1000.0
    tau_steps: int = 500
    tau_spacing: str = 'linear'   # 'linear' or 'log' (survival-curve τ grid)
    tau_log_min: float = 1e-2     # smallest nonzero τ for the log grid (ignored when linear)
    method: str = 'expm'  # 'expm' or 'ode'
    sparse: bool = False
    compute_states: bool = False
    dimensionless: bool = True
    
    # Execution parameters
    batch_size: int = 10
    n_workers: int = 10
    max_nucs: Optional[int] = None
    
    # Output parameters
    save_survival: bool = True
    save_states: bool = False
    save_mfpt: bool = True
    save_generator: bool = False
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        # Time grid (dimensionless τ)
        self.tau_grid = self._build_tau_grid()

        # Protamine parameters dictionary
        self.protamine_params = {
            'k_bind': self.prot_k_bind,
            'k_unbind': self.prot_k_unbind,
            'p_conc': self.prot_p_conc,
            'cooperativity': self.prot_cooperativity
        }
        
        # Validate parameters
        self._validate()

    def _build_tau_grid(self) -> np.ndarray:
        """Build the dimensionless-time evaluation grid for the survival curve.

        'linear' — uniform spacing on [0, tau_max]; simple, but wastes points in
                   the flat tail while under-resolving the steep early decay.
        'log'    — τ=0 followed by log-spaced points on [tau_log_min, tau_max];
                   dense where S(τ) drops, sparse in the smooth exponential tail.
                   Preferred for MFPT-agnostic survival shape and for matching a
                   Gillespie empirical survival where it has statistical power.

        Note: the MFPT is a direct linear solve and does NOT depend on this grid;
        tau_spacing/tau_steps only affect the saved survival curve.
        """
        if self.tau_spacing == 'linear':
            return np.linspace(0, self.tau_max, self.tau_steps)
        if self.tau_spacing == 'log':
            if self.tau_steps < 2:
                raise ValueError(
                    f"log tau_spacing needs tau_steps >= 2, got {self.tau_steps}")
            if not (0 < self.tau_log_min < self.tau_max):
                raise ValueError(
                    f"log tau_spacing needs 0 < tau_log_min < tau_max, got "
                    f"tau_log_min={self.tau_log_min}, tau_max={self.tau_max}")
            return np.concatenate((
                [0.0],
                np.logspace(np.log10(self.tau_log_min), np.log10(self.tau_max),
                            self.tau_steps - 1),
            ))
        raise ValueError(
            f"tau_spacing must be 'linear' or 'log', got {self.tau_spacing!r}")

    def _validate(self):
        """Validate configuration parameters."""
        if self.k_wrap <= 0:
            raise ValueError(f"k_wrap must be positive, got {self.k_wrap}")
        
        if self.binding_sites <= 0:
            raise ValueError(f"binding_sites must be positive, got {self.binding_sites}")
        
        if self.tau_max <= 0:
            raise ValueError(f"tau_max must be positive, got {self.tau_max}")
        
        if self.tau_steps <= 0:
            raise ValueError(f"tau_steps must be positive, got {self.tau_steps}")
        
        if self.method not in ['expm', 'ode']:
            raise ValueError(f"method must be 'expm' or 'ode', got {self.method}")
        
        if self.n_workers <= 0:
            raise ValueError(f"n_workers must be positive, got {self.n_workers}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # With no protamine in solution, cooperativity has no physical effect:
        # there is no neighbor to couple to. Reject the combination to prevent
        # generating duplicate runs that all collapse to the (conc=0, coop=0) result.
        if self.prot_p_conc == 0.0 and self.prot_cooperativity != 0.0:
            raise ValueError(
                f"prot_cooperativity must be 0.0 when prot_p_conc is 0.0 "
                f"(got prot_cooperativity={self.prot_cooperativity}). "
                f"Without protamine, cooperativity has no effect."
            )
    
    def get_info_dict(self) -> dict:
        """Get configuration as dictionary for logging/saving."""
        return {
            'k_wrap': self.k_wrap,
            'binding_sites': self.binding_sites,
            'prot_k_bind': self.prot_k_bind,
            'prot_k_unbind': self.prot_k_unbind,
            'prot_p_conc': self.prot_p_conc,
            'prot_cooperativity': self.prot_cooperativity,
            'tau_max': self.tau_max,
            'tau_steps': self.tau_steps,
            'tau_spacing': self.tau_spacing,
            'tau_log_min': self.tau_log_min,
            'method': self.method,
            'sparse': self.sparse,
            'n_workers': self.n_workers,
            'batch_size': self.batch_size
        }
