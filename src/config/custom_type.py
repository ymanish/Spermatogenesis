from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, NamedTuple, Optional

import numpy as np

from src.core.helper.tau_min import _compute_tau_min

class SiteState(IntEnum):
    CLOSED = 0   # fully wrapped
    OPEN   = 1   # unwrapped / unbound
    BOUND  = 2   # unwrapped + protamine

@dataclass
class SimulationState:
    tau: float
    time: float
    cs_total: int
    detached_total: int
    bprot: int
    # cs: Optional[List[np.ndarray]] = None 
    # t_blocked: Optional[float] = None
    # nucs_snapshot: Optional[List[np.ndarray]] = None


class ReactionType(IntEnum):
    UNWRAPPING = 0
    REWRAPPING = 1
    BINDING    = 2
    UNBINDING  = 3

class Rates(NamedTuple):
    persite: Dict[ReactionType, Dict[int, float]]
    total:   Dict[ReactionType, float]

@dataclass
class ReactionChoice:
    nuc_idx: int
    reaction: ReactionType

REACTION_TARGET_STATE: Dict[ReactionType, SiteState] = {
    ReactionType.UNWRAPPING: SiteState.OPEN,
    ReactionType.REWRAPPING: SiteState.CLOSED,
    ReactionType.BINDING:    SiteState.BOUND,
    ReactionType.UNBINDING:  SiteState.OPEN,
}


class SimulationConfig:
    """
    Configuration for full-scale Gillespie simulations.
    
    This class encapsulates all parameters needed to run nucleosome simulations,
    following the same pattern as PilotConfig for consistency.
    
    Attributes:
        Nucleosome parameters:
            k_wrap: Wrapping energy constant (k_B T)
            binding_sites: Number of DNA-histone binding sites (default: 14)
        
        Protamine parameters:
            prot_k_unbind: Protamine unbinding rate
            prot_k_bind: Protamine binding rate
            prot_p_conc: Protamine concentration (μM)
            prot_cooperativity: Cooperativity factor (k_B T)
        
        Simulation parameters:
            tau_max: Maximum dimensionless time
            tau_steps: Number of time steps for integration
            inf_protamine: Whether to use infinite protamine
            renucleation: Whether to enable renucleation
            replicates: Number of replicates per nucleosome
        
        Execution parameters:
            batch_size: Number of nucleosomes per batch
            n_workers: Number of parallel workers
            flush_every: Flush frequency for logging
        
        Trajectory parameters:
            save_trajectories: Whether to save trajectory data
            maxpoints_saved_trajectories: Maximum trajectory points to save
    
    Computed attributes:
        tau_points: Array of tau time points (computed from tau_max/tau_steps)
        prot_params: Dictionary of protamine parameters
        tau_min: Minimum tau for renucleation (computed if renucleation=True)
    
    Example:
        >>> config = SimulationConfig(
        ...     k_wrap=1.0,
        ...     prot_p_conc=100.0,
        ...     prot_cooperativity=4.5,
        ...     tau_max=10000.0,
        ...     tau_steps=1000,
        ...     replicates=20,
        ...     n_workers=10,
        ...     save_trajectories=True
        ... )
        >>> print(config.tau_points.shape)
        (1000,)
        >>> print(config.prot_params)
        {'k_unbind': 89.7, 'k_bind': 1.0, 'p_conc': 100.0, 'cooperativity': 4.5}
    """
    
    def __init__(self,
                 # Nucleosome parameters
                 k_wrap: float = 1.0,
                 binding_sites: int = 14,
                 
                 # Protamine parameters
                 prot_k_unbind: float = 89.7,
                 prot_k_bind: float = 1.0,
                 prot_p_conc: float = 0.0,
                 prot_cooperativity: float = 0.0,
                 
                 # Simulation time parameters
                 tau_max: float = 10000.0,
                 tau_steps: int = 1000,
                 
                 # Simulation behavior
                 inf_protamine: bool = True,
                 renucleation: bool = False,
                 replicates: int = 20,
                 
                 # Execution parameters
                 batch_size: int = 10,
                 n_workers: int = 4,
                 flush_every: int = 10000,
                 
                 # Trajectory parameters
                 save_trajectories: bool = False,
                 maxpoints_saved_trajectories: int = 100):
        """
        Initialize simulation configuration.
        
        Args:
            k_wrap: Nucleosome wrapping constant (default: 1.0)
            binding_sites: Number of binding sites (default: 14)
            prot_k_unbind: Protamine unbinding rate (default: 89.7)
            prot_k_bind: Protamine binding rate (default: 1.0)
            prot_p_conc: Protamine concentration in μM (default: 0.0)
            prot_cooperativity: Cooperativity factor (default: 0.0)
            tau_max: Maximum dimensionless time (default: 10000.0)
            tau_steps: Number of time steps (default: 1000)
            inf_protamine: Use infinite protamine (default: True)
            renucleation: Enable renucleation (default: False)
            replicates: Number of replicates per nucleosome (default: 20)
            batch_size: Nucleosomes per batch (default: 10)
            n_workers: Number of parallel workers (default: 4)
            flush_every: Flush frequency for logging (default: 10000)
            save_trajectories: Save trajectory data (default: False)
            maxpoints_saved_trajectories: Max trajectory points (default: 100)
        
        Raises:
            ValueError: If maxpoints_saved_trajectories > tau_steps
        """
        # Nucleosome parameters
        self.k_wrap = k_wrap
        self.binding_sites = binding_sites
        
        # Time parameters
        self.tau_max = tau_max
        self.tau_steps = tau_steps
        self.tau_points = np.linspace(0, tau_max, tau_steps)
        
        # Protamine parameters
        self.prot_params = {
            'k_unbind': prot_k_unbind,
            'k_bind': prot_k_bind,
            'p_conc': prot_p_conc,
            'cooperativity': prot_cooperativity
        }
        
        # Simulation behavior
        self.inf_protamine = inf_protamine
        self.renucleation = renucleation
        self.replicates = replicates
        
        # Execution parameters
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.flush_every = flush_every
        
        # Trajectory parameters
        self.save_trajectories = save_trajectories
        self.maxpoints_saved_trajectories = maxpoints_saved_trajectories
        
        # Validate trajectory parameters
        if save_trajectories and maxpoints_saved_trajectories > tau_steps:
            raise ValueError(
                f"maxpoints_saved_trajectories ({maxpoints_saved_trajectories}) "
                f"cannot exceed tau_steps ({tau_steps}). "
                f"Either increase tau_steps or decrease maxpoints_saved_trajectories."
            )
        
        # Calculate tau_min for renucleation
        if renucleation:
            self.tau_min = _compute_tau_min(k_wrap=k_wrap, ends=2, gamma=5.0)
        else:
            self.tau_min = None
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"SimulationConfig("
            f"k_wrap={self.k_wrap}, "
            f"prot_p_conc={self.prot_params['p_conc']}, "
            f"prot_coop={self.prot_params['cooperativity']}, "
            f"tau_max={self.tau_max}, "
            f"replicates={self.replicates}, "
            f"n_workers={self.n_workers})"
        )
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'k_wrap': self.k_wrap,
            'binding_sites': self.binding_sites,
            'tau_max': self.tau_max,
            'tau_steps': self.tau_steps,
            'prot_params': self.prot_params.copy(),
            'inf_protamine': self.inf_protamine,
            'renucleation': self.renucleation,
            'replicates': self.replicates,
            'batch_size': self.batch_size,
            'n_workers': self.n_workers,
            'flush_every': self.flush_every,
            'save_trajectories': self.save_trajectories,
            'maxpoints_saved_trajectories': self.maxpoints_saved_trajectories,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SimulationConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
        
        Returns:
            SimulationConfig instance
        """
        # Extract protamine parameters if nested
        prot_params = config_dict.get('prot_params', {})
        
        return cls(
            k_wrap=config_dict.get('k_wrap', 1.0),
            binding_sites=config_dict.get('binding_sites', 14),
            prot_k_unbind=prot_params.get('k_unbind', 89.7),
            prot_k_bind=prot_params.get('k_bind', 1.0),
            prot_p_conc=prot_params.get('p_conc', 0.0),
            prot_cooperativity=prot_params.get('cooperativity', 0.0),
            tau_max=config_dict.get('tau_max', 10000.0),
            tau_steps=config_dict.get('tau_steps', 1000),
            inf_protamine=config_dict.get('inf_protamine', True),
            renucleation=config_dict.get('renucleation', False),
            replicates=config_dict.get('replicates', 20),
            batch_size=config_dict.get('batch_size', 10),
            n_workers=config_dict.get('n_workers', 4),
            flush_every=config_dict.get('flush_every', 10000),
            save_trajectories=config_dict.get('save_trajectories', False),
            maxpoints_saved_trajectories=config_dict.get('maxpoints_saved_trajectories', 100),
        )
