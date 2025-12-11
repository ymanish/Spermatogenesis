"""
QSSA Configuration
==================

Configuration class for QSSA validation parameters.

Author: MY
Date: 2025-11-27
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class QSSAConfig:
    """
    Configuration for QSSA validation.
    
    This class encapsulates all parameters needed for QSSA validation,
    following the same pattern as PilotConfig for consistency.
    
    Attributes:
        Nucleosome parameters:
            k_wrap: Wrapping rate constant (1/s)
            kT: Thermal energy (dimensionless, default: 1.0)
            binding_sites: Number of DNA-histone binding sites (default: 14)
        
        Protamine parameters:
            prot_k_unbind: Protamine unbinding rate (1/s)
            prot_k_bind: Protamine binding rate (1/(μM·s))
            prot_p_conc: Protamine concentration (μM)
            prot_cooperativity: Cooperativity parameter J (dimensionless)
        
        QSSA parameters:
            threshold: QSSA validity threshold for epsilon (default: 0.1)
            beta: Inverse temperature (1/kT) (default: 1.0)
        
        Execution parameters:
            max_nucleosomes: Maximum number of nucleosomes to validate
            verbose: Whether to print detailed output
            output_dir: Directory to save results (optional)
            
    Examples:
        >>> config = QSSAConfig(
        ...     k_wrap=21.0,
        ...     prot_p_conc=100.0,
        ...     prot_k_unbind=100.0,
        ...     threshold=0.1
        ... )
    """
    
    # Nucleosome parameters
    k_wrap: float = 21.0
    kT: float = 1.0
    binding_sites: int = 14
    
    # Protamine parameters
    prot_k_unbind: float = 100.0
    prot_k_bind: float = 1.0
    prot_p_conc: float = 100.0
    prot_cooperativity: float = 0.0
    
    # QSSA parameters
    threshold: float = 0.1
    beta: float = 1.0
    
    # Execution parameters
    max_nucleosomes: Optional[int] = None
    verbose: bool = True
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate positive values
        if self.k_wrap <= 0:
            raise ValueError(f"k_wrap must be positive, got {self.k_wrap}")
        if self.kT <= 0:
            raise ValueError(f"kT must be positive, got {self.kT}")
        if self.binding_sites <= 0:
            raise ValueError(f"binding_sites must be positive, got {self.binding_sites}")
        if self.prot_k_unbind < 0:
            raise ValueError(f"prot_k_unbind must be non-negative, got {self.prot_k_unbind}")
        if self.prot_k_bind < 0:
            raise ValueError(f"prot_k_bind must be non-negative, got {self.prot_k_bind}")
        if self.prot_p_conc < 0:
            raise ValueError(f"prot_p_conc must be non-negative, got {self.prot_p_conc}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        
        # Convert output_dir to Path if it's a string
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
    
    def to_dict(self):
        """Convert configuration to dictionary for serialization."""
        return {
            'nucleosome': {
                'k_wrap': self.k_wrap,
                'kT': self.kT,
                'binding_sites': self.binding_sites
            },
            'protamine': {
                'k_unbind': self.prot_k_unbind,
                'k_bind': self.prot_k_bind,
                'p_conc': self.prot_p_conc,
                'cooperativity': self.prot_cooperativity
            },
            'qssa': {
                'threshold': self.threshold,
                'beta': self.beta
            },
            'execution': {
                'max_nucleosomes': self.max_nucleosomes,
                'verbose': self.verbose,
                'output_dir': str(self.output_dir) if self.output_dir else None
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create configuration from dictionary."""
        return cls(
            k_wrap=data['nucleosome']['k_wrap'],
            kT=data['nucleosome']['kT'],
            binding_sites=data['nucleosome']['binding_sites'],
            prot_k_unbind=data['protamine']['k_unbind'],
            prot_k_bind=data['protamine']['k_bind'],
            prot_p_conc=data['protamine']['p_conc'],
            prot_cooperativity=data['protamine']['cooperativity'],
            threshold=data['qssa']['threshold'],
            beta=data['qssa']['beta'],
            max_nucleosomes=data['execution']['max_nucleosomes'],
            verbose=data['execution']['verbose'],
            output_dir=Path(data['execution']['output_dir']) if data['execution']['output_dir'] else None
        )
    
    def __repr__(self):
        """String representation of configuration."""
        return (
            f"QSSAConfig(\n"
            f"  Nucleosome: k_wrap={self.k_wrap}, kT={self.kT}, "
            f"binding_sites={self.binding_sites}\n"
            f"  Protamine: P_conc={self.prot_p_conc}, k_bind={self.prot_k_bind}, "
            f"k_unbind={self.prot_k_unbind}, cooperativity={self.prot_cooperativity}\n"
            f"  QSSA: threshold={self.threshold}, beta={self.beta}\n"
            f"  Execution: max_nucs={self.max_nucleosomes}, verbose={self.verbose}\n"
            f")"
        )
