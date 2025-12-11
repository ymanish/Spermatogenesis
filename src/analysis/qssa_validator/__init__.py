"""
QSSA Validator Package
======================

A modular package for validating the Quasi-Steady-State Approximation (QSSA)
in nucleosome-protamine simulations.

The QSSA is valid when protamine binding/unbinding equilibrates much faster
than nucleosome wrapping/unwrapping dynamics (epsilon = tau_prot/tau_slow << 1).

Main API:
---------
from src.analysis.qssa_validator import (
    validate_qssa_for_system,
    validate_qssa_for_nucleosome,
    compute_protamine_fast_timescale
)

Components:
-----------
- core: Timescale computation functions (tau_prot, tau_slow)
- validation: QSSA validation logic for nucleosomes and systems
- io: File I/O, reporting, and data export
- visualization: Plotting functions for QSSA analysis
- cli: Command-line interface

Data Classes:
-------------
- QSSAValidationResult: Per-nucleosome validation results
- SystemQSSAResult: System-wide validation results

Author: MY
Date: 2025-11-27
"""

from pathlib import Path
from typing import Optional

# Configuration
from .config import QSSAConfig

# Data classes
from .validation import (
    QSSAValidationResult,
    SystemQSSAResult
)

# Core functionality
from .core import (
    compute_protamine_fast_timescale,
    compute_nucleosome_slow_timescale_per_ij
)

# Validation
from .validation import (
    validate_qssa_for_nucleosome,
    validate_qssa_for_system
)

# I/O and reporting
from .io import (
    print_qssa_summary,
    generate_qssa_report,
    save_qssa_data
)

# CLI - imported lazily to avoid circular dependencies
def run_qssa_cli():
    """Run the QSSA validator CLI."""
    from .cli import main
    return main()

# Package version
__version__ = "1.0.0"

# High-level convenience function
def validate_qssa(
    nucleosomes,
    protamines,
    threshold: float = 0.1,
    beta: float = 1.0,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> SystemQSSAResult:
    """
    Convenience function to validate QSSA and optionally save results.
    
    Args:
        nucleosomes: Nucleosome or Nucleosomes instance
        protamines: protamines instance
        threshold: QSSA validity threshold (default: 0.1)
        beta: Inverse temperature (default: 1.0)
        output_dir: Optional directory to save reports and data
        verbose: Print summary to console
        
    Returns:
        SystemQSSAResult with validation details
        
    Example:
        >>> from src.analysis.qssa_validator import validate_qssa
        >>> result = validate_qssa(nucs, prot, output_dir=Path("output/qssa"))
        >>> if result.system_qssa_valid:
        >>>     print("QSSA is valid! Can use hybrid simulator.")
    """
    # Import here to avoid circular dependencies
    from src.core.nucleosomes import Nucleosome, Nucleosomes
    
    # Handle single nucleosome vs multiple
    if isinstance(nucleosomes, Nucleosome):
        from src.core.nucleosomes import Nucleosomes
        nucs = Nucleosomes([nucleosomes])
    else:
        nucs = nucleosomes
    
    # Validate
    result = validate_qssa_for_system(nucs, protamines, threshold=threshold, beta=beta)
    
    # Print summary
    if verbose:
        print_qssa_summary(result, verbose=verbose)
    
    # Save results if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = output_dir / "qssa_validation_report.txt"
        generate_qssa_report(result, report_path, include_details=True)
        
        # Save data
        data_path = output_dir / "qssa_validation_data.tsv"
        save_qssa_data(result, data_path)
    
    return result


__all__ = [
    # Main API
    'validate_qssa',
    'validate_qssa_for_system',
    'validate_qssa_for_nucleosome',
    'compute_protamine_fast_timescale',
    
    # Configuration
    'QSSAConfig',
    
    # Data classes
    'QSSAValidationResult',
    'SystemQSSAResult',
    
    # I/O
    'print_qssa_summary',
    'generate_qssa_report',
    'save_qssa_data',
    
    # CLI
    'run_qssa_cli',
    
    # Version
    '__version__',
]
