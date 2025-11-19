"""
RMST Replicate Estimator Package
==================================

A modular package for RMST (Restricted Mean Survival Time) based replicate
estimation in nucleosome simulations.

Main API:
---------
from src.analysis.rmst_estimator import estimate_replicates_rmst

Components:
-----------
- core: RMST computation functions
- sampling: Memory-efficient nucleosome sampling
- simulation: Parallel simulation execution
- variance: Statistical analysis
- visualization: Plotting functions
- io: File I/O and reporting
- cli: Command-line interface

Author: MY
Date: 2025-11-14
"""

from pathlib import Path
from typing import Optional
from dataclasses import asdict

# Core functionality
from .core import (
    compute_rmst_from_survival,
    extract_survival_curve_from_timeseries
)

# Sampling
from .sampling import (
    sample_nucleosomes,
    reservoir_sample,
    batcher
)

# Simulation
from .simulation import run_rmst_pilot_study

# Variance analysis
from .variance import (
    compute_rmst_variance_components,
    calculate_required_replicates_rmst,
    analyze_rmst_replicates
)

# Visualization
from .visualization import plot_rmst_analysis

# I/O
from .io import (
    print_rmst_analysis_report,
    save_rmst_analysis_text,
    save_rmst_analysis_json,
    create_output_directory,
    save_run_metadata
)

# Type imports
from src.config.custom_type import PilotConfig, RMSTAnalysis


__all__ = [
    # Main API
    'estimate_replicates_rmst',
    
    # Core
    'compute_rmst_from_survival',
    'extract_survival_curve_from_timeseries',
    
    # Sampling
    'sample_nucleosomes',
    'reservoir_sample',
    'batcher',
    
    # Simulation
    'run_rmst_pilot_study',
    
    # Variance
    'compute_rmst_variance_components',
    'calculate_required_replicates_rmst',
    'analyze_rmst_replicates',
    
    # Visualization
    'plot_rmst_analysis',
    
    # I/O
    'print_rmst_analysis_report',
    'save_rmst_analysis_text',
    'save_rmst_analysis_json',
    'create_output_directory',
    'save_run_metadata',
    
    # Types
    'PilotConfig',
    'RMSTAnalysis'
]


def estimate_replicates_rmst(
    file_path: Path,
    config: PilotConfig,
    condition_label: str = "condition",
    save_path: Optional[Path] = None,
    plot: bool = True,
    n_workers: int = 1,
    tolerance: Optional[float] = None,
    random_sample: bool = True,
    seed: Optional[int] = None,
    batch_size: int = 1
) -> RMSTAnalysis:
    """
    Complete RMST-based replicate estimation pipeline.
    
    This is the main entry point for the RMST estimator package.
    
    Args:
        file_path: Path to nucleosome data file
        config: PilotConfig object with simulation parameters
        condition_label: Label for this experimental condition
        save_path: Optional path to save outputs
        plot: Whether to generate visualization (default: True)
        n_workers: Number of parallel workers (default: 1)
        tolerance: Tolerance ε for N_rep calculation (optional)
        random_sample: Use reservoir sampling vs sequential (default: True)
        seed: Random seed for reproducibility (optional)
        batch_size: Nucleosomes per batch for parallel processing (default: 1)
    
    Returns:
        RMSTAnalysis object with variance components and recommendations
    
    Examples:
        >>> from src.analysis.rmst_estimator import estimate_replicates_rmst
        >>> from src.config.custom_type import PilotConfig
        >>> 
        >>> config = PilotConfig(
        ...     k_wrap=1.0,
        ...     prot_k_unbind=89.7,
        ...     prot_k_bind=1.0,
        ...     prot_p_conc=100.0,
        ...     prot_cooperativity=4.5,
        ...     n_pilot_nucleosomes=50,
        ...     n_pilot_replicates=20,
        ...     tau_max=10000.0,
        ...     tau_steps=1000
        ... )
        >>> 
        >>> analysis = estimate_replicates_rmst(
        ...     file_path=Path("data.tsv"),
        ...     config=config,
        ...     condition_label="RET_prot100_coop4.5",
        ...     n_workers=10,
        ...     tolerance=0.05,
        ...     random_sample=True,
        ...     seed=42
        ... )
        >>> 
        >>> print(f"R-ratio: {analysis.R:.4f}")
        >>> print(f"Required replicates: {analysis.n_reps_required}")
    
    Pipeline Steps:
        1. Sample nucleosomes (memory-efficient)
        2. Run simulations in parallel
        3. Compute RMST for each replicate
        4. Analyze variance components
        5. Generate recommendations
        6. Create visualizations (optional)
        7. Save results (optional)
    
    Notes:
        - Uses ProcessPoolExecutor for parallelization
        - Memory-efficient: never loads full dataset
        - Progress bars show completion status
        - Reproducible with seed parameter
    """
    # Step 1: Run pilot study
    rmst_data = run_rmst_pilot_study(
        file_path=file_path,
        config=config,
        n_workers=n_workers,
        verbose=True,
        random_sample=random_sample,
        seed=seed,
        batch_size=batch_size
    )
    
    # Step 2: Analyze variance components
    analysis = analyze_rmst_replicates(
        rmst_data=rmst_data,
        condition_label=condition_label,
        tolerance=tolerance,
        tau_max=config.tau_max,
        tau_steps=config.tau_steps
    )
    
    # Step 3: Print console report
    print_rmst_analysis_report(analysis)
    
    # Step 4: Generate visualization
    if plot:
        plot_rmst_analysis(rmst_data, analysis, save_path)
    
    # Step 5: Save outputs
    if save_path:
        save_path = Path(save_path)
        
        # Save text report
        text_path = save_path / f"rmst_analysis_{condition_label.replace(' ', '_')}.txt"
        save_rmst_analysis_text(analysis, text_path)
        
        # Save JSON
        json_path = save_path / f"rmst_analysis_{condition_label.replace(' ', '_')}.json"
        save_rmst_analysis_json(analysis, rmst_data, json_path)
    
    return analysis


# Version info
__version__ = "2.0.0"
__author__ = "MY"
__date__ = "2025-11-14"
