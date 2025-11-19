"""
I/O and Reporting Functions
============================

File operations and report generation for RMST analysis results.

Author: MY
Date: 2025-11-14
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from src.config.custom_type import RMSTAnalysis


def print_rmst_analysis_report(analysis: RMSTAnalysis) -> None:
    """
    Print human-readable RMST analysis report to console.
    
    Args:
        analysis: RMSTAnalysis object with all results
    
    Examples:
        >>> analysis = analyze_rmst_replicates(rmst_data, ...)
        >>> print_rmst_analysis_report(analysis)
        # Prints formatted report to console
    """
    print(f"\n{'='*70}")
    print(f"RMST-BASED REPLICATE ESTIMATION ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Condition: {analysis.condition_label}")
    print(f"Metric: RMST (Restricted Mean Survival Time)")
    print(f"N Nucleosomes: {analysis.n_nucleosomes}")
    print(f"N Replicates per nucleosome: {analysis.n_replicates}")
    if analysis.tau_max:
        print(f"Integration range: τ ∈ [0, {analysis.tau_max:.1f}]")
    if analysis.delta_tau:
        print(f"Grid spacing: Δτ = {analysis.delta_tau:.4f}")
    print()
    
    print(f"Summary Statistics:")
    print(f"  Mean RMST: {analysis.mean_rmst:.4f}")
    print(f"  Std RMST:  {analysis.std_rmst:.4f}")
    print()
    
    print(f"Variance Components:")
    print(f"  σ²_within  (simulation noise):      {analysis.sigma_within_sq:.6f}")
    print(f"  σ²_between (sequence heterogeneity): {analysis.sigma_between_sq:.6f}")
    print(f"  σ_within:                            {np.sqrt(analysis.sigma_within_sq):.6f}")
    print(f"  σ_between:                           {np.sqrt(analysis.sigma_between_sq):.6f}")
    print()
    
    print(f"Variance Ratio:")
    print(f"  R = σ²_within / (σ²_within + σ²_between)")
    print(f"  R = {analysis.R:.4f}")
    print()
    
    print(f"Interpretation:")
    if analysis.R < 0.1:
        print(f"  → Sequence heterogeneity DOMINATES (R < 0.1)")
        print(f"  → RMST varies mainly due to nucleosome differences")
    elif analysis.R < 0.5:
        print(f"  → MIXED contribution (0.1 ≤ R < 0.5)")
        print(f"  → Both simulation noise and sequence matter")
    else:
        print(f"  → Simulation noise DOMINATES (R ≥ 0.5)")
        print(f"  → RMST varies mainly due to stochastic effects")
    print()
    
    print(f"RECOMMENDATION:")
    print(f"  {analysis.recommended_replicates}")
    print()
    
    if analysis.tolerance is not None:
        print(f"Tolerance-Based Calculation:")
        print(f"  Tolerance (ε): {analysis.tolerance}")
        print(f"  Formula: N_rep >= ⌈R / ((1-R) × ε)⌉")
        if np.isinf(analysis.n_reps_required):
            print(f"  Required replicates: ∞ (no heterogeneity)")
        else:
            print(f"  Required replicates: {analysis.n_reps_required}")
        print()
    
    print(f"Per-Nucleosome Statistics:")
    print(f"  Mean of means: {np.mean(analysis.nucleosome_mean_rmsts):.4f}")
    print(f"  Mean of stds:  {np.mean(analysis.nucleosome_std_rmsts):.4f}")
    print(f"  Range of means: [{np.min(analysis.nucleosome_mean_rmsts):.4f}, "
          f"{np.max(analysis.nucleosome_mean_rmsts):.4f}]")
    print()
    print(f"{'='*70}\n")


def save_rmst_analysis_text(analysis: RMSTAnalysis, output_path: Path) -> None:
    """
    Save RMST analysis to formatted text file.
    
    Args:
        analysis: RMSTAnalysis object
        output_path: Path to save text report
    
    Examples:
        >>> save_rmst_analysis_text(analysis, Path("output/report.txt"))
        ✓ Saved RMST text report to output/report.txt
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RMST-BASED REPLICATE ESTIMATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Condition:     {analysis.condition_label}\n")
        f.write(f"Metric:        RMST (Restricted Mean Survival Time)\n")
        f.write(f"N Nucleosomes: {analysis.n_nucleosomes}\n")
        f.write(f"N Replicates:  {analysis.n_replicates}\n")
        if analysis.tau_max:
            f.write(f"Integration:   τ ∈ [0, {analysis.tau_max:.1f}], Δτ = {analysis.delta_tau:.4f}\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Mean RMST: {analysis.mean_rmst:.4f}\n")
        f.write(f"Std RMST:  {analysis.std_rmst:.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("VARIANCE COMPONENTS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"σ²_within  (simulation noise):      {analysis.sigma_within_sq:.6f}\n")
        f.write(f"σ²_between (sequence heterogeneity): {analysis.sigma_between_sq:.6f}\n")
        f.write(f"σ_within:                            {np.sqrt(analysis.sigma_within_sq):.6f}\n")
        f.write(f"σ_between:                           {np.sqrt(analysis.sigma_between_sq):.6f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("VARIANCE RATIO\n")
        f.write("-"*80 + "\n\n")
        f.write(f"R = σ²_within / (σ²_within + σ²_between)\n")
        f.write(f"R = {analysis.R:.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-"*80 + "\n\n")
        if analysis.R < 0.1:
            f.write("• Sequence heterogeneity DOMINATES (R < 0.1)\n")
            f.write("• RMST varies mainly due to nucleosome sequence differences\n")
        elif analysis.R < 0.5:
            f.write("• MIXED contribution (0.1 ≤ R < 0.5)\n")
            f.write("• Both simulation noise and sequence heterogeneity matter\n")
        else:
            f.write("• Simulation noise DOMINATES (R ≥ 0.5)\n")
            f.write("• RMST varies mainly due to stochastic simulation effects\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{analysis.recommended_replicates}\n\n")
        
        if analysis.tolerance is not None:
            f.write("-"*80 + "\n")
            f.write("TOLERANCE-BASED CALCULATION\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Tolerance (ε): {analysis.tolerance}\n")
            f.write(f"Formula: N_rep >= ⌈R / ((1-R) × ε)⌉\n")
            if np.isinf(analysis.n_reps_required):
                f.write(f"Required replicates: ∞ (no heterogeneity)\n")
            else:
                f.write(f"Required replicates: {analysis.n_reps_required}\n")
            f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("PER-NUCLEOSOME STATISTICS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Mean of nucleosome means: {np.mean(analysis.nucleosome_mean_rmsts):.4f}\n")
        f.write(f"Mean of nucleosome stds:  {np.mean(analysis.nucleosome_std_rmsts):.4f}\n")
        f.write(f"Range of means: [{np.min(analysis.nucleosome_mean_rmsts):.4f}, "
                f"{np.max(analysis.nucleosome_mean_rmsts):.4f}]\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Saved RMST text report to {output_path}")


def save_rmst_analysis_json(
    analysis: RMSTAnalysis,
    rmst_data: Dict[str, list],
    output_path: Path
) -> None:
    """
    Save RMST analysis to JSON format.
    
    Args:
        analysis: RMSTAnalysis object
        rmst_data: Raw RMST data (nucleosome_key -> RMST values)
        output_path: Path to save JSON file
    
    Examples:
        >>> save_rmst_analysis_json(analysis, rmst_data, Path("output/data.json"))
        ✓ Saved JSON to output/data.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from dataclasses import asdict
    
    data = asdict(analysis)
    data['timestamp'] = datetime.now().isoformat()
    data['rmst_data'] = {k: [float(v) for v in vals] for k, vals in rmst_data.items()}
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved JSON to {output_path}")


def create_output_directory(
    base_dir: Path,
    dataset: str,
    k_wrap: float,
    prot_conc: float,
    cooperativity: float,
    n_nucs: int,
    n_reps: int,
    seed: int
) -> Path:
    """
    Create hierarchical output directory based on parameters.
    
    Format: {base_dir}/{dataset}/kwrap{k}_prot{conc}_coop{coop}/n{nucs}_r{reps}_s{seed}_{timestamp}
    
    Args:
        base_dir: Base output directory
        dataset: Dataset type (e.g., 'bound', 'unbound')
        k_wrap: Wrapping energy parameter
        prot_conc: Protamine concentration
        cooperativity: Cooperativity value
        n_nucs: Number of nucleosomes
        n_reps: Number of replicates
        seed: Random seed
    
    Returns:
        Path to created directory
    
    Examples:
        >>> output_dir = create_output_directory(
        ...     Path("output"),
        ...     "bound",
        ...     k_wrap=1.0,
        ...     prot_conc=100.0,
        ...     cooperativity=4.5,
        ...     n_nucs=50,
        ...     n_reps=20,
        ...     seed=42
        ... )
        >>> print(output_dir)
        output/bound/kwrap1.0_prot100_coop4.5/n50_r20_s42_20251114_143052
    """
    # Format parameter values for directory name
    prot_str = f"prot{prot_conc:.0f}" if prot_conc > 0 else "noprot"
    coop_str = f"coop{cooperativity:.1f}" if cooperativity > 0 else "nocoop"
    
    # Create hierarchical directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = (
        Path(base_dir) / 
        dataset / 
        f"kwrap{k_wrap:.1f}_{prot_str}_{coop_str}" / 
        f"n{n_nucs}_r{n_reps}_s{seed}_{timestamp}"
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_run_metadata(
    output_dir: Path,
    metadata: Dict[str, Any]
) -> None:
    """
    Save run metadata to JSON.
    
    Args:
        output_dir: Output directory
        metadata: Dictionary with metadata fields
    
    Examples:
        >>> metadata = {
        ...     'timestamp': datetime.now().isoformat(),
        ...     'dataset': 'bound',
        ...     'parameters': {...},
        ...     'analysis': {...}
        ... }
        >>> save_run_metadata(Path("output"), metadata)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'run_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
