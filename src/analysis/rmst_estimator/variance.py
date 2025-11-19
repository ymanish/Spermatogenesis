"""
Variance Analysis Functions
============================

Statistical analysis of RMST data to compute variance components
and replicate recommendations.

Author: MY
Date: 2025-11-14
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from src.config.custom_type import RMSTAnalysis


def compute_rmst_variance_components(
    rmst_data: Dict[str, List[float]]
) -> Tuple[float, float, List[float], List[float]]:
    """
    Compute within and between variance components from RMST data.
    
    Variance decomposition:
    - σ²_within = (1/N) Σ_i s²_i  (pooled within-nucleosome variance)
    - σ²_between = Var(m̄_i)       (between-nucleosome variance)
    
    Args:
        rmst_data: Dict mapping nucleosome_key -> [rmst values for replicates]
    
    Returns:
        Tuple of (sigma_within_sq, sigma_between_sq, nucleosome_means, nucleosome_stds)
    
    Examples:
        >>> rmst_data = {
        ...     'nuc_001': [100.5, 102.3, 99.8, 101.2, 100.1],
        ...     'nuc_002': [95.2, 96.8, 94.5, 95.9, 96.1],
        ...     'nuc_003': [110.3, 112.1, 109.8, 111.5, 110.9]
        ... }
        >>> sigma_w_sq, sigma_b_sq, means, stds = compute_rmst_variance_components(rmst_data)
        >>> print(f"Within variance: {sigma_w_sq:.2f}")
        >>> print(f"Between variance: {sigma_b_sq:.2f}")
    
    Notes:
        - Within variance represents simulation noise (stochastic variability)
        - Between variance represents biological heterogeneity (sequence effects)
        - R = σ²_within / (σ²_within + σ²_between) gives relative contribution
    """
    nucleosome_means = []
    nucleosome_vars = []
    
    for nuc_key, rmst_values in rmst_data.items():
        rmst_array = np.array(rmst_values)
        
        # Per-nucleosome mean and variance
        mean_i = np.mean(rmst_array)
        var_i = np.var(rmst_array, ddof=1)  # Sample variance (Bessel correction)
        
        nucleosome_means.append(mean_i)
        nucleosome_vars.append(var_i)
    
    # Within-nucleosome variance (pooled)
    sigma_within_sq = np.mean(nucleosome_vars)
    
    # Between-nucleosome variance
    sigma_between_sq = np.var(nucleosome_means, ddof=1)
    
    # Convert to standard deviations
    nucleosome_stds = [np.sqrt(v) for v in nucleosome_vars]
    
    return sigma_within_sq, sigma_between_sq, nucleosome_means, nucleosome_stds


def calculate_required_replicates_rmst(
    sigma_within_sq: float,
    sigma_between_sq: float,
    tolerance: float
) -> int:
    """
    Calculate required number of replicates from R-ratio and tolerance.
    
    Formula: N_rep >= ⌈R / ((1-R) × ε)⌉
    
    where R = σ²_within / (σ²_within + σ²_between)
    
    Args:
        sigma_within_sq: Within-nucleosome variance (simulation noise)
        sigma_between_sq: Between-nucleosome variance (sequence heterogeneity)
        tolerance: Tolerance ε (e.g., 0.1 for 10% acceptable error)
    
    Returns:
        Required number of replicates (rounded up to integer)
        Returns np.inf if no heterogeneity (σ²_between = 0)
    
    Examples:
        >>> # High simulation noise (R = 0.8)
        >>> n_rep = calculate_required_replicates_rmst(
        ...     sigma_within_sq=0.8,
        ...     sigma_between_sq=0.2,
        ...     tolerance=0.1
        ... )
        >>> print(f"Need {n_rep} replicates")
        Need 40 replicates
        
        >>> # Low simulation noise (R = 0.1)
        >>> n_rep = calculate_required_replicates_rmst(
        ...     sigma_within_sq=0.1,
        ...     sigma_between_sq=0.9,
        ...     tolerance=0.1
        ... )
        >>> print(f"Need {n_rep} replicates")
        Need 2 replicates
    
    Notes:
        - Returns 1 if σ²_within = 0 (no simulation noise)
        - Returns np.inf if σ²_between = 0 (no heterogeneity)
        - Higher tolerance → fewer replicates needed
        - Higher R (more noise) → more replicates needed
    """
    if sigma_within_sq == 0:
        return 1
    
    if sigma_between_sq == 0:
        return np.inf
    
    R = sigma_within_sq / (sigma_within_sq + sigma_between_sq)
    n_required = R / ((1 - R) * tolerance)
    
    return int(np.ceil(n_required))


def analyze_rmst_replicates(
    rmst_data: Dict[str, List[float]],
    condition_label: str = "condition",
    tolerance: Optional[float] = None,
    tau_max: Optional[float] = None,
    tau_steps: Optional[int] = None
) -> RMSTAnalysis:
    """
    Analyze RMST pilot data and compute replicate recommendations.
    
    Complete analysis pipeline:
    1. Compute variance components
    2. Calculate R-ratio
    3. Generate recommendations
    4. Calculate required replicates (if tolerance provided)
    
    Args:
        rmst_data: Dict of nucleosome_key -> RMST values
        condition_label: Label for this experimental condition
        tolerance: Optional tolerance ε for N_rep calculation
        tau_max: Maximum tau (for reporting)
        tau_steps: Number of tau steps (for reporting)
    
    Returns:
        RMSTAnalysis object with all results
    
    Examples:
        >>> rmst_data = {
        ...     'nuc_001': [100.5, 102.3, 99.8],
        ...     'nuc_002': [95.2, 96.8, 94.5],
        ...     # ... more nucleosomes ...
        ... }
        >>> analysis = analyze_rmst_replicates(
        ...     rmst_data,
        ...     condition_label="RET_prot100_coop4.5",
        ...     tolerance=0.05,
        ...     tau_max=10000.0,
        ...     tau_steps=1000
        ... )
        >>> print(f"R-ratio: {analysis.R:.4f}")
        >>> print(f"Recommendation: {analysis.recommended_replicates}")
    
    Notes:
        - Recommendation based on R thresholds:
          - R < 0.1: 1 replicate (sequence dominates)
          - 0.1 ≤ R < 0.5: 2-3 replicates (mixed)
          - R ≥ 0.5: 5-10 replicates (noise dominates)
        - Tolerance-based calculation more precise
    """
    # Compute variance components
    sigma_within_sq, sigma_between_sq, nuc_means, nuc_stds = \
        compute_rmst_variance_components(rmst_data)
    
    # Compute R-ratio
    R = sigma_within_sq / (sigma_within_sq + sigma_between_sq)
    
    # Recommendation based on R thresholds
    if R < 0.1:
        recommendation = "1 replicate (sequence heterogeneity dominates)"
    elif R < 0.5:
        recommendation = "2-3 replicates (moderate simulation noise)"
    else:
        recommendation = "5-10 replicates (high simulation noise)"
    
    # Calculate required replicates if tolerance provided
    n_reps_required = None
    if tolerance is not None:
        n_reps_required = calculate_required_replicates_rmst(
            sigma_within_sq, sigma_between_sq, tolerance
        )
    
    # Overall statistics
    all_rmst = []
    for values in rmst_data.values():
        all_rmst.extend(values)
    
    mean_rmst = np.mean(all_rmst)
    std_rmst = np.std(all_rmst)
    
    # Compute delta_tau
    delta_tau = tau_max / tau_steps if tau_max and tau_steps else None
    
    return RMSTAnalysis(
        sigma_within_sq=sigma_within_sq,
        sigma_between_sq=sigma_between_sq,
        R=R,
        recommended_replicates=recommendation,
        n_nucleosomes=len(rmst_data),
        n_replicates=len(next(iter(rmst_data.values()))),
        condition_label=condition_label,
        mean_rmst=mean_rmst,
        std_rmst=std_rmst,
        nucleosome_mean_rmsts=nuc_means,
        nucleosome_std_rmsts=nuc_stds,
        tolerance=tolerance,
        n_reps_required=n_reps_required,
        tau_max=tau_max,
        delta_tau=delta_tau
    )
