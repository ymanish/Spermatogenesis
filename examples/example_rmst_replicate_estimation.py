"""
Example: RMST-Based Replicate Estimation
=========================================

This example demonstrates using RMST (Restricted Mean Survival Time) instead
of detachment time for replicate estimation.

Advantages of RMST:
- Uses entire survival curve, not just endpoint
- Always defined (even if nucleosomes don't fully detach)
- Captures dynamics throughout simulation
- More robust for stable nucleosomes

Author: MY
Date: 2025-11-12
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

from pathlib import Path
# from src.analysis.replicate_estimator_rmst import estimate_replicates_rmst
from src.config.custom_type import PilotConfig
from src.analysis.rmst_estimator import estimate_replicates_rmst
# =============================================================================
# CONFIGURATION
# =============================================================================

# Data files
DATA_FILE_BOUND = Path(
    "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/"
    "hamnucret_data/exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
)

DATA_FILE_UNBOUND = Path(
    "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/"
    "hamnucret_data/exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"
)

# Output directory
OUTPUT_DIR = Path("output/rmst_replicate_estimation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXAMPLE 1: No Protamine (Stable Nucleosomes)
# =============================================================================

def example_no_protamine():
    """
    Example with no protamine - nucleosomes rarely detach.
    
    RMST is ideal here because:
    - detachment_time would be mostly NaN
    - RMST captures partial unwrapping dynamics
    - Always defined regardless of full detachment
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: No Protamine (RMST captures stability)")
    print("="*80 + "\n")
    
    config = PilotConfig(
        prot_p_conc=0.0,  # No protamine
        prot_cooperativity=0.0,
        n_pilot_nucleosomes=20,
        n_pilot_replicates=10,
        tau_max=10000.0,
        tau_steps=1000
    )
    
    analysis = estimate_replicates_rmst(
        file_path=DATA_FILE_BOUND,
        config=config,
        condition_label="bound_noprot",
        save_path=OUTPUT_DIR / "example1_noprot",
        plot=True,
        n_workers=20,
        tolerance=0.1,
        seed=40, 
        batch_size=1
    )
    
    print("\nKey Results:")
    print(f"  R = {analysis.R:.4f}")
    print(f"  Mean RMST = {analysis.mean_rmst:.2f}")
    print(f"  Required replicates (ε=0.1): {analysis.n_reps_required}")
    
    return analysis


# =============================================================================
# EXAMPLE 2: Medium Protamine with Cooperativity
# =============================================================================

def example_medium_protamine():
    """
    Example with medium protamine - mixed detachment behavior.
    
    RMST advantages:
    - Captures both fast and slow detachment
    - Handles partial NaN values gracefully
    - Integrates full detachment dynamics
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Medium Protamine + Cooperativity")
    print("="*80 + "\n")
    
    config = PilotConfig(
        k_wrap=1.0,
        prot_p_conc=200.0,
        prot_cooperativity=4.5,
        n_pilot_nucleosomes=20,
        n_pilot_replicates=2,
        tau_max=1000.0,
        tau_steps=1000
    )
    
    analysis = estimate_replicates_rmst(
        file_path=DATA_FILE_UNBOUND,
        config=config,
        condition_label="unbound_prot200_coop4.5",
        save_path=OUTPUT_DIR / "example2_medium",
        plot=True,
        n_workers=20,
        tolerance=0.05,
        seed=123
    )
    
    print("\nKey Results:")
    print(f"  R = {analysis.R:.4f}")
    print(f"  Mean RMST = {analysis.mean_rmst:.2f}")
    print(f"  Required replicates (ε=0.05): {analysis.n_reps_required}")
    
    return analysis


# =============================================================================
# EXAMPLE 3: Compare RMST vs Detachment Time
# =============================================================================

def example_comparison():
    """
    Compare RMST-based estimation with detachment time-based estimation.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: RMST vs Detachment Time Comparison")
    print("="*80 + "\n")
    
    from src.analysis.replicate_estimator import estimate_replicates
    
    config = PilotConfig(
        prot_p_conc=100.0,
        prot_cooperativity=2.0,
        n_pilot_nucleosomes=30,
        n_pilot_replicates=15,
        tau_max=10000.0,
        tau_steps=1000
    )
    
    # Method 1: RMST
    print("\n--- Method 1: RMST-Based ---")
    rmst_analysis = estimate_replicates_rmst(
        file_path=DATA_FILE_BOUND,
        config=config,
        condition_label="bound_prot100_rmst",
        save_path=OUTPUT_DIR / "example3_comparison",
        plot=True,
        n_workers=8,
        tolerance=0.05,
        seed=456
    )
    
    # Method 2: Detachment Time
    print("\n--- Method 2: Detachment Time-Based ---")
    detach_analysis = estimate_replicates(
        file_path=DATA_FILE_BOUND,
        config=config,
        condition_label="bound_prot100_detach",
        metric="detachment_time",
        save_path=OUTPUT_DIR / "example3_comparison",
        plot=True,
        n_workers=8,
        tolerance=0.05,
        seed=456
    )
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<25} {'RMST':<15} {'Detachment Time':<15}")
    print("-"*55)
    print(f"{'R value':<25} {rmst_analysis.R:<15.4f} {detach_analysis.R:<15.4f}")
    print(f"{'σ²_within':<25} {rmst_analysis.sigma_within_sq:<15.4f} "
          f"{detach_analysis.sigma_within_sq:<15.4f}")
    print(f"{'σ²_between':<25} {rmst_analysis.sigma_between_sq:<15.4f} "
          f"{detach_analysis.sigma_between_sq:<15.4f}")
    print(f"{'Required reps (ε=0.05)':<25} {rmst_analysis.n_reps_required:<15} "
          f"{detach_analysis.n_reps_required:<15}")
    print()
    
    print("Interpretation:")
    print("- RMST tends to have LOWER variance ratio (more stable)")
    print("- RMST captures full dynamics, not just endpoint")
    print("- Both methods valid, RMST better for stable nucleosomes")
    
    return rmst_analysis, detach_analysis


# =============================================================================
# EXAMPLE 4: High Protamine (Highly Stable)
# =============================================================================

def example_high_protamine():
    """
    Example with very high protamine - nucleosomes very stable.
    
    This is where RMST really shines:
    - detachment_time would be almost all NaN
    - RMST still captures subtle differences in stability
    - Meaningful analysis even without full detachment
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: High Protamine (RMST essential)")
    print("="*80 + "\n")
    
    config = PilotConfig(
        prot_p_conc=1000.0,
        prot_cooperativity=4.5,
        n_pilot_nucleosomes=20,
        n_pilot_replicates=10,
        tau_max=10000.0,
        tau_steps=1000
    )
    
    analysis = estimate_replicates_rmst(
        file_path=DATA_FILE_BOUND,
        config=config,
        condition_label="bound_prot1000_coop4.5",
        save_path=OUTPUT_DIR / "example4_high_prot",
        plot=True,
        n_workers=20,
        tolerance=0.05,
        seed=789
    )
    
    print("\nKey Results:")
    print(f"  R = {analysis.R:.4f}")
    print(f"  Mean RMST = {analysis.mean_rmst:.2f}")
    print(f"  Required replicates (ε=0.1): {analysis.n_reps_required}")
    print("\nNote: detachment_time metric would fail here (all NaN)")
    print("      RMST successfully captures stability differences")
    
    return analysis


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RMST-BASED REPLICATE ESTIMATION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate replicate estimation using")
    print("Restricted Mean Survival Time (RMST) instead of detachment time.")
    print("\nAdvantages:")
    print("  ✓ Uses entire survival curve")
    print("  ✓ Always defined (no NaN issues)")
    print("  ✓ Captures full dynamics")
    print("  ✓ Better for stable nucleosomes")
    
    # Run examples
    try:
        # Example 1: No protamine
        # result1 = example_no_protamine()
        
        # # Example 2: Medium protamine
        result2 = example_medium_protamine()
        
        # # Example 3: Comparison
        # rmst_result, detach_result = example_comparison()
        
        # # Example 4: High protamine
        # result4 = example_high_protamine()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nCheck the following files:")
        print("  - rmst_analysis_*.txt  (text reports)")
        print("  - rmst_analysis_*.json (JSON data)")
        print("  - rmst_plot_*.png      (visualizations)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
