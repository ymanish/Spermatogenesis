#!/usr/bin/env python3
"""
Example: Running Markov Solver on Multiple Nucleosomes
=======================================================

This script demonstrates how to use the markov_execution module
to run Markov solver calculations on multiple nucleosomes in parallel.

Author: MY
Date: 2025-12-11
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

from pathlib import Path
from src.markov_execution import run_markov_solver, MarkovConfig
from src.markov_execution.storage import MarkovStorage
from src.config.path import HAMNUCRET_DATA_DIR, RESULTS_DIR


def example_1_basic():
    """Example 1: Basic usage without protamine."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Markov Solver (No Protamine)")
    print("=" * 70)
    
    # Setup storage
    storage_dir = RESULTS_DIR / "markov_execution_examples"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = MarkovStorage(base_dir=storage_dir, use_index=False)
    
    # Create configuration
    config = MarkovConfig(
        k_wrap=10.0,
        prot_p_conc=0.0,  # No protamine
        tau_max=5000.0,
        tau_steps=1000,
        n_workers=10,
        batch_size=1,
        save_survival=True,
        save_mfpt=True,
        save_states=False, 
        max_nucs=10
    )
    
    # Prepare parameters for storage
    prot_params = {
        'k_unbind': config.prot_k_unbind,
        'k_bind': config.prot_k_bind,
        'p_conc': config.prot_p_conc,
        'cooperativity': config.prot_cooperativity
    }
    
    params = {
        'k_wrap': config.k_wrap,
        'prot_params': prot_params,
        'binding_sites': config.binding_sites,
        'tau_max': config.tau_max,
        'tau_steps': config.tau_steps,
        'method': config.method,
        'sparse': config.sparse,
        'dimensionless': config.dimensionless,
    }
    
    # Get output paths
    file_id = "001"
    output_paths = storage.get_output_paths(params, file_id)
    

    run_markov_solver(
        file_path=HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv",
        tsv_outfile=output_paths['summary'],
        survival_outfile=output_paths['survivals'],
        config=config,
        max_nucs=config.max_nucs
    )
    
    print("\n✓ Results saved to:", output_paths['param_dir'])


def example_2_with_protamine():
    """Example 2: With protamine effects."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Markov Solver with Protamine")
    print("=" * 70)
    
    # Setup storage
    storage_dir = RESULTS_DIR / "markov_execution_examples"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = MarkovStorage(base_dir=storage_dir, use_index=False)
    
    config = MarkovConfig(
        k_wrap=1.0,
        prot_k_bind=1.0,
        prot_k_unbind=89.7,
        prot_p_conc=10.0,
        prot_cooperativity=0.0,
        t_max=1000.0,
        t_steps=500,
        method='expm',
        n_workers=10,
        batch_size=10,
        save_survival=True,
        save_mfpt=True
    )
    
    # Prepare parameters for storage
    prot_params = {
        'k_unbind': config.prot_k_unbind,
        'k_bind': config.prot_k_bind,
        'p_conc': config.prot_p_conc,
        'cooperativity': config.prot_cooperativity
    }
    
    params = {
        'k_wrap': config.k_wrap,
        'prot_params': prot_params,
        'binding_sites': config.binding_sites,
        't_max': config.t_max,
        't_steps': config.t_steps,
        'method': config.method,
        'sparse': config.sparse,
        'dimensionless': config.dimensionless,
        'kT': config.kT
    }
    
    file_id = "001"
    output_paths = storage.get_output_paths(params, file_id)
    
    config_text = storage.get_config_from_params(params)
    with open(output_paths['config'], 'w') as f:
        f.write(config_text)
    
    run_markov_solver(
        file_path=HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv",
        tsv_outfile=output_paths['summary'],
        survival_outfile=output_paths['results'],
        config=config,
        max_nucs=20
    )
    
    print("\n✓ Results saved to:", output_paths['param_dir'])
    return output_paths  # Return for example 5


def example_3_ode_solver():
    """Example 3: Using ODE solver for faster computation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: ODE Solver (Faster for Large Systems)")
    print("=" * 70)
    
    # Setup storage
    storage_dir = RESULTS_DIR / "markov_execution_examples"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = MarkovStorage(base_dir=storage_dir, use_index=False)
    
    config = MarkovConfig(
        k_wrap=1.0,
        prot_p_conc=50.0,
        prot_cooperativity=0.0,
        t_max=2000.0,
        t_steps=800,
        method='ode',  # ODE solver instead of matrix exponential
        sparse=True,   # Use sparse matrices
        n_workers=10,
        batch_size=5,
        save_survival=True,
        save_mfpt=True
    )
    
    prot_params = {
        'k_unbind': config.prot_k_unbind,
        'k_bind': config.prot_k_bind,
        'p_conc': config.prot_p_conc,
        'cooperativity': config.prot_cooperativity
    }
    
    params = {
        'k_wrap': config.k_wrap,
        'prot_params': prot_params,
        'binding_sites': config.binding_sites,
        't_max': config.t_max,
        't_steps': config.t_steps,
        'method': config.method,
        'sparse': config.sparse,
        'dimensionless': config.dimensionless,
        'kT': config.kT
    }
    
    file_id = "001"
    output_paths = storage.get_output_paths(params, file_id)
    
    config_text = storage.get_config_from_params(params)
    with open(output_paths['config'], 'w') as f:
        f.write(config_text)
    
    run_markov_solver(
        file_path=HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv",
        tsv_outfile=output_paths['summary'],
        survival_outfile=output_paths['results'],
        config=config,
        max_nucs=50
    )
    
    print("\n✓ Results saved to:", output_paths['param_dir'])


def example_4_parameter_scan():
    """Example 4: Scan protamine concentration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Parameter Scan (Protamine Concentration)")
    print("=" * 70)
    
    concentrations = [0.0, 1.0, 10.0, 100.0]
    
    for p_conc in concentrations:
        print(f"\n--- Processing concentration: {p_conc} μM ---")
        
        config = MarkovConfig(
            k_wrap=1.0,
            prot_k_bind=1.0,
            prot_k_unbind=89.7,
            prot_p_conc=p_conc,
            prot_cooperativity=0.0,
            t_max=1000.0,
            t_steps=500,
            n_workers=10,
            batch_size=10,
            save_survival=True,
            save_mfpt=True
        )
        
        run_markov_solver(
            file_path=HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv",
            output_dir=RESULTS_DIR / "markov_execution_examples" / f"scan_c{p_conc:.0f}",
            config=config,
            max_nucs=30
        )
    
    print("\n✓ Parameter scan complete!")


def example_5_analyze_results():
    """Example 5: Load and analyze results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Analyzing Markov Solver Results")
    print("=" * 70)
    
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Check if results exist
    results_dir = RESULTS_DIR / "markov_execution_examples" / "with_protamine"
    summary_file = results_dir / "markov_summary.tsv"
    detailed_file = results_dir / "markov_results.parquet"
    
    if not summary_file.exists():
        print("⚠ No results found. Run example_2_with_protamine() first.")
        return
    
    # Load summary
    print("\nLoading summary data...")
    summary = pl.read_csv(summary_file, separator='\t', comment_prefix='#')
    
    print(f"\nProcessed {len(summary)} nucleosomes")
    print(f"Mean MFPT: {summary['mfpt'].mean():.2f} ± {summary['mfpt'].std():.2f} τ")
    print(f"Mean half-life: {summary['half_life'].mean():.2f} ± {summary['half_life'].std():.2f} τ")
    print(f"Mean final survival: {summary['final_survival'].mean():.4f}")
    
    # Load detailed results
    if detailed_file.exists():
        print("\nLoading detailed survival data...")
        results = pl.read_parquet(detailed_file)
        
        # Plot first 5 nucleosomes
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx in range(min(6, len(results))):
            row = results.row(idx, named=True)
            t_grid = np.array(row['t_grid'])
            S = np.array(row['survival'])
            mfpt = row['mfpt']
            
            ax = axes[idx]
            ax.plot(t_grid, S, 'b-', lw=2)
            ax.axvline(mfpt, color='r', linestyle='--', alpha=0.7, label=f'MFPT={mfpt:.1f}')
            ax.set_xlabel('Time (τ)')
            ax.set_ylabel('S(t)')
            ax.set_title(f"Nuc {row['id']}-{row['subid']}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Hide last subplot if odd number
        if len(results) < 6:
            axes[-1].axis('off')
        
        plt.tight_layout()
        output_fig = results_dir / "survival_plots.png"
        plt.savefig(output_fig, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plots saved to: {output_fig}")
        plt.show()


if __name__ == "__main__":
    import sys
    
    # Set up temp directory
    tmp_dir = Path(__file__).parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    
    # Run examples
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic()
            raise SystemExit  # Prevent running all examples
        elif example_num == "2":
            example_2_with_protamine()
        elif example_num == "3":
            example_3_ode_solver()
        elif example_num == "4":
            example_4_parameter_scan()
        elif example_num == "5":
            example_5_analyze_results()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python example_markov_execution.py [1|2|3|4|5]")
    else:
        print("\nRunning all examples...")
        print("\nYou can also run individual examples:")
        print("  python example_markov_execution.py 1  # Basic usage")
        print("  python example_markov_execution.py 2  # With protamine")
        print("  python example_markov_execution.py 3  # ODE solver")
        print("  python example_markov_execution.py 4  # Parameter scan")
        print("  python example_markov_execution.py 5  # Analyze results")
        print()
        
        # Run a subset for demo
        example_2_with_protamine()
        example_5_analyze_results()
