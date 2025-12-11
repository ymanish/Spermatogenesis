"""
Test script for Hybrid Rejection + Inner Fast Tau-Leap Simulator
==================================================================

This script compares two simulation approaches:
1. Gillespie (exact SSA)
2. Hybrid Rejection with Inner Fast Tau-Leap

The hybrid rejection method is more physically accurate than standard tau-leap
because it properly handles protamine blocking of rewrapping events through
explicit rejection sampling.

Author: MY
Date: 2024-11-25
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.config import path
from src.config.custom_type import SimulationState
from src.core.gillespie_simulator import GillespieSimulator
from src.core.hybrid_rejection_simulator import HybridRejectionSimulator
from src.core.protamine import protamines
from src.core.build_nucleosomes import nucleosome_generator
from src.core.nucleosomes import Nucleosome, Nucleosomes


def run_comparison(nuc_id: str, subid: int, num_replicates: int = 5):
    """
    Compare Gillespie, Tau-Leap, and Hybrid Rejection simulators.
    
    Args:
        nuc_id: Nucleosome ID (e.g., "ENST00000000233.10")
        subid: Nucleosome sub-ID
        num_replicates: Number of simulation replicates
    """
    print("=" * 80)
    print("HYBRID REJECTION SIMULATOR COMPARISON")
    print("=" * 80)
    print(f"Nucleosome: {nuc_id}, subid={subid}")
    print(f"Replicates: {num_replicates}")
    print()
    
    # --- Simulation parameters ---
    p_conc = 100  # μM
    cooperativity = 0.0  # kT
    epsilon = 0.1  # Tau-leap accuracy
    
    # Time grid (dimensionless)
    tau_max = 1000.0
    num_points = 500
    tau_grid = np.linspace(0, tau_max, num_points)
    
    print(f"Parameters:")
    print(f"  Protamine concentration: {p_conc} μM")
    print(f"  Cooperativity: {cooperativity} kT")
    print(f"  Epsilon (tau-leap): {epsilon}")
    print(f"  Time grid: {num_points} points, tau_max={tau_max}")
    print()
    
    # --- Initialize shared components ---
    # Load nucleosome
    file_path = str(path.HAMNUCRET_DATA_DIR / "exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv")
    k_wrap = 1.0
    
    gen = nucleosome_generator(
        file_path=file_path,
        k_wrap=k_wrap,
        binding_sites=14,
        ids=[nuc_id],
        subids=[subid]
    )
    
    nucleosomes = list(gen)
    if len(nucleosomes) == 0:
        raise ValueError(f"No nucleosome found with id={nuc_id}, subid={subid}")
    
    nuc_template = nucleosomes[0]
    print(f"Loaded nucleosome: {nuc_template.id} (L={nuc_template.binding_sites} sites)")
    print()
    
    # Storage for results
    results = {
        'gillespie': {'trajectories': [], 'times': []},
        'hybrid': {'trajectories': [], 'times': []}
    }
    
    # --- Run simulations ---
    for rep in range(num_replicates):
        print(f"Replicate {rep + 1}/{num_replicates}...")
        
        # Reset nucleosome state for each replicate
        for method in ['gillespie', 'hybrid']:
            # Fresh nucleosome instance (copy from template)
            nuc_copy = Nucleosomes(
                k_wrap=k_wrap,
                kT=nuc_template.kT,
                nucleosomes=[
                    Nucleosome(
                        nuc_id=nuc_template.id,
                        subid=nuc_template.subid,
                        sequence=None,
                        G_mat=nuc_template.G_mat.copy(),
                        k_wrap=k_wrap,
                        kT=nuc_template.kT,
                        binding_sites=nuc_template.binding_sites
                    )
                ]
            )
            
            # Fresh protamine instance
            prot_copy = protamines(
                k_unbind=89.7,
                k_bind=1.0,
                p_conc=p_conc,
                cooperativity=cooperativity
            )
            
            # Run simulation
            if method == 'gillespie':
                sim = GillespieSimulator(
                    nuc_inst=nuc_copy,
                    prot_inst=prot_copy,
                    t_points=None,
                    max_steps=None,
                    tau_points=tau_grid,
                    inf_protamine=True,
                    seed=42 + rep
                )
            else:  # hybrid
                sim = HybridRejectionSimulator(
                    nuc_inst=nuc_copy,
                    prot_inst=prot_copy,
                    tau_points=tau_grid,
                    epsilon=epsilon,
                    inf_protamine=True,
                    seed=42 + rep
                )
            
            start_time = time.time()
            states = list(sim.run())
            elapsed = time.time() - start_time
            
            # Extract trajectory
            n_closed_traj = np.array([s.cs_total for s in states])
            n_bound_traj = np.array([s.bprot for s in states])
            
            results[method]['trajectories'].append((n_closed_traj, n_bound_traj))
            results[method]['times'].append(elapsed)
            
            print(f"  {method:12s}: {elapsed:.4f}s")
            
            # Print stats for hybrid
            if method == 'hybrid':
                sim.print_stats()
    
    print()
    
    # --- Compute statistics ---
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for method in ['gillespie', 'hybrid']:
        avg_time = np.mean(results[method]['times'])
        std_time = np.std(results[method]['times'])
        print(f"{method.capitalize():12s}: {avg_time:.4f} ± {std_time:.4f}s per replicate")
    
    gillespie_time = np.mean(results['gillespie']['times'])
    hybrid_time = np.mean(results['hybrid']['times'])
    
    print()
    print(f"Speedup (vs Gillespie):")
    print(f"  Hybrid:   {gillespie_time / hybrid_time:.2f}×")
    print()
    
    # --- Plot results ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Simulation Comparison: {nuc_id} (subid={subid})', fontsize=14)
    
    methods = ['gillespie', 'hybrid']
    titles = ['Gillespie (Exact)', 'Hybrid Rejection']
    colors = ['blue', 'green']
    
    for col, (method, title, color) in enumerate(zip(methods, titles, colors)):
        # Plot n_closed
        ax = axes[0, col]
        for n_closed_traj, _ in results[method]['trajectories']:
            ax.plot(tau_grid, n_closed_traj, alpha=0.5, color=color, linewidth=1)
        
        # Mean trajectory
        all_n_closed = np.array([traj[0] for traj in results[method]['trajectories']])
        mean_n_closed = all_n_closed.mean(axis=0)
        ax.plot(tau_grid, mean_n_closed, color='black', linewidth=2, label='Mean')
        
        ax.set_xlabel('Dimensionless time τ')
        ax.set_ylabel('Closed contacts')
        ax.set_title(f'{title}\n(Closed Contacts)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot n_bound
        ax = axes[1, col]
        for _, n_bound_traj in results[method]['trajectories']:
            ax.plot(tau_grid, n_bound_traj, alpha=0.5, color=color, linewidth=1)
        
        # Mean trajectory
        all_n_bound = np.array([traj[1] for traj in results[method]['trajectories']])
        mean_n_bound = all_n_bound.mean(axis=0)
        ax.plot(tau_grid, mean_n_bound, color='black', linewidth=2, label='Mean')
        
        ax.set_xlabel('Dimensionless time τ')
        ax.set_ylabel('Bound protamines')
        ax.set_title(f'{title}\n(Bound Protamines)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("output/hybrid_rejection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_dir / f"comparison_{nuc_id}_{subid}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_path}")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    # Test with a real nucleosome
    results = run_comparison(
        nuc_id="ENST00000000412.8",
        subid=2016,
        num_replicates=3
    )
    
    print("\n✓ Comparison complete!")
