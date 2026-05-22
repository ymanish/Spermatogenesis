#!/usr/bin/env python3
"""
Example: Tau-Leaping Simulator Usage
=====================================

Demonstrates how to use the hybrid τ-leaping simulator for nucleosome-protamine
dynamics. This simulator is efficient in the QSSA regime where protamines
equilibrate much faster than nucleosomes.

Usage:
    python3 examples/example_tau_leaping_simulator.py

Author: MY
Date: 2024-11-24
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import itertools
import matplotlib.pyplot as plt

from src.core.build_nucleosomes import nucleosome_generator
from src.core.protamine import protamines
from src.core.tau_leaping_simulator import TauLeapingSimulator, convert_to_gillespie_state


def example_single_trajectory():
    """
    Example 1: Run single τ-leaping trajectory.
    """
    print("=" * 80)
    print("EXAMPLE 1: Single Tau-Leaping Trajectory")
    print("=" * 80)
    
    # Load a nucleosome
    file_path = "hamnucret_data/unboundprom/breath_energy/001.tsv"
    print(f"\nLoading nucleosome from {file_path}...")
    
    gen = nucleosome_generator(
        file_path=file_path,
        k_wrap=22.0,
        binding_sites=14
    )
    
    nuc = next(itertools.islice(gen, 1))
    print(f"✓ Loaded nucleosome: ID={nuc.id}, subid={nuc.subid}")
    
    # Create protamine instance
    prot = protamines(
        k_unbind=0.23,
        k_bind=2113,
        p_conc=0.1,  # μM
        cooperativity=4.5
    )
    print(f"✓ Created protamine instance")
    
    # Create τ-leaping simulator
    tau_max = 1000.0
    epsilon = 0.1  # Accuracy parameter
    
    sim = TauLeapingSimulator(
        nuc=nuc,
        prot=prot,
        tau_max=tau_max,
        epsilon=epsilon,
        beta=1.0,
        seed=42,
        record_interval=10  # Record every 10 steps
    )
    
    print(f"\n✓ Created τ-leaping simulator")
    print(f"  tau_max: {tau_max}")
    print(f"  epsilon: {epsilon}")
    print(f"  k_on: {sim.k_on:.2f}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    states = []
    for state in sim.run():
        states.append(state)
    
    print(f"✓ Simulation complete")
    print(f"  Total steps: {sim.step_count}")
    print(f"  States recorded: {len(states)}")
    print(f"  Final tau: {states[-1].tau:.2f}")
    print(f"  Detached: {states[-1].detached}")
    
    # Extract trajectory
    taus = np.array([s.tau for s in states])
    n_closed = np.array([s.n_closed for s in states])
    n_bound = np.array([np.sum(s.s) for s in states])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(taus, n_closed, 'b-', lw=2)
    ax1.set_ylabel('Wrapped Sites')
    ax1.set_title(f'Tau-Leaping Simulation (ε={epsilon})')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(taus, n_bound, 'r-', lw=2)
    ax2.set_xlabel('Dimensionless Time (τ)')
    ax2.set_ylabel('Bound Protamines')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("output/tau_leaping")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "single_trajectory.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_dir}/single_trajectory.png")
    
    plt.show()
    
    return states


def example_compare_epsilon_values():
    """
    Example 2: Compare simulations with different ε values.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare Different Epsilon Values")
    print("=" * 80)
    
    # Load nucleosome
    file_path = "hamnucret_data/unboundprom/breath_energy/001.tsv"
    gen = nucleosome_generator(file_path=file_path, k_wrap=22.0, binding_sites=14)
    nuc = next(itertools.islice(gen, 1))
    
    # Protamine
    prot = protamines(k_unbind=0.23, k_bind=2113, p_conc=0.1, cooperativity=4.5)
    
    # Test different epsilon values
    epsilons = [0.01, 0.05, 0.1, 0.2]
    tau_max = 500.0
    
    results = {}
    
    for eps in epsilons:
        print(f"\nRunning with ε={eps}...")
        
        sim = TauLeapingSimulator(
            nuc=nuc,
            prot=prot,
            tau_max=tau_max,
            epsilon=eps,
            seed=42,
            record_interval=5
        )
        
        states = list(sim.run())
        results[eps] = {
            'states': states,
            'steps': sim.step_count
        }
        
        print(f"  Steps: {sim.step_count}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for (eps, data), color in zip(results.items(), colors):
        states = data['states']
        taus = [s.tau for s in states]
        n_closed = [s.n_closed for s in states]
        
        ax1.plot(taus, n_closed, color=color, label=f'ε={eps}', alpha=0.7, lw=2)
    
    ax1.set_xlabel('Dimensionless Time (τ)')
    ax1.set_ylabel('Wrapped Sites')
    ax1.set_title('Effect of ε on Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot steps vs epsilon
    eps_vals = list(results.keys())
    step_counts = [results[e]['steps'] for e in eps_vals]
    
    ax2.plot(eps_vals, step_counts, 'o-', color='purple', lw=2, markersize=8)
    ax2.set_xlabel('ε (accuracy parameter)')
    ax2.set_ylabel('Total Steps')
    ax2.set_title('Computational Cost vs Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    output_dir = Path("output/tau_leaping")
    plt.savefig(output_dir / "epsilon_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_dir}/epsilon_comparison.png")
    
    plt.show()


def example_multiple_replicates():
    """
    Example 3: Run multiple replicates and compute statistics.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multiple Replicates")
    print("=" * 80)
    
    # Load nucleosome
    file_path = "hamnucret_data/unboundprom/breath_energy/001.tsv"
    gen = nucleosome_generator(file_path=file_path, k_wrap=22.0, binding_sites=14)
    nuc = next(itertools.islice(gen, 1))
    
    # Protamine
    prot = protamines(k_unbind=0.23, k_bind=2113, p_conc=0.1, cooperativity=4.5)
    
    # Run replicates
    n_replicates = 10
    tau_max = 500.0
    epsilon = 0.1
    
    print(f"\nRunning {n_replicates} replicates...")
    
    all_trajectories = []
    
    for rep in range(n_replicates):
        sim = TauLeapingSimulator(
            nuc=nuc,
            prot=prot,
            tau_max=tau_max,
            epsilon=epsilon,
            seed=42 + rep,  # Different seed per replicate
            record_interval=10
        )
        
        states = list(sim.run())
        
        # Extract uniform time grid by interpolation
        taus = np.array([s.tau for s in states])
        n_closed = np.array([s.n_closed for s in states])
        
        # Interpolate to common grid
        tau_grid = np.linspace(0, tau_max, 100)
        n_closed_interp = np.interp(tau_grid, taus, n_closed)
        
        all_trajectories.append(n_closed_interp)
        
        if (rep + 1) % 5 == 0:
            print(f"  Completed {rep + 1}/{n_replicates}")
    
    # Compute statistics
    all_traj_array = np.array(all_trajectories)
    mean_traj = np.mean(all_traj_array, axis=0)
    std_traj = np.std(all_traj_array, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual trajectories
    for traj in all_trajectories:
        ax.plot(tau_grid, traj, color='gray', alpha=0.3, lw=1)
    
    # Plot mean ± std
    ax.plot(tau_grid, mean_traj, 'b-', lw=3, label='Mean')
    ax.fill_between(
        tau_grid,
        mean_traj - std_traj,
        mean_traj + std_traj,
        color='blue',
        alpha=0.2,
        label='± 1 std'
    )
    
    ax.set_xlabel('Dimensionless Time (τ)')
    ax.set_ylabel('Wrapped Sites')
    ax.set_title(f'Tau-Leaping Simulation ({n_replicates} replicates, ε={epsilon})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("output/tau_leaping")
    plt.savefig(output_dir / "multiple_replicates.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_dir}/multiple_replicates.png")
    
    plt.show()
    
    print(f"\n✓ All replicates complete")
    print(f"  Mean final wrapped: {mean_traj[-1]:.2f} ± {std_traj[-1]:.2f}")


if __name__ == "__main__":
    # Run examples
    
    # Example 1: Single trajectory
    states = example_single_trajectory()
    
    # Example 2: Compare epsilon values
    # Uncomment to run:
    # example_compare_epsilon_values()
    
    # Example 3: Multiple replicates
    # Uncomment to run:
    # example_multiple_replicates()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Uncomment other examples to run them")
    print("  2. Adjust parameters (epsilon, tau_max, etc.)")
    print("  3. Compare with full Gillespie simulation")
    print("  4. Check QSSA validity using validate_qssa_for_system()")
