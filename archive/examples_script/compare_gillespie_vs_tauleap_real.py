#!/usr/bin/env python3
"""
Compare Gillespie vs Tau-Leaping Simulators with Real Nucleosome Data
======================================================================

This script compares the Gillespie and tau-leaping simulators using real
nucleosome data from the nucleosome generator (not synthetic G-matrices).

Uses:
- Real G-matrices from TSV files (like example_sim.py)
- Same nucleosome generator as production code
- Multiple replicates for statistical comparison
- Visualization of trajectories

Author: MY
Date: 2024-11-25
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import copy

# Import simulation components
from src.core.build_nucleosomes import nucleosome_generator
from src.core.nucleosomes import Nucleosomes
from src.core.protamine import protamines
from src.core.gillespie_simulator import GillespieSimulator
from src.core.tau_leaping_simulator import TauLeapingSimulator

from src.config.path import HAMNUCRET_DATA_DIR, RESULTS_DIR
from src.config.var import seed_for


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default input file (bound promoter regions)
DEFAULT_FILE = Path(
    HAMNUCRET_DATA_DIR / "exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
)

# Output directory
OUTPUT_DIR = RESULTS_DIR / "local_tests" / "gillespie_vs_tauleap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Nucleosome parameters
K_WRAP = 1.0  # 1/second
BINDING_SITES = 14

# Protamine parameters
PROT_PARAMS = {
    'k_bind': 1.0,      # μM^-1 s^-1
    'k_unbind': 89.7,     # s^-1
    'p_conc': 100,       # μM (concentration)
    'cooperativity': 0 # kT
}

# Simulation parameters
TAU_MAX = 10000.0  # Dimensionless time
N_POINTS = 500   # Number of sampling points
REPLICATES = 5   # Number of replicate trajectories
INF_PROTAMINE = True

# Tau-leaping parameter
EPSILON = 0.1  # Accuracy parameter (smaller = more accurate, slower)

# IDs to simulate (from TSV file)
IDS_TO_SIMULATE = ['ENST00000000233.10']  # Transcript ID
SUBIDS_TO_SIMULATE = [2086, 2087]  # First 2 nucleosomes from this transcript


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_nucleosomes_instance(nuc, k_wrap, binding_sites):
    """Create Nucleosomes instance (deep copy for independence)."""
    nuc_copy = copy.deepcopy(nuc)
    return Nucleosomes(k_wrap=k_wrap,
                      nucleosomes=[nuc_copy],
                      binding_sites=binding_sites)


def create_protamines_instance(prot_params):
    """Create protamines instance."""
    return protamines(**prot_params)


def simulate_gillespie(nuc, k_wrap, binding_sites, tau_points, prot_params, 
                       inf_protamine, replicates):
    """
    Run Gillespie simulation with multiple replicates.
    
    Returns:
        List of (times, cs_total, bprot) tuples for each replicate
    """
    print(f"\n  Running Gillespie simulation...")
    print(f"    Nucleosome: {nuc.id}, subid: {nuc.subid}")
    print(f"    Replicates: {replicates}")
    
    all_trajectories = []
    total_time = 0.0
    
    for r in range(replicates):
        # Create fresh instances
        nucs = create_nucleosomes_instance(nuc, k_wrap, binding_sites)
        prots = create_protamines_instance(prot_params)
        seed = seed_for(nuc, r)
        
        sim = GillespieSimulator(
            nuc_inst=nucs,
            prot_inst=prots,
            t_points=None,
            max_steps=None,
            tau_points=tau_points,
            inf_protamine=inf_protamine,
            seed=seed
        )
        
        times = []
        cs_list = []
        bprot_list = []
        
        start = time.time()
        for st in sim.run():
            times.append(st.tau)
            cs_list.append(st.cs_total)
            bprot_list.append(st.bprot)
        elapsed = time.time() - start
        total_time += elapsed
        
        all_trajectories.append((
            np.array(times),
            np.array(cs_list),
            np.array(bprot_list)
        ))
    
    avg_time = total_time / replicates
    print(f"    ✓ Completed in {total_time:.2f}s (avg {avg_time:.3f}s per replicate)")
    
    return all_trajectories, avg_time


def simulate_tauleap(nuc, k_wrap, binding_sites, tau_points, prot_params,
                     inf_protamine, replicates, epsilon):
    """
    Run tau-leaping simulation with multiple replicates.
    
    Returns:
        List of (times, cs_total, bprot) tuples for each replicate
    """
    print(f"\n  Running Tau-Leaping simulation (ε={epsilon})...")
    print(f"    Nucleosome: {nuc.id}, subid: {nuc.subid}")
    print(f"    Replicates: {replicates}")
    
    all_trajectories = []
    total_time = 0.0
    
    for r in range(replicates):
        # Create fresh instances
        nucs = create_nucleosomes_instance(nuc, k_wrap, binding_sites)
        prots = create_protamines_instance(prot_params)
        seed = seed_for(nuc, r)
        
        sim = TauLeapingSimulator(
            nuc_inst=nucs,
            prot_inst=prots,
            tau_points=tau_points,
            epsilon=epsilon,
            inf_protamine=inf_protamine,
            seed=seed
        )
        
        times = []
        cs_list = []
        bprot_list = []
        
        start = time.time()
        for st in sim.run():
            times.append(st.tau)
            cs_list.append(st.cs_total)
            bprot_list.append(st.bprot)
        elapsed = time.time() - start
        total_time += elapsed
        
        all_trajectories.append((
            np.array(times),
            np.array(cs_list),
            np.array(bprot_list)
        ))
    
    avg_time = total_time / replicates
    print(f"    ✓ Completed in {total_time:.2f}s (avg {avg_time:.3f}s per replicate)")
    
    return all_trajectories, avg_time


def plot_comparison(gill_trajectories, tauleap_trajectories, nuc, output_dir):
    """
    Plot comparison of Gillespie vs Tau-Leaping trajectories.
    
    Shows:
    - Individual trajectories (light)
    - Average trajectory (bold)
    - Wrapped sites and bound protamines
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Extract data
    gill_times_list = [traj[0] for traj in gill_trajectories]
    gill_cs_list = [traj[1] for traj in gill_trajectories]
    gill_bprot_list = [traj[2] for traj in gill_trajectories]
    
    tauleap_times_list = [traj[0] for traj in tauleap_trajectories]
    tauleap_cs_list = [traj[1] for traj in tauleap_trajectories]
    tauleap_bprot_list = [traj[2] for traj in tauleap_trajectories]
    
    # Plot wrapped sites
    ax = axes[0]
    
    # Gillespie individual trajectories
    for times, cs in zip(gill_times_list, gill_cs_list):
        ax.plot(times, cs, 'b-', alpha=0.2, linewidth=0.5)
    
    # Gillespie average
    gill_cs_avg = np.mean(np.vstack(gill_cs_list), axis=0)
    ax.plot(gill_times_list[0], gill_cs_avg, 'b-', linewidth=2, label='Gillespie (avg)')
    
    # Tau-leaping individual trajectories
    for times, cs in zip(tauleap_times_list, tauleap_cs_list):
        ax.plot(times, cs, 'r-', alpha=0.2, linewidth=0.5)
    
    # Tau-leaping average
    tauleap_cs_avg = np.mean(np.vstack(tauleap_cs_list), axis=0)
    ax.plot(tauleap_times_list[0], tauleap_cs_avg, 'r-', linewidth=2, label=f'Tau-Leap (ε={EPSILON}, avg)')
    
    ax.set_ylabel('Wrapped Sites', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Gillespie vs Tau-Leaping: {nuc.id} (subid={nuc.subid})', fontweight='bold')
    
    # Plot bound protamines
    ax = axes[1]
    
    # Gillespie individual
    for times, bprot in zip(gill_times_list, gill_bprot_list):
        ax.plot(times, bprot, 'b-', alpha=0.2, linewidth=0.5)
    
    # Gillespie average
    gill_bprot_avg = np.mean(np.vstack(gill_bprot_list), axis=0)
    ax.plot(gill_times_list[0], gill_bprot_avg, 'b-', linewidth=2, label='Gillespie (avg)')
    
    # Tau-leaping individual
    for times, bprot in zip(tauleap_times_list, tauleap_bprot_list):
        ax.plot(times, bprot, 'r-', alpha=0.2, linewidth=0.5)
    
    # Tau-leaping average
    tauleap_bprot_avg = np.mean(np.vstack(tauleap_bprot_list), axis=0)
    ax.plot(tauleap_times_list[0], tauleap_bprot_avg, 'r-', linewidth=2, label=f'Tau-Leap (ε={EPSILON}, avg)')
    
    ax.set_xlabel('Dimensionless Time (τ)', fontweight='bold')
    ax.set_ylabel('Bound Protamines', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"comparison_{nuc.id}_subid{nuc.subid}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\n  ✓ Figure saved: {fig_path}")
    
    plt.show()


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(file_path, ids_list, subids_list, output_dir):
    """
    Run comparison for specified nucleosomes.
    
    Args:
        file_path: Path to TSV file with G-matrices
        ids_list: List of chromosome IDs to simulate
        subids_list: List of subids to simulate
        output_dir: Directory for output files
    """
    print("\n" + "="*70)
    print("GILLESPIE VS TAU-LEAPING COMPARISON")
    print("="*70)
    print(f"Input file: {file_path}")
    print(f"IDs: {ids_list}")
    print(f"Subids: {subids_list}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Create tau points
    tau_points = np.linspace(0, TAU_MAX, N_POINTS)
    
    # Generate nucleosomes
    print("\n[1/3] Loading nucleosomes...")
    gen = nucleosome_generator(
        file_path=file_path,
        k_wrap=K_WRAP,
        binding_sites=BINDING_SITES,
        ids=ids_list,
        subids=subids_list
    )
    
    nucleosomes = list(gen)
    print(f"  ✓ Loaded {len(nucleosomes)} nucleosome(s)")
    
    # Run comparisons for each nucleosome
    for idx, nuc in enumerate(nucleosomes):
        print(f"\n[{idx+2}/{len(nucleosomes)+2}] Simulating nucleosome {idx+1}/{len(nucleosomes)}")
        print(f"  ID: {nuc.id}, subid: {nuc.subid}")
        
        # Run Gillespie
        gill_traj, gill_time = simulate_gillespie(
            nuc, K_WRAP, BINDING_SITES, tau_points,
            PROT_PARAMS, INF_PROTAMINE, REPLICATES
        )
        
        # Run Tau-Leaping
        tauleap_traj, tauleap_time = simulate_tauleap(
            nuc, K_WRAP, BINDING_SITES, tau_points,
            PROT_PARAMS, INF_PROTAMINE, REPLICATES, EPSILON
        )
        
        # Calculate speedup
        speedup = gill_time / tauleap_time if tauleap_time > 0 else 0
        print(f"\n  Performance:")
        print(f"    Gillespie: {gill_time:.3f}s per replicate")
        print(f"    Tau-Leap:  {tauleap_time:.3f}s per replicate")
        print(f"    Speedup:   {speedup:.2f}×")
        
        # Plot comparison
        plot_comparison(gill_traj, tauleap_traj, nuc, output_dir)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Check if input file exists
    if not DEFAULT_FILE.exists():
        print(f"ERROR: Input file not found: {DEFAULT_FILE}")
        print("Please check the path or use a different file.")
        exit(1)
    
    # Run comparison
    run_comparison(
        file_path=DEFAULT_FILE,
        ids_list=IDS_TO_SIMULATE,
        subids_list=SUBIDS_TO_SIMULATE,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✓ All done!")
    print(f"  Results saved to: {OUTPUT_DIR}")
