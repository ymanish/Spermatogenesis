
import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *  # Triggers env_settings import

import itertools
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

# Import your classes and functions
from src.core.build_nucleosomes import nucleosome_generator
from src.core.nucleosomes import Nucleosomes
from src.core.protamine import protamines
from src.core.gillespie_simulator import GillespieSimulator

def simulate_single(nuc, k_wrap, binding_sites, t_points,
                    k_unbind, k_bind, p_conc, cooperativity, inf_protamine, tau_min):
    """
    Given a single nucleosome object 'nuc', set up the simulation components,
    run the GillespieSimulator and return the time vector and the cs_total array.
    """
    print(f"Simulating nucleosome: {nuc.id}, subid: {nuc.subid}")
    # Build the Nucleosomes object wrapping the single nucleosome:
    nucs = Nucleosomes(k_wrap=k_wrap, kT=1.0, nucleosomes=[nuc], binding_sites=binding_sites)
    # Create the protamine instance using provided parameters:
    prot_inst = protamines(k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity)
    # Initialize simulator with a fixed seed (or parameterize as needed)
    sim = GillespieSimulator(nuc_inst=nucs,
                             prot_inst=prot_inst,
                             t_points=t_points,
                             max_steps=None,
                             inf_protamine=inf_protamine,
                             seed=4, 
                             tau_min=tau_min)
    times = []
    cs_list = []
    bprot_list = []
    t_block = []
    for st in sim.run():
        times.append(st.time)
        cs_list.append(st.cs_total)
        bprot_list.append(st.bprot)
        t_block.append(st.t_blocked)
    return np.array(times), np.array(cs_list), np.array(bprot_list), np.array(t_block)




def _compute_tau_min(k_wrap: float, ends: int = 2, gamma: float = 3.0) -> float:
    # If either end can initiate rewrap from fully unwrapped, ends=2
    import math
    w0 = ends * float(k_wrap)
    t099 = np.log(100.0) / w0
    return gamma * t099

def run_simulations_for_condition(file_path, ids, subids, k_wrap, binding_sites, t_points,
                                  k_unbind, k_bind, p_conc, cooperativity, inf_protamine, max_nucs=100):
    """
    Run Gillespie simulations in parallel for nucleosomes generated from a given file,
    ids and subids. Returns a list of (times, cs_array) tuples.
    """
    # Create the nucleosome generator and limit number of items
    gen = nucleosome_generator(file_path=file_path, k_wrap=k_wrap, binding_sites=binding_sites,
                               ids=ids, subids=subids)
    start_idx = 2000
    gen = itertools.islice(gen, start_idx, start_idx + max_nucs)


    tau_min = _compute_tau_min(k_wrap=k_wrap, ends=2, gamma=5.0)
    print(f"Computed tau_min: {tau_min:.4f} for k_wrap: {k_wrap}")
    # import sys
    # sys.exit(0)

    
    results = []
    # Use a process pool to run individual simulation tasks in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(simulate_single, nuc, k_wrap, binding_sites, t_points,
                                   k_unbind, k_bind, p_conc, cooperativity, inf_protamine, tau_min)
                   for nuc in gen]
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"Simulation error: {e}")
    return results

# --- Main script parameters ---
# Bound vs unbound file paths and id lists.
# file_path_bound = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/boundprom/breath_energy/001.tsv"
# file_path_unbound = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/unboundprom/breath_energy/001.tsv"

# ids_bound = ["ENST00000210633.4"]      # Bound condition ids
# ids_unbound = ["ENST00000695399.1"]      # Unbound condition ids

file_path_bound = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/exactpoint_boundpromoter_regions_breath/breath_energy/001.tsv"
file_path_unbound = "/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv"

ids_bound = ["ENST00000046087.7"]      # Bound condition ids
ids_unbound = ["ENST00000294179.8"]      # Unbound condition ids
ids_bound = None
ids_unbound = None

# # Simulation parameters
# k_wrap = 0.1
# binding_sites = 14
# k_unbind = 0.1
# k_bind = 1.0
# p_conc = 100.0
# cooperativity = 10.0
# inf_protamine = True

# t_max = 10000.0
# t_steps = 10000
# t_points = np.linspace(0, t_max, t_steps)

# # Choose a subid range (modify as needed)
# subids_range = np.arange(1965, 2160).tolist()
# subids_range = None
# max_nucs = 100  # maximum nucleosome simulations to run per condition

# # --- Run simulations in parallel for both conditions ---
# results_bound = run_simulations_for_condition(file_path_bound, ids_bound, subids=subids_range,
#                                               k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
#                                               k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
#                                               inf_protamine=inf_protamine, max_nucs=max_nucs)

# results_unbound = run_simulations_for_condition(file_path_unbound, ids_unbound, subids=subids_range,
#                                                 k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
#                                                 k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
#                                                 inf_protamine=inf_protamine, max_nucs=max_nucs)

# # for i, (times, cs_array) in enumerate(results_unbound):
# #     print(f"Simulation {i+1}: Time points: {len(times)}, CS array shape: {cs_array.shape}")
# #     plt.plot(times, cs_array, label=f"Simulation {i+1}")
# #     plt.show()

#     # Optionally, save or process the results further
#     # For example, you could save to a file or plot the results here

# # --- Combine results ---
# if results_bound and results_unbound:
#     # Assuming each simulation yields identical time grid (use first simulation's times)
#     # avg_times_bound = results_bound[0][0]
#     avg_cs_bound = np.mean(np.vstack([res[1] for res in results_bound]), axis=0)
#     avg_bprot_bound = np.mean(np.vstack([res[2] for res in results_bound]), axis=0)

#     # avg_times_unbound = results_unbound[0][0]
#     avg_cs_unbound = np.mean(np.vstack([res[1] for res in results_unbound]), axis=0)
#     avg_bprot_unbound = np.mean(np.vstack([res[2] for res in results_unbound]), axis=0)

#     plt.figure(figsize=(8, 4))
#     plt.plot(t_points, avg_cs_bound, color="blue", lw=2, label="Bound")
#     plt.plot(t_points, avg_cs_unbound, color="black", lw=2, label="Unbound")
#     plt.xlabel("Time")
#     plt.ylabel("Total wrapped (n_closed)")
#     plt.xlim(0, t_max)
#     plt.ylim(0, binding_sites)
#     plot_title = f"Avg Total Wrapped Over Time\n(k_wrap={k_wrap}, k_unbind={k_unbind}, k_bind={k_bind}, p_conc={p_conc}, coop={cooperativity})"
#     plt.title(plot_title)
#     plt.legend()
#     plt.show()

#     # plt.figure(figsize=(8, 4))
#     # plt.plot(t_points, avg_bprot_bound, color="blue", lw=2, label="Bound")
#     # plt.plot(t_points, avg_bprot_unbound, color="black", lw=2, label="Unbound")
#     # plt.xlabel("Time")
#     # plt.ylabel("Total protamine (n_protamine)")
#     # plt.xlim(0, t_max)
#     # plt.ylim(0, binding_sites)
#     # plot_title = f"Avg Total Protamine Over Time\n(k_wrap={k_wrap}, k_unbind={k_unbind}, k_bind={k_bind}, p_conc={p_conc}, coop={cooperativity})"
#     # plt.title(plot_title)
#     # plt.legend()
#     # plt.show()
# else:
#     print("No simulation runs were completed for one or both conditions.")





import matplotlib as mpl
from pathlib import Path
thesis_dir = Path("/home/pol_schiessel/maya620d/pol/Projects/Thesis")
# --- publication‐style rcParams ---
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,          # base font size (used if specific sizes not given)
    'axes.labelsize': 14,     # x/y label font size
    'axes.labelweight': 'bold',   # make x/y label text bold
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',   # (optional) bold titles too
    'xtick.labelsize': 12,    # tick number font size (x)
    'ytick.labelsize': 12,    # tick number font size (y)
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'svg.fonttype': 'none',
    # --- added to pull labels closer ---
    'axes.labelpad': 1,        # distance axis spine <-> axis label (default ~4)
    'axes.titlepad': 2,        # distance title <-> top of axes (lower if you want closer)
    'xtick.major.pad': 2,      # distance tick labels <-> axis
    'ytick.major.pad': 2,
    'xtick.minor.pad': 2,
    'ytick.minor.pad': 2,
})


#########################################################
#########################################################
###################### NO PROTAMINE ######################
#########################################################
#########################################################
#########################################################
# def no_protamine_analysis(save_path=None):
#     import seaborn as sns
#     import pandas as pd
#     from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator


#     # Simulation parameters
#     k_wrap = 1.0
#     binding_sites = 14
#     k_unbind = 0.1
#     k_bind = 1.0
#     inf_protamine = True
#     t_max = 10000.0
#     t_steps = 10000
#     t_points = np.linspace(0, t_max, t_steps)
#     subids_range = None
#     max_nucs = 10
#     p_conc = 0.0
#     cooperativity = 0.0


#     results_bound = run_simulations_for_condition(
#             file_path_bound, ids_bound, subids=subids_range,
#             k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
#             k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
#             inf_protamine=inf_protamine, max_nucs=max_nucs
#         )

#     results_unbound = run_simulations_for_condition(
#             file_path_unbound, ids_unbound, subids=subids_range,
#             k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
#             k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
#             inf_protamine=inf_protamine, max_nucs=max_nucs
#         )



#     # --- Ensure we have results and build vr dict ---
#     if not (results_bound and results_unbound):
#         print("No simulation runs were completed for one or both conditions.")
#         return

#     avg_cs_bound = np.mean(np.vstack([res[1] for res in results_bound]), axis=0)
#     avg_cs_unbound = np.mean(np.vstack([res[1] for res in results_unbound]), axis=0)

#     vr = {
#         'times': t_points,
#         'avg_cs_bound': avg_cs_bound,
#         'avg_cs_unbound': avg_cs_unbound,
#         'p_conc': p_conc,
#         'coop': cooperativity
#     }

#     # --- compute std (or fallback to zeros) to show uncertainty band ---
#     try:
#         std_cs_bound = np.std(np.vstack([res[1] for res in results_bound]), axis=0)
#         std_cs_unbound = np.std(np.vstack([res[1] for res in results_unbound]), axis=0)
#     except Exception:
#         std_cs_bound = np.zeros_like(vr['avg_cs_bound'])
#         std_cs_unbound = np.zeros_like(vr['avg_cs_unbound'])

#     # --- Plot using a single Axes object (improved styling) ---
#     fig, ax = plt.subplots(figsize=(5, 3))

#     # shaded uncertainty (±1 std)
#     ax.fill_between(vr['times'],
#                     np.clip(vr['avg_cs_bound'] - std_cs_bound, 0, binding_sites),
#                     np.clip(vr['avg_cs_bound'] + std_cs_bound, 0, binding_sites),
#                     color='tab:green', alpha=0.12, linewidth=0)

#     ax.fill_between(vr['times'],
#                     np.clip(vr['avg_cs_unbound'] - std_cs_unbound, 0, binding_sites),
#                     np.clip(vr['avg_cs_unbound'] + std_cs_unbound, 0, binding_sites),
#                     color='tab:orange', alpha=0.10, linewidth=0)

#     # stronger lines for averages
#     ax.plot(vr['times'], vr['avg_cs_bound'], label='RET', color='tab:green', lw=2.2)
#     ax.plot(vr['times'], vr['avg_cs_unbound'], label='EVI', color='tab:orange', ls='--', lw=2.0)

#     # x-axis: plain seconds + minor ticks
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
#     ax.xaxis.set_minor_locator(AutoMinorLocator(4))
#     ax.tick_params(axis='x', which='major', length=6)
#     ax.tick_params(axis='x', which='minor', length=3, color='0.3')

#     # y-axis: strict integers 0..binding_sites and small padding
#     # ax.set_ylim(0, binding_sites + 0.2)
#     # ax.set_yticks(np.arange(0, binding_sites + 1))
#     # ax.set_ylabel("Wrapped Sites")
#     from matplotlib.ticker import MultipleLocator

#     ax.set_ylim(0, binding_sites + 0.2)
#     # major ticks every 2 (labelled), minor ticks every 1 (no label)
#     ax.yaxis.set_major_locator(MultipleLocator(2))
#     ax.yaxis.set_minor_locator(MultipleLocator(1))
#     ax.tick_params(axis='y', which='major', length=6)
#     ax.tick_params(axis='y', which='minor', length=3, labelleft=False, color='0.4')
#     ax.set_ylabel("Wrapped Sites")

#     ax.set_xlabel("Time (s)")
#     ax.grid(which='major', alpha=0.35)
#     ax.grid(which='minor', alpha=0.12)
#     ax.legend(framealpha=0.85,loc='lower right')

#     plt.tight_layout()

#     # save
#     if save_path is not None:
#         for ext in ["png", "pdf", "svg"]:
#             plt.savefig(save_path / f"no_protamine_traj.{ext}",
#                         dpi=300,
#                         bbox_inches='tight',
#                         transparent=True)
        




# no_protamne_analysis(save_path=thesis_dir / "Chapter_HistProt" / "SSA")
##### for this analysis the start_idx in run_simulations_for_condition should be 0 to regenerate the figures
###and for later 2000


# import sys
# sys.exit(0)




# def with_protamine_analysis(save_path=None):

#     import seaborn as sns
#     import pandas as pd
#     from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator


#     # Simulation parameters
#     k_wrap = 1.0
#     binding_sites = 14
#     k_unbind = 0.1
#     k_bind = 1.0
#     inf_protamine = True
#     t_max = 10000.0
#     t_steps = 10000
#     t_points = np.linspace(0, t_max, t_steps)
#     subids_range = None
#     max_nucs = 10
#     p_conc = 0.2
#     cooperativity = 2.0


#     results_bound = run_simulations_for_condition(
#             file_path_bound, ids_bound, subids=subids_range,
#             k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
#             k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
#             inf_protamine=inf_protamine, max_nucs=max_nucs
#         )

#     results_unbound = run_simulations_for_condition(
#             file_path_unbound, ids_unbound, subids=subids_range,
#             k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
#             k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
#             inf_protamine=inf_protamine, max_nucs=max_nucs
#         )



#     # --- Ensure we have results and build vr dict ---
#     if not (results_bound and results_unbound):
#         print("No simulation runs were completed for one or both conditions.")
#         return

#     avg_cs_bound = np.mean(np.vstack([res[1] for res in results_bound]), axis=0)
#     avg_cs_unbound = np.mean(np.vstack([res[1] for res in results_unbound]), axis=0)

#     vr = {
#         'times': t_points,
#         'avg_cs_bound': avg_cs_bound,
#         'avg_cs_unbound': avg_cs_unbound,
#         'p_conc': p_conc,
#         'coop': cooperativity
#     }

#     # --- compute std (or fallback to zeros) to show uncertainty band ---
#     try:
#         std_cs_bound = np.std(np.vstack([res[1] for res in results_bound]), axis=0)
#         std_cs_unbound = np.std(np.vstack([res[1] for res in results_unbound]), axis=0)
#     except Exception:
#         std_cs_bound = np.zeros_like(vr['avg_cs_bound'])
#         std_cs_unbound = np.zeros_like(vr['avg_cs_unbound'])

#     # --- Plot using a single Axes object (improved styling) ---
#     fig, ax = plt.subplots(figsize=(5, 3))

#     # shaded uncertainty (±1 std)
#     ax.fill_between(vr['times'],
#                     np.clip(vr['avg_cs_bound'] - std_cs_bound, 0, binding_sites),
#                     np.clip(vr['avg_cs_bound'] + std_cs_bound, 0, binding_sites),
#                     color='tab:green', alpha=0.12, linewidth=0)

#     ax.fill_between(vr['times'],
#                     np.clip(vr['avg_cs_unbound'] - std_cs_unbound, 0, binding_sites),
#                     np.clip(vr['avg_cs_unbound'] + std_cs_unbound, 0, binding_sites),
#                     color='tab:orange', alpha=0.10, linewidth=0)

#     # stronger lines for averages
#     ax.plot(vr['times'], vr['avg_cs_bound'], label='RET', color='tab:green', lw=2.2)
#     ax.plot(vr['times'], vr['avg_cs_unbound'], label='EVI', color='tab:orange', ls='--', lw=2.0)

#     # x-axis: plain seconds + minor ticks
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
#     ax.xaxis.set_minor_locator(AutoMinorLocator(4))
#     ax.tick_params(axis='x', which='major', length=6)
#     ax.tick_params(axis='x', which='minor', length=3, color='0.3')

#     # y-axis: strict integers 0..binding_sites and small padding
#     # ax.set_ylim(0, binding_sites + 0.2)
#     # ax.set_yticks(np.arange(0, binding_sites + 1))
#     # ax.set_ylabel("Wrapped Sites")
#     from matplotlib.ticker import MultipleLocator

#     ax.set_ylim(0, binding_sites + 0.2)
#     # major ticks every 2 (labelled), minor ticks every 1 (no label)
#     ax.yaxis.set_major_locator(MultipleLocator(2))
#     ax.yaxis.set_minor_locator(MultipleLocator(1))
#     ax.tick_params(axis='y', which='major', length=6)
#     ax.tick_params(axis='y', which='minor', length=3, labelleft=False, color='0.4')
#     ax.set_ylabel("Wrapped Sites")
#     ax.set_title(f"c:{p_conc} $\mathbf{{\mu M}}$," + f" J:{cooperativity} $\mathbf{{k_B T}}$")

#     ax.set_xlabel("Time (s)")
#     ax.grid(which='major', alpha=0.35)
#     ax.grid(which='minor', alpha=0.12)
#     ax.legend(framealpha=0.85,loc='lower left')

#     plt.tight_layout()
#     # plt.show()

#     # save
#     if save_path is not None:
#         for ext in ["png", "pdf", "svg"]:
#             if cooperativity == 0.0:
#                 plt.savefig(save_path / f"with_protamine_nocop_traj.{ext}",
#                             dpi=300,
#                             bbox_inches='tight',
#                             transparent=True)
#             else:
#                 plt.savefig(save_path / f"with_protamine_cop_traj.{ext}",
#                             dpi=300,
#                             bbox_inches='tight',
#                             transparent=True)

# with_protamine_analysis(save_path=thesis_dir / "Chapter_HistProt" / "SSA")
# import sys
# sys.exit(0)














# import numpy as np
# from scipy.optimize import brentq

# import numpy as np

# def T_and_b(f, K):
#     ef2 = np.exp(0.5*f)
#     T = np.array([[1.0, ef2],
#                   [ef2, np.exp(f+K)]], float)
#     b = np.array([1.0, ef2], float)
#     return T, b

# def Zn(n, f, K):
#     T, b = T_and_b(f, K)
#     Tp = np.eye(2)
#     for _ in range(n-1):
#         Tp = Tp @ T
#     return float(b @ (Tp @ b))

# def f_half(n, K, f_lo=-20, f_hi=20):
#     # solve Z_n(f,K)=2 by bisection
#     for _ in range(80):
#         f_mid = 0.5*(f_lo+f_hi)
#         if Zn(n, f_mid, K) > 2.0:
#             f_hi = f_mid
#         else:
#             f_lo = f_mid
#     return 0.5*(f_lo+f_hi)

# def c0_align(n, J_vals, c_mid):
#     f12 = np.array([f_half(n, K) for K in J_vals])  # β=1 ⇒ K=J
#     return np.exp(np.log(c_mid) - f12)              # c0(J)=c_mid / e^{f1/2(J)}


# def f_half_from_Z(n, K):
#     # solve g_n(f,K)=1-1/Z = 0.5  ->  Z(f,K)=2
#     def F(f): return Zn(n, f, K) - 2.0
#     # robust bracket in f ~ ln c : choose wide range
#     return brentq(F, -20.0, +20.0)

# def c0_for_J(n, J_kBT, c_mid_uM, k_on_per_uM_s):
#     f_half = f_half_from_Z(n, K=J_kBT)
#     c0 = c_mid_uM / np.exp(f_half)  # uM
#     k_off = k_on_per_uM_s * c0
#     return c0, k_off



# ...existing code...

# ---- Parameter variants (p_conc, cooperativity) ----
variants = [
    (0.0, 0.0),          # no protamine, no cooperativity
    (0.4, 0.0),        # baseline concentration, no cooperativity
    (0.4, 4.0),       # baseline concentration, high cooperativity
]


k_wrap = 1.0
binding_sites = 14
k_unbind = 0.1
k_bind = 1.0
inf_protamine = True
t_max = 10000
t_steps = 10000
t_points = np.linspace(0, t_max, t_steps)
subids_range = None
max_nucs = 10


variant_results = []

for p_conc, cooperativity in variants:
    print(f"\n--- Running variant p_conc={p_conc}, coop={cooperativity} ---")
    # k_unbind = c0_for_J(n=binding_sites, J_kBT=cooperativity, c_mid_uM=0.15, k_on_per_uM_s=k_bind)[1]
    # print(f"Derived k_unbind={k_unbind:.4f}")

    results_bound = run_simulations_for_condition(
        file_path_bound, ids_bound, subids=subids_range,
        k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
        k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
        inf_protamine=inf_protamine, max_nucs=max_nucs
    )

    results_unbound = run_simulations_for_condition(
        file_path_unbound, ids_unbound, subids=subids_range,
        k_wrap=k_wrap, binding_sites=binding_sites, t_points=t_points,
        k_unbind=k_unbind, k_bind=k_bind, p_conc=p_conc, cooperativity=cooperativity,
        inf_protamine=inf_protamine, max_nucs=max_nucs
    )

    t_block_all = [res[3][-1]/t_max for res in results_bound]
    print(f"Avg fraction blocked (bound): {np.mean(t_block_all):.4f} ± {np.std(t_block_all):.4f} (n={len(t_block_all)})")
    t_block_all = [res[3][-1]/t_max for res in results_unbound]
    print(f"Avg fraction blocked (unbound): {np.mean(t_block_all):.4f} ± {np.std(t_block_all):.4f} (n={len(t_block_all)})")


    if not (results_bound and results_unbound):
        print("Skipping variant (no runs).")
        continue

    # Aggregate (assume identical time grids)
    avg_cs_bound = np.mean(np.vstack([res[1] for res in results_bound]), axis=0)
    avg_cs_unbound = np.mean(np.vstack([res[1] for res in results_unbound]), axis=0)
    ratio = np.divide(avg_cs_unbound, avg_cs_bound,
                      out=np.full_like(avg_cs_bound, np.nan),
                      where=avg_cs_bound > 0)

    variant_results.append(dict(
        p_conc=p_conc,
        coop=cooperativity,
        times=t_points,
        avg_cs_bound=avg_cs_bound,
        avg_cs_unbound=avg_cs_unbound,
        ratio=ratio
    ))

# ---- Plot: averaged cs_total (bound vs unbound) per variant (small multiples) ----
if variant_results:
    n = len(variant_results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, vr in zip(axes, variant_results):
        ax.plot(vr['times'], vr['avg_cs_bound'], label='Bound', color='blue')
        ax.plot(vr['times'], vr['avg_cs_unbound'], label='Unbound', color='black', ls='--')
        ax.set_title(f"p={vr['p_conc']}, coop={vr['coop']}")
        ax.set_xlabel("Time")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Avg cs_total (n_closed)")
    axes[0].legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot: ratio of bound/unbound avg cs_total across variants ----
    plt.figure(figsize=(8,5))
    for vr in variant_results:
        plt.plot(vr['times'], vr['ratio'],
                 label=f"p={vr['p_conc']}, coop={vr['coop']}")
    plt.xlabel("Time")
    plt.ylabel("Avg cs_total Unbound / Bound")
    plt.title("Ratio of Avg Wrapped Sites (Unbound / Bound)")
    plt.grid(alpha=0.3)
    plt.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()
else:
    print("No variant results to plot.")
# ...existing code...