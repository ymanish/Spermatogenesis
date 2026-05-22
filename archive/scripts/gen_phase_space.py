
import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *  # Triggers env_settings import


import matplotlib.pyplot as plt
import seaborn as sns  # For better heatmaps
# from matplotlib.colors import DivergingNorm
from tqdm import tqdm
import numpy as np
from pathlib import Path
from src.config.storage import SimulationStorage
from src.core.build_nucleosomes import nucleosome_generator
from src.scripts.exec_sim import main
import polars as pl
from typing import Optional, Iterable, List
from src.utils.logger_util import get_logger
from scipy.stats import ks_2samp, linregress

def compute_survival_from_tsv_polars(tsv_df: pl.LazyFrame, t_max: float, t_steps: int, bp_bound: [int, int]):
    shift_value = 73 - 2000
    tsv_df = tsv_df.with_columns((pl.col("subid").cast(pl.Int32) + shift_value).alias("subid"))

    # Apply binding probability bounds
    tsv_df = tsv_df.filter((pl.col("subid") >= bp_bound[0]) & (pl.col("subid") <= bp_bound[1]))

    # Replace -1 with t_max + 1 and clip negative values to 0.
    tsv_df = tsv_df.with_columns(
        pl.when(pl.col("detach_time") < 0)
            .then(t_max + 1)
            .otherwise(pl.col("detach_time"))
            .alias("detach_time")
    )

    df_eager = tsv_df.drop("cs_total", "bprot", "id", "subid").collect(engine="streaming")
    total_particles = df_eager.height
    detach_times = df_eager["detach_time"].to_numpy()

    # Create time grid and compute survival probability S(t)
    times = np.linspace(0, t_max, t_steps)
    s_t = np.array([np.sum(detach_times > t) / total_particles for t in times])
    
    # Compute AUC
    auc = np.trapezoid(s_t, times)  # Integral of S(t) dt
    
    return times, s_t, auc



def read_tsv_from_storage(storage: SimulationStorage, first: int = 1, **params) -> pl.LazyFrame:
    """
    Read and concatenate all TSV files in the storage directory using Polars.
    Assumes that storage.path is a Path object pointing to the storage folder.
    """

    matches = storage.find_simulations(**params)
    if matches.empty:
        raise ValueError(f"No matching simulation found for parameters: {params}")
    param_hash = matches["param_hash"].values[0]
    # Get all TSV files in the storage folder
    all_tsv_files = []
    for i in range(1, first+1):
        try:
            results = storage.load_simulation(param_hash, file_id=f"{i:03d}")
        except FileNotFoundError as e:
            print(f"Skipping simulation {i:03d}: {e}")
            continue

        # Access files
        traj_path = results['trajectory']
        summary_path = results['summary']
        # print(f"Loaded trajectory path: {traj_path}")
        # print(f"Loaded summary path: {summary_path}")

        if summary_path is None or not Path(summary_path).exists():
            print(f"Summary file {summary_path} does not exist!")
            continue
        all_tsv_files.append(summary_path)
    
    if not all_tsv_files:
        raise ValueError(f"No TSV files found for the given parameters: {params}")
    # Read and concatenate all files together
    df_list = [pl.read_csv(tsv_file, separator="\t") for tsv_file in all_tsv_files]

    return pl.concat(df_list).lazy()


def exp_decay(t, lambda_):
    return np.exp(-lambda_ * t)

def get_metric_vals(metric_type, times_bound, probs_bound, times_unbound, probs_unbound, t_max):
    # Compute type-specific values
    if metric_type == 'auc':
        val_bound = np.trapezoid(probs_bound, times_bound)
        val_unbound = np.trapezoid(probs_unbound, times_unbound)

    elif metric_type == 'half_life':
        def get_half_life(times, probs, t_max):
            if np.min(probs) >= 0.5:
                return t_max * 2  # Stable: large finite value
            return np.interp(0.5, probs[::-1], times[::-1])
        
        val_bound = get_half_life(times_bound, probs_bound, t_max)
        val_unbound = get_half_life(times_unbound, probs_unbound, t_max)

    elif metric_type == 'ks':
        ks_stat, _ = ks_2samp(probs_bound, probs_unbound)
        sign = np.sign(np.mean(probs_bound - probs_unbound))  # Positive if bound higher
        val_bound = ks_stat * sign if metric_type == 'ks' else None  # For KS, metric is signed stat (no norm needed)
        val_unbound = 0  # KS is difference, so metric = val_bound

    elif metric_type == 'decay_rate':
        def get_decay_rate(times, probs):
            mask = probs > 0
            log_probs = np.log(probs[mask])
            slope = linregress(times[mask], log_probs).slope
            return -slope  # Positive rate
        val_bound = get_decay_rate(times_bound, probs_bound)
        val_unbound = get_decay_rate(times_unbound, probs_unbound)
        # For rate, invert sense: higher rate = faster drop, so metric (unbound - bound) / sum
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    
    return val_bound, val_unbound

def run_main_simulation(file_path: Path, 
                        params: dict, 
                        storage: SimulationStorage,
                          t_points: np.ndarray, 
                          n_workers: int = 4,
                            batch_size: int = 10, 
                            inf_protamine: bool = True, 
                            save_trajectories: bool = False):

    file_id = file_path.stem
    logger.info(f"Running simulation for file: {file_path} with ID: {file_id}")
    output_paths = storage.get_output_paths(params, file_id)
    traj_outfile = output_paths['trajectory']
    tsv_outfile = output_paths['summary']
    
    main(file_path=file_path, traj_outfile=traj_outfile, tsv_outfile=tsv_outfile,  
         k_wrap=params['k_wrap'],
        prot_params=params['prot_params'], 
        binding_sites=14,
        batch_size=batch_size,
        n_workers=n_workers,
        t_points=t_points,
        inf_protamine=inf_protamine,
        save_trajectories=save_trajectories, logger=logger)


def generate_phase_space(p_conc_values: np.ndarray, 
                         coop_values: np.ndarray, 
                            k_bind_values: np.ndarray,
                         bound_storage: SimulationStorage, 
                         unbound_storage: SimulationStorage, 
                         base_params: dict, 
                         t_max: float, 
                         t_steps: int, 
                         bp_bound: list[int],
                         n_workers: int = 4,
                            batch_size: int = 10,
                            inf_protamine: bool = True,
                            save_trajectories: bool = False,
                            param_file: Optional[Path] = None, 
                            plot_file: Optional[Path] = None):
    """
    Generate phase space metric over p_conc and coop grids.
    :param p_conc_values: Array of p_conc to test (e.g., np.logspace(-4, 2, 10))
    :param coop_values: Array of cooperativity to test (e.g., np.linspace(0, 20, 10))
    :param base_params: Dict of fixed params (copy and update prot_params['p_conc'], ['cooperativity'])
    :param plot_file: Optional path to save plot
    :return: metrics matrix, fig
    """
    metrics = np.zeros((len(coop_values), len(p_conc_values), len(k_bind_values)))
    # auc = np.zeros((len(coop_values), len(p_conc_values)))
    half_life = np.zeros((len(coop_values), len(p_conc_values), len(k_bind_values)))
    decay_rate = np.zeros((len(coop_values), len(p_conc_values), len(k_bind_values)))
    for i, coop in enumerate(tqdm(coop_values, desc="Coop loop")):
        for j, p_conc in enumerate(tqdm(p_conc_values, desc="p_conc loop", leave=False)):
            for k, k_bind in enumerate(tqdm(k_bind_values, desc="k_bind loop", leave=False)):

                if p_conc == 0 and coop != 0:
                    logger.info(f"Skipping simulation for p_conc=0 and coop={coop}")
                    metrics[i, j, k] = np.nan  # or assign a default value (e.g. 0)
                    # auc[i, j] = np.nan
                    half_life[i, j, k] = np.nan
                    decay_rate[i, j, k] = np.nan
                    continue

                # Update params for this point
                params = base_params.copy()
                params['prot_params'] = params['prot_params'].copy()
                params['prot_params']['p_conc'] = p_conc
                params['prot_params']['cooperativity'] = coop
                params['prot_params']['k_bind'] = k_bind
                t_points = np.linspace(0, t_max, t_steps)

                # Run main for bound and unbound (assumes main updates TSVs)

                run_main_simulation(file_path=BOUND_INFILE, 
                                    params=params, 
                                    storage=bound_storage,
                                    t_points=t_points, 
                                    n_workers=n_workers, 
                                    batch_size=batch_size, 
                                    inf_protamine=inf_protamine, 
                                    save_trajectories=save_trajectories)


                run_main_simulation(file_path=UNBOUND_INFILE, 
                                    params=params, 
                                    storage=unbound_storage,
                                    t_points=t_points, 
                                    n_workers=n_workers, 
                                    batch_size=batch_size, 
                                    inf_protamine=inf_protamine, 
                                    save_trajectories=save_trajectories)

                # Load TSVs
                bound_lz_df = read_tsv_from_storage(bound_storage, first=10, **params)
                unbound_lz_df = read_tsv_from_storage(unbound_storage, first=10, **params)

                # Compute AUCs
                times_bound, probs_bound, auc_bound = compute_survival_from_tsv_polars(bound_lz_df, t_max, t_steps, bp_bound)
                times_unbound, probs_unbound, auc_unbound = compute_survival_from_tsv_polars(unbound_lz_df, t_max, t_steps, bp_bound)
                val_bound, val_unbound = get_metric_vals('half_life', times_bound, probs_bound, times_unbound, probs_unbound, t_max)
                # Metric
                denom = auc_bound + auc_unbound
                metrics[i, j, k] = (auc_bound - auc_unbound) / denom if denom != 0 else 0
                half_life[i, j, k] = (val_bound - val_unbound) / (val_bound + val_unbound) if (val_bound + val_unbound) != 0 else 0
                
                val_bound, val_unbound = get_metric_vals('decay_rate', times_bound, probs_bound, times_unbound, probs_unbound, t_max)
                decay_rate[i, j, k] = (val_bound - val_unbound) / (val_bound + val_unbound) if (val_bound + val_unbound) != 0 else 0




    import pandas as pd
    p_conc_mesh, coop_mesh, k_bind_mesh = np.meshgrid(p_conc_values, coop_values, k_bind_values)
    df_long = pd.DataFrame({
        'p_conc': p_conc_mesh.ravel(),
        'coop': coop_mesh.ravel(),
        'k_bind': k_bind_mesh.ravel(),
        'metric': metrics.ravel(),
        # 'auc': auc.ravel(),
        'half_life': half_life.ravel(),
         'decay_rate': decay_rate.ravel()
    })
    df_long.to_csv(param_file, index=False)
    logger.info("Raw metrics saved to phase_space_raw_metrics.csv in long format")

    # ##### Plot: Check size and use contourf or fallback
    # fig, ax = plt.subplots(figsize=(10, 8))
    # if metrics.shape[0] < 2 or metrics.shape[1] < 2:
    #     logger.warning("Grid too small for contour; using scatter plot instead.")
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     p_conc_mesh, coop_mesh = np.meshgrid(p_conc_values, coop_values)
    #     scatter = ax.scatter(p_conc_mesh.flatten(), coop_mesh.flatten(), 
    #                  c=metrics.flatten(), cmap='RdBu', 
    #                  norm=plt.Normalize(-1, 1), s=100)
    #     fig.colorbar(scatter, ax=ax, label='Metric Value')
    #     for i in range(len(coop_values)):
    #         for j in range(len(p_conc_values)):
    #             ax.text(p_conc_values[j], coop_values[i], f"{metrics[i, j]:.2f}", ha='center', va='center', color='black')
    # else:
    #     levels = np.linspace(-1, 1, 21)
    #     contour = ax.contourf(p_conc_values, coop_values, metrics, levels=levels, cmap='RdBu', norm=plt.Normalize(-1, 1))
    #     fig.colorbar(contour, ax=ax, label='Metric Value')
    #     ax.contour(p_conc_values, coop_values, metrics, levels=[0], colors='black', linestyles='dashed')

    # ax.set_xlabel('p_conc')
    # ax.set_ylabel('cooperativity')
    # ax.set_title('(S_b - S_un) / (S_b + S_un)')
    # ax.set_xscale('log')
    # ax.grid(True)
        
    # if plot_file:
    #     fig.savefig(plot_file)
    #     logger.info(f"Phase space saved to {plot_file}")
    

if __name__ == "__main__":
    import time
    start = time.perf_counter()

    from src.utils.logger_util import get_logger
    from src.config.path import RESULTS_DIR, HAMNUCRET_DATA_DIR
    from pathlib import Path
    import os

    logger = get_logger(__name__, level="INFO")
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)


    params1 = {
    'k_wrap': 10.0,
    'prot_params': {
        'k_unbind': 0.01,
        'k_bind': 10.0,
        'p_conc': 0.001,
        'cooperativity': 10.0
    },
    'binding_sites': 14,
    't_max': 5000.0,
    't_steps': 10000,
    'inf_protamine': True

    }

    # lower = np.logspace(-3, -1, 3)    # 0.001, 0.01, 0.1
    # upper = np.logspace(0, 2, 3)      # 1.0, 10, 100
    # p_conc_grid = np.concatenate((lower, [0], upper))
    # coop_grid = np.linspace(0, 10, 5)
    # k_bind = np.logspace(-2, 2, 5)  # Example range for k_bind
    p_conc_grid = [2.0]
    coop_grid = [0, 10]  # 0.001 to 100
    k_bind = [100.0]

    print (f"p_conc_grid: {p_conc_grid}")
    print (f"coop_grid: {coop_grid}")
    print (f"k_bind: {k_bind}")
    # import sys
    # sys.exit()

    # #### For Cluster #######
    # bound_dir  = HAMNUCRET_DATA_DIR / "minpoint_boundpromoter_regions_breath"
    # unbound_dir = HAMNUCRET_DATA_DIR / "minpoint_unboundpromoter_regions_breath"
    # BOUND_INFILE = bound_dir / "breath_energy/001.tsv"
    # UNBOUND_INFILE = unbound_dir / "breath_energy/001.tsv"

    # BOUND_STORAGE_dir = RESULTS_DIR / f"minpoint_boundpromoter_regions_breath/paramexp_{params1['k_wrap']}"
    # UNBOUND_STORAGE_dir = RESULTS_DIR / f"minpoint_unboundpromoter_regions_breath/paramsexp_{params1['k_wrap']}"
    # PHASE_SPACE_DIR = RESULTS_DIR / f"minpoint_phase_space_metrics"


    #####################

    
    BOUND_INFILE = HAMNUCRET_DATA_DIR / "boundprom/breath_energy/001.tsv"
    UNBOUND_INFILE = HAMNUCRET_DATA_DIR / "unboundprom/breath_energy/001.tsv"

    BOUND_STORAGE_dir = RESULTS_DIR / f"boundprom/paramexp_{params1['k_wrap']}"
    UNBOUND_STORAGE_dir = RESULTS_DIR / f"unboundprom/paramexp_{params1['k_wrap']}"
    PHASE_SPACE_DIR = RESULTS_DIR / f"phase_space_metrics"


    if not BOUND_STORAGE_dir.exists():
        BOUND_STORAGE_dir.mkdir(parents=True, exist_ok=True)
    if not UNBOUND_STORAGE_dir.exists():
        UNBOUND_STORAGE_dir.mkdir(parents=True, exist_ok=True)
    if not PHASE_SPACE_DIR.exists():
        PHASE_SPACE_DIR.mkdir(parents=True, exist_ok=True)

    storage_bound = SimulationStorage(BOUND_STORAGE_dir)
    storage_unbound = SimulationStorage(UNBOUND_STORAGE_dir)


    generate_phase_space(p_conc_values=p_conc_grid,
                        coop_values=coop_grid,
                        k_bind_values=k_bind,
                        bound_storage=storage_bound, 
                        unbound_storage=storage_unbound, 
                        base_params=params1, 
                        t_max=params1['t_max'],
                        t_steps=params1['t_steps'],
                        bp_bound=(-500, 500),
                        n_workers=20,
                        batch_size=10,
                        inf_protamine=True,
                        save_trajectories=False,
                        param_file=PHASE_SPACE_DIR / f"phase_space_raw_metrics_{params1['k_wrap']}.csv",
                        plot_file=PHASE_SPACE_DIR / "phase_space.png")

    end = time.perf_counter()
    logger.info(f"Phase space generation completed in {end - start:.2f} seconds")