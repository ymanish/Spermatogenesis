"""
Adaptive replicate-estimation script for Gillespie simulations.

This script runs pilot or user-specified replicate batches per nucleosome,
estimates detachment-time uncertainty, and determines how many replicates are
needed to reach a target relative confidence-interval width. It uses the
low-level Gillespie simulator and storage helpers directly rather than the
newer high-level simulation CLI.
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *  # Triggers env_settings import
import time
import itertools
import concurrent.futures
from functools import partial
import shutil
from tqdm import tqdm
from pathlib import Path
import argparse

import math
import numpy as np
from statistics import mean, pstdev
from math import sqrt
from scipy.stats import t, norm 
from src.config.var import seed_for
from src.core.gillespie_simulator import GillespieSimulator
from src.core.build_nucleosomes import nucleosome_generator 
from src.core.nucleosomes import Nucleosomes, Nucleosome
from src.core.protamine import protamines
from src.utils.logger_util import get_logger
from src.config.storage import SimulationStorage

import logging
from typing import Optional, List
from src.core.helper.tau_min import _compute_tau_min
import src.core.helper.bkeep as bk

import psutil
import datetime as dt

def z_from_alpha(alpha: float) -> float:
    # Normal quantile for two-sided alpha
    return norm.ppf(1.0 - alpha/2.0)

def ci_for_mean(times: np.ndarray, alpha: float = 0.05):
    # Two-sided (1-alpha) CI for mean (Student-t, robust for moderate N)
    x = times[~np.isnan(times)]
    n = len(x)
    if n == 0:
        return (math.nan, math.nan, math.nan, 0)
    mu = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n > 1 else math.nan
    if n > 1 and s > 0.0:
        tcrit = t.ppf(1.0 - alpha/2.0, df=n-1)
        half = tcrit * s / sqrt(n)
        return (mu, mu - half, mu + half, n)
    else:
        return (mu, math.nan, math.nan, n)

def n_required_for_rel_halfwidth(mu: float, sigma: float, eps: float, alpha: float=0.05) -> int:
    # Target: half-width <= eps * mu at (1-alpha) confidence
    if mu <= 0 or sigma <= 0:
        return 1
    zcrit = z_from_alpha(alpha)
    n = (zcrit * sigma / (eps * mu))**2
    return max(1, int(math.ceil(n)))

def run_reps_for_one_nuc(nuc,
                          R: int, 
                         build_params: dict, 
                         t_points: np.ndarray,
                         inf_protamine: bool, 
                         tau_min: float,
                         start_rep_index: int = 0,
                         save_traj: bool = False,
                         traj_sink: dict | None = None):
    """Run R replicates for a single nucleosome; return list of eviction times.
       Optionally append trajectories to traj_sink (dict of lists) with 'replicate' column."""
    results = []
    for r in range(start_rep_index, start_rep_index + R):
        # Fresh instances per replicate
        nucs = build_params['nucs_factory'](nuc)
        prots = build_params['prot_factory']()
        seed = seed_for(nuc, r)

        sim = GillespieSimulator(nuc_inst=nucs, prot_inst=prots,
                                 t_points=t_points, max_steps=None,
                                 inf_protamine=inf_protamine,
                                 seed=seed, tau_min=tau_min)

        times = []; cs_totals = []; bprots = []; detached_totals = []
        detach_time = math.nan

        for step, state in enumerate(sim.run()):
            if save_traj and traj_sink is not None:
                times.append(state.time); cs_totals.append(state.cs_total)
                bprots.append(state.bprot); detached_totals.append(state.detached_total)
            if detach_time is math.nan and state.detached_total > 0:
                detach_time = state.time

        results.append(detach_time)

        if save_traj and traj_sink is not None:
            total_steps = len(t_points)
            traj_sink['id'].extend([nuc.id]*total_steps)
            traj_sink['subid'].extend([nuc.subid]*total_steps)
            traj_sink['replicate'].extend([r]*total_steps)
            traj_sink['time'].extend(times)
            traj_sink['cs_total'].extend(cs_totals)
            traj_sink['bprot'].extend(bprots)
            traj_sink['detached_total'].extend(detached_totals)

    return results

def run_rep_type(replicate_params: dict,
                 nuc: Nucleosome,
                 build_params: dict,
                 t_points: np.ndarray,
                 inf_protamine: bool,
                 tau_min: float,
                 save_trajectories: bool = False,
                 traj_data: dict | None = None) -> list[float]:
    """Run replicates for one nucleosome according to user specifications.  
       If user_reps is provided, run that many replicates.  
       Otherwise, run adaptive estimation to meet target relative error in CI."""

    user_reps = replicate_params.get('user_reps')
    max_reps = replicate_params.get('max_reps')
    sequential = replicate_params.get('sequential')
    target_rel_error = replicate_params.get('target_rel_error')
    alpha = replicate_params.get('alpha')
    pilot_reps = replicate_params.get('pilot_reps')


    if user_reps is not None:
        # User overrides sample size; just run fixed count
        times = run_reps_for_one_nuc(nuc, R=user_reps, 
                                     build_params=build_params,
                                       t_points=t_points,
                                    inf_protamine=inf_protamine, 
                                    tau_min=tau_min,
                                    start_rep_index=0,
                                    save_traj=save_trajectories)
        return times
    # else:
    #     # Pilot
    #     pilot = max(3, pilot_reps)
    #     times_p = run_reps_for_one_nuc(nuc, pilot, build_params, t_points,
    #                                 inf_protamine, tau_min,
    #                                 start_rep_index=0,
    #                                 save_traj=save_trajectories, 
    #                                 traj_sink=traj_data if sequential else None)

    #     mu_p, _, _, _ = ci_for_mean(np.array(times_p), alpha=alpha)
    #     sigma_p = float(np.std(times_p, ddof=1)) if len(times_p) > 1 else 0.0

    #     n_req = n_required_for_rel_halfwidth(mu_p, sigma_p, target_rel_error, alpha=alpha)
    #     n_req = min(max_reps, max(n_req, pilot))

    #     if sequential:
    #         # Adaptive batches until CI meets target or max reached
    #         batch = max(5, min(20, n_req//5))
    #         times = list(times_p)
    #         next_rep = pilot
    #         while next_rep < max_reps:
    #             # check current CI
    #             mu, lo, hi, n = ci_for_mean(np.array(times), alpha=alpha)
    #             if n >= n_req:
    #                 break
    #             # run one more batch
    #             run = min(batch, max_reps - next_rep)
    #             new_times = run_reps_for_one_nuc(nuc, run, build_params, t_points,
    #                                             inf_protamine, tau_min,
    #                                             start_rep_index=next_rep,
    #                                             save_traj=save_trajectories, traj_sink=traj_data)
    #             times.extend(new_times)
    #             next_rep += run
    #     else:
    #         # One-shot plan: run the rest to reach n_req
    #         rest = n_req - pilot
    #         more = run_reps_for_one_nuc(nuc, rest, build_params, t_points,
    #                                     inf_protamine, tau_min,
    #                                     start_rep_index=pilot,
    #                                     save_traj=save_trajectories, traj_sink=traj_data)
    #         times = list(times_p) + list(more)
    
    return times

def run_replicate_batch(batch: List[Nucleosome],
                  replicate_params: dict,
                  build_params: dict,
                  t_points: np.ndarray,
                  inf_protamine: bool,
                  tau_min: float,
                  k_wrap: float) -> Path:
    
    """Run simulations for a batch of nucleosomes; return paths to temp TSV and Parquet files."""
    start_g = dt.datetime.now()
    proc = psutil.Process(os.getpid())
    bk.WORKER_LOGGER.info("Processing batch of %d sequences", len(batch))


    tmpfile_tsv, writer_tsv = bk.new_batch_writer(fmt="tsv", suffix=".tsv")
    bk.WORKER_LOGGER.info("Temporary file created: %s", tmpfile_tsv)

    for nuc_idx, nuc in enumerate(batch):
        detach_times = run_rep_type(replicate_params=replicate_params,
                 nuc=nuc,
                 build_params=build_params,
                 t_points=t_points,
                 inf_protamine=inf_protamine,
                 tau_min=tau_min,
                 save_trajectories=False)

        writer_tsv.writerow([nuc.id, nuc.subid, detach_times])

    rss = proc.memory_info().rss / 2**20 ## CONVERT FROM BYTES TO MB
    bk.WORKER_LOGGER.info("Batch of %d done by %s; RSS %.1f MB; t %.1fs",
                len(batch), os.getpid(),
                rss, (dt.datetime.now() - start_g).total_seconds())


    bk.WORKER_LOGGER.info("Temporary file %s written", tmpfile_tsv.name)
    return tmpfile_tsv.name


def main(file_path: Path,
         traj_outfile: Path,
         tsv_outfile: Path,
         k_wrap: float,
         prot_params: dict, 
         replicate_params: dict,
         binding_sites: int = 14,
         batch_size: int = 10, 
         n_workers: int = 4,
         flush_every: int = 10000,
         t_points: np.ndarray = None, 
         inf_protamine: bool = True, 
         save_trajectories: bool = False, 
         logger: Optional[logging.Logger] = None) -> None:


    gen = nucleosome_generator(file_path=file_path, k_wrap=k_wrap, binding_sites=binding_sites)
    gen = itertools.islice(gen, 20)
    #### --- define factories for per-replicate instances ---
    build_params = dict(
        nucs_factory=lambda nuc: Nucleosomes(k_wrap=k_wrap,
                                                nucleosomes=[nuc],
                                                binding_sites=binding_sites),
        prot_factory=lambda: protamines(**prot_params)
    )


    batches = bk.batcher(gen, batch_size)

    tau_min = _compute_tau_min(k_wrap=k_wrap, ends=2, gamma=5.0)

    temp_parquet_paths = []
    temp_tsv_paths = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers,
                                                initializer=bk.init_worker, 
                                                initargs=(flush_every,)) as pool:
        func = partial(run_replicate_batch, 
                       k_wrap=k_wrap,
                       replicate_params=replicate_params,
                       build_params=build_params,
                       t_points=t_points,   
                       inf_protamine=inf_protamine,
                       tau_min=tau_min)
        futures = [pool.submit(func, batch=batch) for batch in batches]

        for fut in tqdm(concurrent.futures.as_completed(futures), desc="Processing batches"):
            tsv_path = fut.result()
            temp_tsv_paths.append(tsv_path)
    
    
    HEADER = ['id', 'subid', 'detach_time']
    # Merge temp files into final output
    with open(tsv_outfile, "w") as final_tsv:
        final_tsv.write(("\t".join(HEADER) + "\n"))
        
        for path in temp_tsv_paths:
            with open(path, "r") as src:
                shutil.copyfileobj(src, final_tsv)
            os.remove(path)


    logger.info(f"Trajectories saved to: {traj_outfile}")
    logger.info(f"Summary saved to: {tsv_outfile}")


def arg_parser():
    import argparse 
    parser = argparse.ArgumentParser(description="Run nucleosome simulations with configurable parameters.")
    
    # Input/output arguments
    parser.add_argument("--infile", type=Path, help="Path to the input FASTA file.")
    parser.add_argument("--storage_dir", type=Path, help="Directory to store simulation results.")
    
    # Execution parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Number of sequences per batch.")
    parser.add_argument("--n_workers", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--flush_every", type=int, default=10000, help="Number of rows to flush to disk per batch.")
    
    # Simulation parameters
    parser.add_argument("--k_wrap", type=float, default=1.0, help="Nucleosome wrapping constant.")
    parser.add_argument("--binding_sites", type=int, default=14, help="Number of binding sites.")
    parser.add_argument("--inf_protamine", action="store_true", help="Enable infinite protamine (default: False).")
    
    # Protamine parameters (as individual arguments)
    parser.add_argument("--prot_k_unbind", type=float, default=89.7, help="Protamine unbinding rate.")
    parser.add_argument("--prot_k_bind", type=float, default=1.0, help="Protamine binding rate.")
    parser.add_argument("--prot_p_conc", type=float, default=10.0, help="Protamine concentration.")
    parser.add_argument("--prot_cooperativity", type=float, default=4.5, help="Protamine cooperativity factor.")

    
    # Time points configuration
    parser.add_argument("--t_stop", type=float, default=10000.0, help="Simulation end time.")
    parser.add_argument("--t_num", type=int, default=1000, help="Number of time points.")
    parser.add_argument("--save_trajectories", action="store_true", help="Save trajectory data (default: False).")


    # Replicate control for adaptive estimation
    parser.add_argument("--user-reps", type=int, default=1, help="Fixed number of replicates per nucleosome (overrides adaptive estimation).")
    parser.add_argument("--pilot-reps", type=int, default=10, help="Pilot replicates to estimate variance (per nucleosome).")
    parser.add_argument("--target-rel-error", type=float, default=0.05, help="Target relative half-width (epsilon) of the 95% CI for mean eviction time.")
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level (e.g. 0.05 -> 95% CI).")
    parser.add_argument("--max-reps", type=int, default=50, help="Hard cap on total replicates per nucleosome.")
    parser.add_argument("--sequential", action="store_true", help="If set, run adaptively in batches until CI target is met or max-reps is reached.")

    return parser.parse_args()


if __name__ == "__main__":


    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level='INFO')


    from src.config.path import RESULTS_DIR
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    logger.info(f"Using temporary directory: {tmp_dir}")


    args = arg_parser()

    if args.infile:
        TSV_INFILE = args.infile
    if args.storage_dir:
        STORAGE_DIR = args.storage_dir

    batch_size = args.batch_size
    n_workers = args.n_workers
    flush_every = args.flush_every

    ### Simulation parameters
    k_wrap = args.k_wrap
    # Parse arguments
    TSV_INFILE = Path("/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/exactpoint_unboundpromoter_regions_breath/breath_energy/001.tsv")
    STORAGE_DIR = RESULTS_DIR / f"exactpoint_unboundpromoter_regions_breath/REP/paramexp_{k_wrap}"


    binding_sites = args.binding_sites
    inf_protamine = args.inf_protamine
    prot_k_unbind = args.prot_k_unbind
    prot_k_bind = args.prot_k_bind
    prot_p_conc = args.prot_p_conc
    prot_cooperativity = args.prot_cooperativity
    t_stop = args.t_stop
    t_num = args.t_num
    save_trajectories = args.save_trajectories
    # Group replicate-related arguments into a dictionary
    replicate_params = {
        'user_reps': args.user_reps,
        'pilot_reps': args.pilot_reps,
        'target_rel_error': args.target_rel_error,
        'alpha': args.alpha,
        'max_reps': args.max_reps,
        'sequential': args.sequential
    }


    print("args:", args)
    if not TSV_INFILE.exists():
        raise FileNotFoundError(f"FASTA file {TSV_INFILE} does not exist. Please check the path.")

    if not STORAGE_DIR.exists():
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)


    storage = SimulationStorage(base_dir=STORAGE_DIR)
    t_points = np.linspace(0, t_stop, t_num)
    print(f"prot_p_conc: {prot_p_conc}, prot_k_unbind: {prot_k_unbind}, prot_k_bind: {prot_k_bind}")

    prot_params = {
        'k_unbind': prot_k_unbind,
        'k_bind': prot_k_bind,
        'p_conc': prot_p_conc,
        'cooperativity': prot_cooperativity
    }

    params = {
        'k_wrap': k_wrap,
        'prot_params': prot_params,
        'binding_sites': binding_sites,
        't_max': t_stop,
        't_steps': t_num,
        'inf_protamine': inf_protamine
    }


    file_id = TSV_INFILE.stem
    logger.info(f"Running simulation for file: {TSV_INFILE} with ID: {file_id}")
    output_paths = storage.get_output_paths(params, file_id)
    traj_outfile = output_paths['trajectory']
    tsv_outfile = output_paths['summary']

    main(file_path=TSV_INFILE,
         traj_outfile=traj_outfile,
         tsv_outfile=tsv_outfile,
            k_wrap=k_wrap,
            prot_params=prot_params,
            replicate_params=replicate_params,
            binding_sites=binding_sites,
            batch_size=batch_size,
            n_workers=n_workers,
            t_points=t_points,
            inf_protamine=inf_protamine,
            save_trajectories=save_trajectories
    )


    end = time.perf_counter()
    logger.info(f"Total execution time: {dt.timedelta(seconds=end - start)}")
