
import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *  # Triggers env_settings import

import itertools
import concurrent.futures
import os
import pickle
from functools import partial
from tqdm import tqdm
from src.core.nucleosomes import Nucleosomes, Nucleosome
from src.core.protamine import protamines
from src.core.gillespie_simulator import GillespieSimulator
from src.core.build_nucleosomes import nucleosome_generator 
import src.core.helper.bkeep as bk
import datetime as dt
from typing import List, Iterable, Optional
import psutil
import numpy as np
from src.utils.logger_util import get_logger
import polars as pl
import shutil
from pathlib import Path
from src.config.storage import SimulationStorage
import time 
import pandas as pd
import pyarrow as pa
import logging

def batcher(it, size):  
    it = iter(it)
    for first in it:
        batch = list(itertools.chain([first], itertools.islice(it, size - 1)))
        yield batch


def run_batch_simulations(batch: List[Nucleosome], 
                          k_wrap: float, 
                          tau_min: float,
                          prot_params: dict, 
                          t_points: np.ndarray, 
                          inf_protamine: bool = True, 
                          kT: float = 1.0, 
                          binding_sites: int = 14, 
                          eq_frac: float = 0.25, 
                          save_trajectories: bool = False):
    """
    Run Gillespie simulations for a batch of nucleosomes, return temp file path with pickled results.
    """
    assert 0.0 < eq_frac <= 1.0, "eq_frac must be between 0 and 1, Telling how much of the trajectory to consider for equilibrium"
    start_g = dt.datetime.now()
    proc = psutil.Process(os.getpid())
    bk.WORKER_LOGGER.info("Processing batch of %d sequences", len(batch))

    # Temp Parquet for trajectories
    traj_data = {'id': [], 'subid': [], 'time': [], 'cs_total': [], 'bprot': [], 'detached_total': []}

    tmpfile_tsv, writer_tsv = bk.new_batch_writer(fmt="tsv", suffix=".tsv")
    bk.WORKER_LOGGER.info("Temporary file created: %s", tmpfile_tsv)

    parquet_path = None
    if save_trajectories:
        parquet_path = bk.new_batch_writer(fmt="parquet", suffix=".parquet")
        bk.WORKER_LOGGER.info("Temporary Parquet file created: %s", parquet_path)

    for nuc_idx, nuc in enumerate(batch):
        nucs = Nucleosomes(
            k_wrap       = k_wrap,
            kT           = kT,
            nucleosomes  = [nuc], # Wrap single nucleosome
            binding_sites= binding_sites,
        ) # Wrap single
        prot_inst = protamines(**prot_params)
        sim = GillespieSimulator(nuc_inst=nucs, 
                                 prot_inst=prot_inst,
                                   t_points=t_points, 
                                      max_steps=None,
                                   inf_protamine=inf_protamine, 
                                   seed=42, 
                                   tau_min=tau_min)

        # Process states iteratively (avoid list(sim.run()))
        times = []
        cs_totals = []
        bprots = []
        detached_totals = []
        eq_cs_sum = 0
        eq_bprot_sum = 0
        eq_count = 0
        detach_time = -1.0
        total_steps = len(t_points)
        eq_start = int(total_steps * (1 - eq_frac))
        
        for step, state in enumerate(sim.run()):  # run() yields one by one

            if save_trajectories:
                times.append(state.time)
                cs_totals.append(state.cs_total)
                bprots.append(state.bprot)
                detached_totals.append(state.detached_total)  # Single nuc
            
            if state.detached_total > 0 and detach_time < 0:
                detach_time = state.time  # First detachment
            
            if step >= eq_start:
                eq_cs_sum += state.cs_total
                eq_bprot_sum += state.bprot
                eq_count += 1
        
        # Trajectories to Parquet data (still lists, but smaller if eq_frac high)
        if save_trajectories:
            ids_arr = np.full(total_steps, nuc.id)
            subids_arr = np.full(total_steps, nuc.subid)
            traj_data['id'].extend(ids_arr)
            traj_data['subid'].extend(subids_arr)
            traj_data['time'].extend(times)
            traj_data['cs_total'].extend(cs_totals)
            traj_data['bprot'].extend(bprots)
            traj_data['detached_total'].extend(detached_totals)

        # Means for TSV
        eq_cs = eq_cs_sum / eq_count if eq_count > 0 else 0
        eq_bprot = eq_bprot_sum / eq_count if eq_count > 0 else 0
        writer_tsv.writerow([nuc.id, nuc.subid, eq_cs, eq_bprot, detach_time])

    rss = proc.memory_info().rss / 2**20 ## CONVERT FROM BYTES TO MB
    bk.WORKER_LOGGER.info("Batch of %d done by %s; RSS %.1f MB; t %.1fs",
                len(batch), os.getpid(),
                rss, (dt.datetime.now() - start_g).total_seconds())

    if save_trajectories:
        pa.set_cpu_count(1)  # Limits compute threads (e.g., for encoding/compression)
        pa.set_io_thread_count(1)  # Limits I/O threads (e.g., for file handling)
        df = pd.DataFrame(traj_data)
        df.to_parquet(parquet_path, engine="pyarrow")
    else:
        parquet_path = None

    bk.WORKER_LOGGER.info("Temporary file %s written", tmpfile_tsv.name)
    return tmpfile_tsv.name, parquet_path




def _compute_tau_min(k_wrap: float, ends: int = 2, gamma: float = 5.0) -> float:
    # If either end can initiate rewrap from fully unwrapped, ends=2
    import math
    w0 = ends * float(k_wrap)
    t099 = math.log(100.0) / w0
    return gamma * t099

def main(
         file_path: Path,
         traj_outfile: Path,
         tsv_outfile: Path,
        k_wrap: float,
        prot_params: dict, 
        binding_sites: int = 14,
         batch_size: int = 10, 
         n_workers: int = 4,
         flush_every: int = 10000,
         t_points: np.ndarray = None, 
         inf_protamine: bool = True, 
         save_trajectories: bool = False, logger: Optional[logging.Logger] = None) -> None:

    # if prot_params is None:
    #     prot_params = {'k_unbind': 0.23, 'k_bind': 2113, 'p_conc': 0.05, 'cooperativity': 1.0}

    ### Generator of nucleosomes
    gen = nucleosome_generator(file_path=file_path, k_wrap=k_wrap, binding_sites=binding_sites, subids=np.arange(2000, 2050).tolist())
    gen = itertools.islice(gen, 200)
    # Define how many copies you want

    # gen = nucleosome_generator(file_path=file_path, k_wrap=k_wrap, binding_sites=binding_sites,
    #                                 subids=[2066, 2076])  # Example subids
    # # Pick one sequence from the generator (e.g., the first one)
    # first_sequence = next(gen)

    # # Now pass the replicated generator to the batcher
    # n = 200
    # replicated_gen_list = [
    #     Nucleosome(
    #         nuc_id=first_sequence.id,
    #         subid=first_sequence.subid,
    #         sequence=first_sequence.sequence,
    #         G_mat=first_sequence.G_mat.copy(),  # ensure a new copy of the matrix
    #         k_wrap=k_wrap,
    #         kT=first_sequence.kT,
    #         binding_sites=binding_sites
    #     )
    #     for _ in range(n)
    # ]

    # batches = batcher(replicated_gen_list, batch_size)


    batches = batcher(gen, batch_size)


    tau_min = _compute_tau_min(k_wrap=k_wrap, ends=2, gamma=5.0)


    temp_parquet_paths = []
    temp_tsv_paths = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers,
                                                initializer=bk.init_worker, 
                                                initargs=(flush_every,)) as pool:
        func = partial(run_batch_simulations, 
                       prot_params=prot_params, 
                       t_points=t_points, 
                       binding_sites=binding_sites,
                       inf_protamine=inf_protamine, 
                       save_trajectories=save_trajectories,
                       tau_min=tau_min)
        futures = [pool.submit(func, batch=batch, k_wrap=k_wrap) for batch in batches]

        for fut in tqdm(concurrent.futures.as_completed(futures), desc="Processing batches"):
            tsv_path, parquet_path = fut.result()
            temp_parquet_paths.append(parquet_path)
            temp_tsv_paths.append(tsv_path)
    if save_trajectories:
        os.environ['POLARS_MAX_THREADS'] = str(n_workers)
        logger.info("All batches processed, writing final output files")
        # Merge Parquet (lazy concat)
        df_lazy = pl.concat([pl.scan_parquet(p) for p in temp_parquet_paths], how='vertical')
        df_lazy.sink_parquet(traj_outfile)  # Writes without full materialization

    else:
        traj_outfile = None
    HEADER = ['id', 'subid', 'cs_total', 'bprot', 'detach_time']
    # Merge temp files into final output
    with open(tsv_outfile, "w") as final_tsv:
        final_tsv.write(("\t".join(HEADER) + "\n"))
        
        for path in temp_tsv_paths:
            with open(path, "r") as src:
                shutil.copyfileobj(src, final_tsv)
            os.remove(path)
    if save_trajectories:
        for p in temp_parquet_paths:
            os.remove(p)

    logger.info(f"Trajectories saved to: {traj_outfile}")
    logger.info(f"Summary saved to: {tsv_outfile}")


def arg_parser():
    import argparse 
    parser = argparse.ArgumentParser(description="Run nucleosome simulations with configurable parameters.")
    
    # Input/output arguments
    parser.add_argument("--infile", type=Path, help="Path to the input FASTA file.")
    parser.add_argument("--storage_dir", type=Path, help="Directory to store simulation results.")
    
    # Execution parameters
    parser.add_argument("--batch_size", type=int, default=50, help="Number of sequences per batch.")
    parser.add_argument("--n_workers", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--flush_every", type=int, default=10000, help="Number of rows to flush to disk per batch.")
    
    # Simulation parameters (with sensible defaults)
    parser.add_argument("--k_wrap", type=float, default=1.0, help="Nucleosome wrapping constant.")
    parser.add_argument("--binding_sites", type=int, default=14, help="Number of binding sites.")
    parser.add_argument("--inf_protamine", action="store_true", help="Enable infinite protamine (default: False).")
    
    # Protamine parameters (as individual arguments)
    parser.add_argument("--prot_k_unbind", type=float, default=0.01, help="Protamine unbinding rate.")
    parser.add_argument("--prot_k_bind", type=float, default=10.0, help="Protamine binding rate.")
    parser.add_argument("--prot_p_conc", type=float, default=0.0, help="Protamine concentration.")
    parser.add_argument("--prot_cooperativity", type=float, default=0.0, help="Protamine cooperativity factor.")
    
    # Time points configuration
    parser.add_argument("--t_stop", type=float, default=10.0, help="Simulation end time.")
    parser.add_argument("--t_num", type=int, default=10000, help="Number of time points.")

    parser.add_argument("--save_trajectories", action="store_true", help="Save trajectory data (default: False).")

    return parser.parse_args()

if __name__ == "__main__":

    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level='INFO')


    from src.config.path import RESULTS_DIR
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    logger.info(f"Using temporary directory: {tmp_dir}")


    # TSV_INFILE = Path("/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/hamnucret_data/boundprom/breath_energy/001.tsv") 
    # STORAGE_DIR = RESULTS_DIR /"boundprom/GSim"


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
    binding_sites = args.binding_sites
    inf_protamine = args.inf_protamine
    prot_k_unbind = args.prot_k_unbind
    prot_k_bind = args.prot_k_bind
    prot_p_conc = args.prot_p_conc
    prot_cooperativity = args.prot_cooperativity
    t_stop = args.t_stop
    t_num = args.t_num
    save_trajectories = args.save_trajectories

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
            binding_sites=binding_sites,
            batch_size=batch_size,
            n_workers=n_workers,
            t_points=t_points,
            inf_protamine=inf_protamine,
            save_trajectories=save_trajectories
    )


    end = time.perf_counter()
    logger.info(f"Total execution time: {dt.timedelta(seconds=end - start)}")