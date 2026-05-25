"""Top-level orchestrator for the event-driven Gillespie pipeline."""

import concurrent.futures
import itertools
import logging
from functools import partial
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config.storage import SimulationStorage
from src.core.build_nucleosomes import nucleosome_generator_sprm
from src.core.helper import bkeep as bk
from src.gillespie_event.batch import run_batch
from src.gillespie_event.config import GillespieEventConfig
from src.gillespie_event.output import merge_output_files


def run_gillespie_event(
    dataset_dir: Path,
    storage_dir: Path,
    config: GillespieEventConfig,
    dataset_label: Optional[str] = None,
    max_nucs: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        from src.utils.logger_util import get_logger
        logger = get_logger(__name__, log_file=None, level="INFO")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    file_id = dataset_dir.name
    if dataset_label:
        file_id = f"{dataset_label}_{file_id}"

    params_for_storage = {
        "k_wrap": config.k_wrap,
        "prot_params": config.prot_params,
        "binding_sites": config.binding_sites,
        "tau_max": config.tau_max,
        "inf_protamine": config.inf_protamine,
        # Add a marker so the hash differs from the old grid-based pipeline
        "pipeline": "gillespie_event",
    }
    storage = SimulationStorage(base_dir=storage_dir, use_index=False)
    paths = storage.get_output_paths(params_for_storage, file_id)
    tsv_outfile = paths["summary"]
    survival_outfile = paths["survival"]
    traj_outfile = paths["trajectory"] if config.save_trajectories else None

    logger.info(f"Configuration: {config}")
    logger.info(f"Outputs:\n  TSV       = {tsv_outfile}\n"
                f"  Survival  = {survival_outfile}\n"
                f"  Trajectories = {traj_outfile}")

    gen = nucleosome_generator_sprm(
        dataset_dir=dataset_dir,
        k_wrap=config.k_wrap,
        kT=1.0,
        binding_sites=config.binding_sites,
    )
    if max_nucs is not None:
        gen = itertools.islice(gen, max_nucs)
    batches = bk.batcher(gen, config.batch_size)

    func = partial(
        run_batch,
        prot_params=config.prot_params,
        tau_max=config.tau_max,
        n_survival_points=config.n_survival_points,
        inf_protamine=config.inf_protamine,
        replicates=config.replicates,
        save_trajectories=config.save_trajectories,
    )

    tmp_tsvs, tmp_survs, tmp_trajs = [], [], []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.n_workers,
        initializer=bk.init_worker,
        initargs=(config.flush_every,),
    ) as pool:
        futures = [pool.submit(func, batch=batch) for batch in batches]
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="Batches"):
            tsv, surv, traj = fut.result()
            tmp_tsvs.append(tsv)
            tmp_survs.append(surv)
            tmp_trajs.append(traj)

    merge_output_files(
        temp_tsv_paths=tmp_tsvs,
        temp_survival_paths=tmp_survs,
        temp_traj_paths=tmp_trajs,
        tsv_outfile=tsv_outfile,
        survival_outfile=survival_outfile,
        traj_outfile=traj_outfile,
        n_workers=config.n_workers,
        logger=logger,
    )
