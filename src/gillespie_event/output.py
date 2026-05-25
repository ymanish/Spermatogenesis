"""TSV/parquet writers + merger for the event-driven Gillespie pipeline."""

import csv
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl
import pyarrow as pa

from src.gillespie_event.aggregate import NucleosomeAggregate


TSV_COLUMNS = [
    "id", "subid", "n_replicates",
    "mfpt_uncensored", "rmst", "half_life",
    "final_survival", "censored_fraction",
    "mean_n_open_mom", "mean_n_open_mom_std", "mean_n_open_tw",
    "mean_bprot_mom", "mean_bprot_mom_std", "mean_bprot_tw",
    "n_events_total", "tau_max",
]


def _tsv_row(agg: NucleosomeAggregate) -> list:
    tau_max = float(agg.tau_grid[-1])
    return [
        agg.id, agg.subid, agg.n_replicates,
        agg.mfpt_uncensored, agg.rmst, agg.half_life,
        agg.final_survival, agg.censored_fraction,
        agg.mean_n_open_mom, agg.mean_n_open_mom_std, agg.mean_n_open_tw,
        agg.mean_bprot_mom, agg.mean_bprot_mom_std, agg.mean_bprot_tw,
        agg.n_events_total, tau_max,
    ]


def write_batch_tsv(rows: List[NucleosomeAggregate], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for agg in rows:
            writer.writerow(_tsv_row(agg))


def write_batch_survival(rows: List[NucleosomeAggregate], path: str) -> None:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    records = []
    for agg in rows:
        records.append({
            "id": agg.id,
            "subid": agg.subid,
            "tau_grid": agg.tau_grid.tolist(),
            "survival": agg.survival.tolist(),
            "detach_times": agg.detach_times.tolist(),
            "n_replicates": agg.n_replicates,
            "censored_fraction": agg.censored_fraction,
        })
    df = pd.DataFrame(records)
    df.to_parquet(path, engine="pyarrow", compression="snappy")


def write_batch_trajectories(rows: List[NucleosomeAggregate], path: str) -> None:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    records = []
    for agg in rows:
        for r_idx in range(agg.n_replicates):
            tau_arr = agg.traj_tau[r_idx]
            nclosed_arr = agg.traj_n_closed[r_idx]
            detach_tau = float(agg.detach_times[r_idx])
            import math as _math
            censored = bool(_math.isnan(detach_tau))
            records.append({
                "id": agg.id,
                "subid": agg.subid,
                "replicate": r_idx,
                "traj_tau": tau_arr.tolist(),
                "traj_n_closed": nclosed_arr.astype(int).tolist(),
                "detach_tau": detach_tau,
                "censored": censored,
                "n_events_total": agg.n_events_total // max(agg.n_replicates, 1),
            })
    df = pd.DataFrame(records)
    df.to_parquet(path, engine="pyarrow", compression="snappy")


def _merge_tsv(temp_paths: List[str], out: Path) -> None:
    # Concatenate, keeping the first file's header and skipping the headers
    # of subsequent files.
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as dst:
        for i, p in enumerate(temp_paths):
            with open(p) as src:
                if i == 0:
                    dst.write(src.read())
                else:
                    # skip first line (header)
                    src.readline()
                    shutil.copyfileobj(src, dst)
            os.remove(p)


def _merge_parquet(temp_paths: List[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df_lazy = pl.concat(
        [pl.scan_parquet(p) for p in temp_paths],
        how="vertical",
    )
    df_lazy.sink_parquet(out)
    for p in temp_paths:
        os.remove(p)


def merge_output_files(
    temp_tsv_paths: List[str],
    temp_survival_paths: List[str],
    temp_traj_paths: List[Optional[str]],
    tsv_outfile: Path,
    survival_outfile: Path,
    traj_outfile: Optional[Path],
    n_workers: int,
    logger: logging.Logger,
) -> None:
    os.environ["POLARS_MAX_THREADS"] = str(n_workers)

    _merge_tsv(temp_tsv_paths, Path(tsv_outfile))
    logger.info(f"Summary TSV merged to {tsv_outfile}")

    _merge_parquet(temp_survival_paths, Path(survival_outfile))
    logger.info(f"Survival parquet merged to {survival_outfile}")

    if traj_outfile is not None:
        valid = [p for p in temp_traj_paths if p is not None]
        if valid:
            _merge_parquet(valid, Path(traj_outfile))
            logger.info(f"Trajectories parquet merged to {traj_outfile}")
