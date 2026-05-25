import math
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.config.custom_type import ReactionType
from src.core.gillespie_event_simulator import ReplicateResult
from src.core.nucleosomes import Nucleosome
from src.gillespie_event.aggregate import (
    NucleosomeAggregate,
    aggregate_replicates,
)
from src.gillespie_event.output import (
    write_batch_tsv,
    write_batch_survival,
    write_batch_trajectories,
    merge_output_files,
)


def _mk_agg(nuc_id="t", subid=0, n_rep=3, tau_max=100.0, n_grid=101):
    L = 14
    G = np.zeros((L, L), dtype=float)
    nuc = Nucleosome(nuc_id=nuc_id, subid=subid, sequence=None, G_mat=G,
                     k_wrap=1.0, kT=1.0, binding_sites=L)
    results = []
    for r in range(n_rep):
        results.append(ReplicateResult(
            detach_tau=10.0 + r,
            censored=False,
            final_tau=10.0 + r,
            mean_n_open=2.5,
            mean_bprot=1.5,
            n_events_by_type={rt: 1 for rt in ReactionType},
            traj_tau=np.array([0.0, 5.0, 10.0 + r]),
            traj_n_closed=np.array([14, 7, 0], dtype=np.uint8),
        ))
    tau_grid = np.linspace(0.0, tau_max, n_grid)
    return aggregate_replicates(nuc, results, tau_grid)


def test_write_batch_tsv_writes_expected_columns(tmp_path):
    out = tmp_path / "batch.tsv"
    rows = [_mk_agg("a", 0), _mk_agg("b", 1)]
    write_batch_tsv(rows, str(out))

    df = pd.read_csv(out, sep="\t")
    expected_cols = [
        "id", "subid", "n_replicates",
        "mfpt_uncensored", "rmst", "half_life",
        "final_survival", "censored_fraction",
        "mean_n_open_mom", "mean_n_open_mom_std", "mean_n_open_tw",
        "mean_bprot_mom", "mean_bprot_mom_std", "mean_bprot_tw",
        "n_events_total", "tau_max",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
    assert set(df["id"]) == {"a", "b"}


def test_write_batch_survival_round_trips(tmp_path):
    out = tmp_path / "surv.parquet"
    rows = [_mk_agg("a", 0), _mk_agg("b", 1)]
    write_batch_survival(rows, str(out))

    df = pd.read_parquet(out)
    assert set(df.columns) >= {
        "id", "subid", "tau_grid", "survival",
        "detach_times", "n_replicates", "censored_fraction",
    }
    assert len(df) == 2
    # tau_grid lengths match survival lengths
    for _, row in df.iterrows():
        assert len(row["tau_grid"]) == len(row["survival"])


def test_write_batch_trajectories_round_trips(tmp_path):
    out = tmp_path / "traj.parquet"
    rows = [_mk_agg("a", 0, n_rep=2), _mk_agg("b", 1, n_rep=2)]
    write_batch_trajectories(rows, str(out))

    df = pd.read_parquet(out)
    assert set(df.columns) >= {
        "id", "subid", "replicate",
        "traj_tau", "traj_n_closed", "detach_tau", "censored", "n_events_total",
    }
    # 2 nucleosomes × 2 replicates each = 4 rows
    assert len(df) == 4
    # replicate column contains 0 and 1 for each nucleosome
    counts = df.groupby("id")["replicate"].apply(sorted).to_dict()
    assert counts == {"a": [0, 1], "b": [0, 1]}


def test_merge_output_files_concatenates(tmp_path):
    # Make two batches' worth of temp files
    rows1 = [_mk_agg("a", 0)]
    rows2 = [_mk_agg("b", 1)]
    tmp_tsv1, tmp_tsv2 = tmp_path / "t1.tsv", tmp_path / "t2.tsv"
    tmp_surv1, tmp_surv2 = tmp_path / "s1.parquet", tmp_path / "s2.parquet"
    tmp_traj1, tmp_traj2 = tmp_path / "j1.parquet", tmp_path / "j2.parquet"

    write_batch_tsv(rows1, str(tmp_tsv1))
    write_batch_tsv(rows2, str(tmp_tsv2))
    write_batch_survival(rows1, str(tmp_surv1))
    write_batch_survival(rows2, str(tmp_surv2))
    write_batch_trajectories(rows1, str(tmp_traj1))
    write_batch_trajectories(rows2, str(tmp_traj2))

    # IMPORTANT: write_batch_tsv writes a header. The merger keeps the first
    # header and skips the others. Verify both inputs really have a header.
    out_tsv = tmp_path / "final.tsv"
    out_surv = tmp_path / "final_surv.parquet"
    out_traj = tmp_path / "final_traj.parquet"

    import logging
    merge_output_files(
        temp_tsv_paths=[str(tmp_tsv1), str(tmp_tsv2)],
        temp_survival_paths=[str(tmp_surv1), str(tmp_surv2)],
        temp_traj_paths=[str(tmp_traj1), str(tmp_traj2)],
        tsv_outfile=out_tsv,
        survival_outfile=out_surv,
        traj_outfile=out_traj,
        n_workers=1,
        logger=logging.getLogger("test"),
    )

    df_tsv = pd.read_csv(out_tsv, sep="\t")
    assert len(df_tsv) == 2
    assert set(df_tsv["id"]) == {"a", "b"}

    df_surv = pd.read_parquet(out_surv)
    assert len(df_surv) == 2

    df_traj = pd.read_parquet(out_traj)
    assert len(df_traj) == 6  # 1+1 nucs × 3 replicates each (default in _mk_agg)

    # Temp files removed
    assert not tmp_tsv1.exists()
    assert not tmp_surv1.exists()
    assert not tmp_traj1.exists()


def test_merge_skips_trajectory_when_outfile_none(tmp_path):
    rows = [_mk_agg("a", 0)]
    tsv = tmp_path / "t.tsv"
    surv = tmp_path / "s.parquet"
    write_batch_tsv(rows, str(tsv))
    write_batch_survival(rows, str(surv))

    out_tsv = tmp_path / "final.tsv"
    out_surv = tmp_path / "final_surv.parquet"

    import logging
    merge_output_files(
        temp_tsv_paths=[str(tsv)],
        temp_survival_paths=[str(surv)],
        temp_traj_paths=[None],
        tsv_outfile=out_tsv,
        survival_outfile=out_surv,
        traj_outfile=None,
        n_workers=1,
        logger=logging.getLogger("test"),
    )
    assert out_tsv.exists()
    assert out_surv.exists()
