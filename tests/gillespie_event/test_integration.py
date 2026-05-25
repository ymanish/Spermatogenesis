"""End-to-end test of the event-driven Gillespie pipeline.

Runs the orchestrator on a tiny SPRM dataset slice (max_nucs=2, replicates=3,
tau_max=200, n_workers=1) and asserts:
  - all three output files exist
  - summary TSV is well-formed
  - survival parquet has expected schema and shape
  - trajectories parquet has expected schema and shape
This is the lowest-cost end-to-end exercise of: orchestrator -> batch ->
replicate -> simulator -> aggregate -> writers -> merger.
"""

from pathlib import Path
import os
import shutil

import pandas as pd
import pytest

from src.gillespie_event.config import GillespieEventConfig
from src.gillespie_event.orchestrator import run_gillespie_event


SPRM_DATA = Path("/home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/SPRM_data")
CANDIDATE = SPRM_DATA / "ctrl01_random_genome_safe_stable147_refined"


@pytest.mark.skipif(
    not CANDIDATE.exists(),
    reason=f"SPRM dataset not available at {CANDIDATE}",
)
def test_full_pipeline_runs_on_tiny_slice(tmp_path, monkeypatch):
    # Required for bk.init_worker
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    storage_dir = tmp_path / "out"
    storage_dir.mkdir()

    cfg = GillespieEventConfig(
        k_wrap=1.0,
        binding_sites=14,
        prot_p_conc=0.0,
        prot_cooperativity=0.0,
        tau_max=200.0,
        n_survival_points=51,
        inf_protamine=True,
        replicates=3,
        batch_size=1,
        n_workers=1,
        save_trajectories=True,
    )

    run_gillespie_event(
        dataset_dir=CANDIDATE,
        storage_dir=storage_dir,
        config=cfg,
        dataset_label=None,
        max_nucs=2,
    )

    # Locate output directory (single hashed subdir under storage_dir)
    subdirs = [p for p in storage_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    out_dir = subdirs[0]

    tsv = out_dir / "summaries" / f"{CANDIDATE.name}.tsv"
    surv = out_dir / "survival" / f"{CANDIDATE.name}.parquet"
    traj = out_dir / "trajectories" / f"{CANDIDATE.name}.parquet"
    assert tsv.exists(), f"missing {tsv}"
    assert surv.exists(), f"missing {surv}"
    assert traj.exists(), f"missing {traj}"

    df_tsv = pd.read_csv(tsv, sep="\t")
    assert len(df_tsv) == 2  # max_nucs=2
    for col in ["id", "subid", "n_replicates", "rmst",
                "mean_n_open_mom", "mean_n_open_tw",
                "censored_fraction", "tau_max"]:
        assert col in df_tsv.columns
    assert (df_tsv["n_replicates"] == 3).all()
    assert ((df_tsv["tau_max"] - 200.0).abs() < 1e-6).all()
    assert ((df_tsv["censored_fraction"] >= 0.0) &
            (df_tsv["censored_fraction"] <= 1.0)).all()

    df_surv = pd.read_parquet(surv)
    assert len(df_surv) == 2
    for col in ["id", "subid", "tau_grid", "survival",
                "detach_times", "n_replicates", "censored_fraction"]:
        assert col in df_surv.columns
    # tau_grid + survival lengths match n_survival_points
    for _, row in df_surv.iterrows():
        assert len(row["tau_grid"]) == 51
        assert len(row["survival"]) == 51
        assert len(row["detach_times"]) == 3

    df_traj = pd.read_parquet(traj)
    # 2 nucleosomes × 3 replicates each
    assert len(df_traj) == 6
    for col in ["id", "subid", "replicate", "traj_tau",
                "traj_n_closed", "detach_tau", "censored"]:
        assert col in df_traj.columns
    # Initial n_closed should be 14
    for _, row in df_traj.iterrows():
        assert row["traj_n_closed"][0] == 14
        assert row["traj_tau"][0] == 0.0
