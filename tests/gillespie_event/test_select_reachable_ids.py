"""Unit tests for the reachable-nucleosome sampler core (select_ids)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load the standalone cluster script as a module.
_SCRIPT = (Path(__file__).resolve().parents[2]
           / "cluster_sim_scripts" / "gillespie_event" / "select_reachable_ids.py")
_spec = importlib.util.spec_from_file_location("select_reachable_ids", _SCRIPT)
sri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sri)


def _summary(n=50):
    # subids 0..n-1; mfpt = subid so the reachability cutoff is exactly the cap.
    return pd.DataFrame({
        "id": [f"peak_{i}" for i in range(n)],
        "subid": list(range(n)),
        "mfpt": [float(i) for i in range(n)],
        "mfpt_flag": ["ok"] * n,
    })


def test_filters_by_cap_and_caps_count():
    df = _summary(50)  # mfpt = subid, so subids 0..9 are < cap=10
    ids, n_reach = sri.select_ids(df, cap=10.0, n=100, seed=1)
    assert n_reach == 10
    assert len(ids) == 10          # min(100, 10)
    assert set(ids) == set(range(10))


def test_takes_n_when_more_reachable():
    df = _summary(50)  # all 50 reachable under a huge cap
    ids, n_reach = sri.select_ids(df, cap=1e9, n=20, seed=1)
    assert n_reach == 50
    assert len(ids) == 20
    assert all(0 <= i < 50 for i in ids)
    assert len(set(ids)) == 20     # no replacement
    assert ids == sorted(ids)      # returned sorted


def test_deterministic_in_seed():
    df = _summary(50)
    a, _ = sri.select_ids(df, cap=1e9, n=20, seed=42)
    b, _ = sri.select_ids(df, cap=1e9, n=20, seed=42)
    c, _ = sri.select_ids(df, cap=1e9, n=20, seed=43)
    assert a == b
    assert a != c                  # different seed -> different draw


def test_excludes_non_ok_and_nonfinite():
    df = _summary(10)              # subids 0..9, mfpt 0..9, all < cap
    df.loc[df.subid == 3, "mfpt_flag"] = "underflowed"
    df.loc[df.subid == 5, "mfpt"] = np.inf
    ids, n_reach = sri.select_ids(df, cap=1e9, n=100, seed=1)
    assert n_reach == 8
    assert 3 not in ids and 5 not in ids


def test_empty_when_none_reachable():
    df = _summary(50)              # min mfpt is 0.0
    ids, n_reach = sri.select_ids(df, cap=0.0, n=10, seed=1)
    assert ids == []
    assert n_reach == 0


# ── Top-up of short cells (nearest misses) ───────────────────────────────────
def test_top_up_fills_shortfall_with_nearest_misses():
    """Filler is the smallest-MFPT nucleosomes ABOVE the cap, ascending."""
    df = _summary(50)                      # cap=10 -> subids 0..9 reachable
    already, _ = sri.select_ids(df, cap=10.0, n=15, seed=1)
    assert len(already) == 10              # short by 5
    filler = sri.top_up_ids(df, cap=10.0, already=already, n=15)
    assert filler == [10, 11, 12, 13, 14]


def test_top_up_returns_empty_when_cell_is_already_full():
    df = _summary(50)
    already, _ = sri.select_ids(df, cap=1e9, n=20, seed=1)
    assert len(already) == 20
    assert sri.top_up_ids(df, cap=1e9, already=already, n=20) == []


def test_top_up_never_repeats_an_already_selected_id():
    df = _summary(50)
    already = list(range(10)) + [12]       # 12 already taken
    filler = sri.top_up_ids(df, cap=10.0, already=already, n=14)
    assert 12 not in filler
    assert filler == [10, 11, 13]          # skips 12, still nearest-first


def test_top_up_excludes_non_ok_and_nonfinite():
    df = _summary(50)
    df.loc[df.subid == 10, "mfpt_flag"] = "underflowed"
    df.loc[df.subid == 11, "mfpt"] = np.inf
    filler = sri.top_up_ids(df, cap=10.0, already=list(range(10)), n=12)
    assert filler == [12, 13]              # 10 and 11 are not usable


def test_top_up_is_capped_by_pool_size():
    """A pool smaller than the shortfall yields what exists, not an error."""
    df = _summary(12)                      # only subids 10, 11 sit above cap=10
    filler = sri.top_up_ids(df, cap=10.0, already=list(range(10)), n=50)
    assert filler == [10, 11]


def test_top_up_breaks_mfpt_ties_by_subid():
    """Deterministic without a seed: equal MFPTs resolve in subid order."""
    df = pd.DataFrame({
        "id": [f"peak_{i}" for i in range(4)],
        "subid": [3, 1, 2, 0],
        "mfpt": [99.0, 99.0, 99.0, 1.0],   # three-way tie above cap
        "mfpt_flag": ["ok"] * 4,
    })
    filler = sri.top_up_ids(df, cap=10.0, already=[0], n=3)
    assert filler == [1, 2]


# ── Markov cell lookup ───────────────────────────────────────────────────────
def test_missing_markov_dataset_dir_is_a_hard_error(tmp_path):
    """An absent dataset dir means the whole cell grid resolves to MISSING.

    That used to emit a manifest full of -1 while leaving stale id-lists in
    place, so the failure had to be read out of the log. Fail loudly instead.
    """
    (tmp_path / "some_other_dataset").mkdir()
    with pytest.raises(SystemExit, match="no Markov dataset directory"):
        sri._require_dataset_dir(tmp_path, "ret_all_Eout_{E_out:.2f}")


def test_present_markov_dataset_dir_passes(tmp_path):
    (tmp_path / "ret_all_Eout_6.70_Ein10.50").mkdir()
    sri._require_dataset_dir(tmp_path, "ret_all_Eout_6.70_Ein10.50")  # no raise
