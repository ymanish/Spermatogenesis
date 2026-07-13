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
