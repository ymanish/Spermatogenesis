import math
import numpy as np
import pytest

from src.core.gillespie_event_simulator import (
    GillespieEventSimulator,
    ReplicateResult,
)
from src.core.nucleosomes import Nucleosome
from src.core.protamine import protamines
from src.config.custom_type import ReactionType


def _make_nuc(binding_sites: int = 14, k_wrap: float = 1.0) -> Nucleosome:
    # Use a triangular G_mat that strongly favors closed states near the center.
    # G_mat[i,j] = abs((binding_sites - 1) - 2 * ((i + j) / 2)) * 0.1
    L = binding_sites
    G = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(i, L):
            G[i, j] = 0.1 * (j - i)
    return Nucleosome(
        nuc_id="test", subid=0, sequence=None, G_mat=G,
        k_wrap=k_wrap, kT=1.0, binding_sites=binding_sites,
    )


def _make_prot(k_bind=1.0, k_unbind=89.7, p_conc=0.0, cooperativity=0.0):
    return protamines(
        k_unbind=k_unbind, k_bind=k_bind,
        p_conc=p_conc, cooperativity=cooperativity,
    )


def test_returns_replicate_result_dataclass():
    nuc = _make_nuc()
    prot = _make_prot()
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1.0, inf_protamine=True, seed=42,
    )
    result = sim.run()
    assert isinstance(result, ReplicateResult)


def test_censored_when_tau_max_tiny():
    """tau_max so small that no unwrapping event can complete: must censor."""
    nuc = _make_nuc()
    prot = _make_prot(p_conc=0.0)  # no binding to keep total rate small/predictable
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1e-12, inf_protamine=True, seed=1,
    )
    result = sim.run()
    assert result.censored is True
    assert math.isnan(result.detach_tau)
    assert result.final_tau == pytest.approx(1e-12)


def test_detaches_when_tau_max_huge():
    """tau_max large enough that detachment is essentially guaranteed."""
    nuc = _make_nuc(binding_sites=3)  # small system, fast detachment
    prot = _make_prot(p_conc=0.0)
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1e8, inf_protamine=True, seed=7,
    )
    result = sim.run()
    assert result.censored is False
    assert not math.isnan(result.detach_tau)
    assert result.detach_tau <= 1e8
    assert result.final_tau == pytest.approx(result.detach_tau)


def test_trajectory_endpoints_present():
    """Trajectory must always contain initial (tau=0, n_closed=binding_sites)
    and terminal points."""
    nuc = _make_nuc(binding_sites=4)
    prot = _make_prot(p_conc=0.0)
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1e6, inf_protamine=True, seed=3,
    )
    result = sim.run()
    assert result.traj_tau[0] == 0.0
    assert result.traj_n_closed[0] == 4
    assert result.traj_tau[-1] == pytest.approx(result.final_tau)
    assert len(result.traj_tau) == len(result.traj_n_closed)


def test_trajectory_only_records_n_closed_changes():
    """No two consecutive n_closed values may be equal (each row reflects a
    change). Allow at most one equal pair (the terminal duplicate when the
    last event before the boundary was a bind/unbind that did not record).

    Uses binding_sites=14 because protamines.protein_unbinding_coop has a
    hardcoded boundary at i==13, so binding_sites!=14 hits an upstream bug.
    """
    nuc = _make_nuc(binding_sites=14)
    prot = _make_prot(p_conc=10.0, k_bind=1.0, k_unbind=10.0)
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1e6, inf_protamine=True, seed=11,
    )
    result = sim.run()
    diffs = np.diff(result.traj_n_closed)
    # Every consecutive pair must differ except the terminal duplicate is
    # allowed only if the last event was non-n_closed-changing (binding/unbinding)
    # AND we're at boundary. Verify there are at most a handful of zeros.
    n_equal_pairs = int((diffs == 0).sum())
    assert n_equal_pairs <= 1


def test_time_weighted_mean_n_open_in_range():
    nuc = _make_nuc(binding_sites=5)
    prot = _make_prot(p_conc=0.0)
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1e6, inf_protamine=True, seed=4,
    )
    result = sim.run()
    assert 0.0 <= result.mean_n_open <= 5.0


def test_n_events_counts_nonzero_total():
    nuc = _make_nuc(binding_sites=4)
    prot = _make_prot(p_conc=0.0)
    sim = GillespieEventSimulator(
        nuc=nuc, prot=prot, tau_max=1e6, inf_protamine=True, seed=8,
    )
    result = sim.run()
    total = sum(result.n_events_by_type.values())
    assert total > 0


def test_deterministic_with_same_seed():
    """Two runs with identical seed produce identical results."""
    def go(seed):
        nuc = _make_nuc(binding_sites=4)
        prot = _make_prot(p_conc=0.0)
        return GillespieEventSimulator(
            nuc=nuc, prot=prot, tau_max=1e6, inf_protamine=True, seed=seed,
        ).run()

    a = go(123)
    b = go(123)
    assert a.detach_tau == b.detach_tau or (math.isnan(a.detach_tau) and math.isnan(b.detach_tau))
    assert a.censored == b.censored
    np.testing.assert_array_equal(a.traj_tau, b.traj_tau)
    np.testing.assert_array_equal(a.traj_n_closed, b.traj_n_closed)


# ─── run_single_replicate ──────────────────────────────────────────────

from src.gillespie_event.replicate import run_single_replicate


def test_run_single_replicate_returns_replicate_result():
    nuc = _make_nuc(binding_sites=4)
    prot_params = {
        "k_unbind": 89.7, "k_bind": 1.0,
        "p_conc": 0.0, "cooperativity": 0.0,
    }
    result = run_single_replicate(
        nuc=nuc, replicate_num=0, prot_params=prot_params,
        tau_max=1e6, inf_protamine=True,
    )
    assert isinstance(result, ReplicateResult)


def test_run_single_replicate_does_not_mutate_input_nuc():
    nuc = _make_nuc(binding_sites=4)
    state_before = nuc.state.copy()
    n_closed_before = nuc.n_closed
    prot_params = {
        "k_unbind": 89.7, "k_bind": 1.0,
        "p_conc": 0.0, "cooperativity": 0.0,
    }
    run_single_replicate(
        nuc=nuc, replicate_num=0, prot_params=prot_params,
        tau_max=1.0, inf_protamine=True,
    )
    np.testing.assert_array_equal(nuc.state, state_before)
    assert nuc.n_closed == n_closed_before


def test_run_single_replicate_deterministic_across_reps():
    """Different replicate_num must yield different results; same rep_num same nuc same result."""
    nuc = _make_nuc(binding_sites=4)
    prot_params = {
        "k_unbind": 89.7, "k_bind": 1.0,
        "p_conc": 0.0, "cooperativity": 0.0,
    }
    a = run_single_replicate(nuc=nuc, replicate_num=0,
                             prot_params=prot_params, tau_max=1e6,
                             inf_protamine=True)
    b = run_single_replicate(nuc=nuc, replicate_num=0,
                             prot_params=prot_params, tau_max=1e6,
                             inf_protamine=True)
    c = run_single_replicate(nuc=nuc, replicate_num=1,
                             prot_params=prot_params, tau_max=1e6,
                             inf_protamine=True)
    # Same rep_num: identical (or both NaN for detach_tau)
    if not (math.isnan(a.detach_tau) and math.isnan(b.detach_tau)):
        assert a.detach_tau == b.detach_tau
    # Different rep_num: trajectory should differ (overwhelmingly likely)
    assert (a.traj_tau.shape != c.traj_tau.shape
            or not np.array_equal(a.traj_tau, c.traj_tau))
