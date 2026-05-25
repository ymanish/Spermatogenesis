import math
import numpy as np
import pytest

from src.config.custom_type import ReactionType
from src.core.gillespie_event_simulator import ReplicateResult
from src.core.nucleosomes import Nucleosome
from src.gillespie_event.aggregate import (
    NucleosomeAggregate,
    aggregate_replicates,
)


def _make_nuc():
    L = 14
    G = np.zeros((L, L), dtype=float)
    return Nucleosome(nuc_id="agg_test", subid=2, sequence=None, G_mat=G,
                      k_wrap=1.0, kT=1.0, binding_sites=L)


def _mk_result(detach_tau, censored, final_tau, mean_n_open=3.0, mean_bprot=2.0):
    return ReplicateResult(
        detach_tau=detach_tau,
        censored=censored,
        final_tau=final_tau,
        mean_n_open=mean_n_open,
        mean_bprot=mean_bprot,
        n_events_by_type={rt: 0 for rt in ReactionType},
        traj_tau=np.array([0.0, final_tau]),
        traj_n_closed=np.array([14, 0 if not censored else 14], dtype=np.uint8),
    )


def test_aggregate_basic_fields():
    nuc = _make_nuc()
    results = [
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=20.0, censored=False, final_tau=20.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
    ]
    tau_grid = np.linspace(0.0, 100.0, 101)
    agg = aggregate_replicates(nuc, results, tau_grid)

    assert agg.id == "agg_test"
    assert agg.subid == 2
    assert agg.n_replicates == 3
    assert agg.censored_fraction == pytest.approx(1 / 3)
    assert agg.mfpt_uncensored == pytest.approx(15.0)


def test_aggregate_all_censored_gives_nan_mfpt():
    nuc = _make_nuc()
    results = [
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
    ]
    tau_grid = np.linspace(0.0, 100.0, 101)
    agg = aggregate_replicates(nuc, results, tau_grid)
    assert math.isnan(agg.mfpt_uncensored)
    assert agg.censored_fraction == 1.0
    assert agg.final_survival == 1.0


def test_aggregate_survival_step_function():
    """Empirical S(tau): S(t) = (# results with effective lifetime > t) / N."""
    nuc = _make_nuc()
    results = [
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=30.0, censored=False, final_tau=30.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
    ]
    tau_grid = np.array([0.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0])
    agg = aggregate_replicates(nuc, results, tau_grid)
    # At tau=0: all 3 alive  -> 1.0
    # At tau=5: all 3 alive  -> 1.0
    # At tau=10: replicate detached at 10.0 is dead at tau=10 (lifetime > tau is False)
    #            -> 2 alive -> 2/3
    # At tau=15: -> 2/3
    # At tau=30: replicate detached at 30.0 dies -> 1/3
    # At tau=50: -> 1/3
    # At tau=100: censored lifetime=+inf > 100 -> 1/3
    expected = np.array([1.0, 1.0, 2/3, 2/3, 1/3, 1/3, 1/3])
    np.testing.assert_allclose(agg.survival, expected, atol=1e-12)


def test_aggregate_rmst_matches_trapz_of_survival():
    nuc = _make_nuc()
    results = [
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
    ]
    tau_grid = np.linspace(0.0, 100.0, 1001)
    agg = aggregate_replicates(nuc, results, tau_grid)
    assert agg.rmst == pytest.approx(np.trapz(agg.survival, tau_grid))


def test_aggregate_half_life_when_crossed():
    nuc = _make_nuc()
    # 4 replicates: 3 detach at tau=10, 1 at tau=100
    results = [
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=100.0, censored=False, final_tau=100.0),
    ]
    tau_grid = np.linspace(0.0, 200.0, 2001)
    agg = aggregate_replicates(nuc, results, tau_grid)
    # S drops from 1.0 to 0.25 at tau=10.  Half-life = smallest tau with S<=0.5
    assert agg.half_life == pytest.approx(10.0, abs=0.2)


def test_aggregate_half_life_nan_when_never_crossed():
    nuc = _make_nuc()
    # All censored: S never drops below 1.0
    results = [
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
    ]
    tau_grid = np.linspace(0.0, 100.0, 101)
    agg = aggregate_replicates(nuc, results, tau_grid)
    assert math.isnan(agg.half_life)


def test_aggregate_mom_and_tw_means_differ_with_unequal_lifetimes():
    """Mean-of-means and time-weighted differ when lifetimes are unequal."""
    nuc = _make_nuc()
    results = [
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0, mean_n_open=8.0),
        _mk_result(detach_tau=100.0, censored=False, final_tau=100.0, mean_n_open=3.0),
    ]
    tau_grid = np.linspace(0.0, 100.0, 101)
    agg = aggregate_replicates(nuc, results, tau_grid)
    assert agg.mean_n_open_mom == pytest.approx((8.0 + 3.0) / 2)
    assert agg.mean_n_open_tw == pytest.approx(
        (8.0 * 10.0 + 3.0 * 100.0) / (10.0 + 100.0)
    )
    assert agg.mean_n_open_mom != pytest.approx(agg.mean_n_open_tw, rel=1e-3)


def test_aggregate_final_survival_equals_censored_fraction():
    nuc = _make_nuc()
    results = [
        _mk_result(detach_tau=10.0, censored=False, final_tau=10.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
        _mk_result(detach_tau=math.nan, censored=True, final_tau=100.0),
    ]
    tau_grid = np.linspace(0.0, 100.0, 101)
    agg = aggregate_replicates(nuc, results, tau_grid)
    assert agg.final_survival == agg.censored_fraction
