import numpy as np
import pytest

from src.gillespie_event.config import GillespieEventConfig


def test_defaults_are_set():
    cfg = GillespieEventConfig()
    assert cfg.k_wrap == 1.0
    assert cfg.binding_sites == 14
    assert cfg.prot_k_unbind == 89.7
    assert cfg.tau_max == 10000.0
    assert cfg.n_survival_points == 1000
    assert cfg.inf_protamine is True
    assert cfg.replicates == 20
    assert cfg.batch_size == 10
    assert cfg.n_workers == 4
    assert cfg.save_trajectories is True


def test_tau_grid_is_linear():
    cfg = GillespieEventConfig(tau_max=100.0, n_survival_points=11)
    grid = cfg.tau_grid
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (11,)
    assert grid[0] == 0.0
    assert grid[-1] == 100.0


def test_prot_params_dict():
    cfg = GillespieEventConfig(
        prot_k_unbind=1.0, prot_k_bind=2.0, prot_p_conc=3.0, prot_cooperativity=4.0
    )
    assert cfg.prot_params == {
        "k_unbind": 1.0, "k_bind": 2.0, "p_conc": 3.0, "cooperativity": 4.0
    }


@pytest.mark.parametrize("kwargs", [
    {"tau_max": 0.0},
    {"tau_max": -1.0},
    {"n_survival_points": 1},
    {"replicates": 0},
    {"batch_size": 0},
    {"n_workers": 0},
])
def test_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        GillespieEventConfig(**kwargs)


def test_to_dict_round_trips():
    cfg = GillespieEventConfig(k_wrap=2.0, prot_p_conc=50.0, replicates=5)
    restored = GillespieEventConfig.from_dict(cfg.to_dict())
    assert restored.k_wrap == 2.0
    assert restored.prot_p_conc == 50.0
    assert restored.replicates == 5
