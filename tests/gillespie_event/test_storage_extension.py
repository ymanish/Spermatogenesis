from src.config.storage import SimulationStorage


def test_get_output_paths_returns_survival_key(tmp_path):
    storage = SimulationStorage(base_dir=tmp_path, use_index=False)
    params = {
        "k_wrap": 1.0,
        "prot_params": {"k_unbind": 89.7, "k_bind": 1.0,
                        "p_conc": 100.0, "cooperativity": 4.5},
        "binding_sites": 14,
        "tau_max": 10000.0,
        "inf_protamine": True,
    }
    paths = storage.get_output_paths(params, "demo")
    assert "trajectory" in paths
    assert "summary" in paths
    assert "survival" in paths
    assert paths["survival"].name == "demo.parquet"
    assert paths["survival"].parent.name == "survival"
    assert paths["trajectory"].parent.name == "trajectories"
    assert paths["summary"].parent.name == "summaries"
