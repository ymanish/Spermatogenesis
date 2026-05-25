#!/usr/bin/env python3
"""CLI for the event-driven Gillespie pipeline.

Loads defaults from src/gillespie_event/gillespie_event_config.yaml and
overrides them with command-line flags. Processes a single SPRM dataset
per invocation.
"""

import os

if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *  # noqa: F401, F403

import argparse
import datetime as dt
import time
from pathlib import Path

import yaml

from src.gillespie_event.config import GillespieEventConfig
from src.gillespie_event.orchestrator import run_gillespie_event
from src.utils.logger_util import get_logger

_DEFAULT_CONFIG = Path(__file__).parent / "gillespie_event_config.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description="Event-driven Gillespie pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help="YAML config (CLI flags override its values).")
    p.add_argument("--dataset_dir", type=Path, default=None,
                   help="SPRM dataset directory.")
    p.add_argument("--storage_dir", type=Path, default=None,
                   help="Root output directory.")
    p.add_argument("--dataset", type=str, default=None,
                   help="Optional label prefix for output file names.")

    p.add_argument("--k_wrap", type=float, default=None)
    p.add_argument("--binding_sites", type=int, default=None)
    p.add_argument("--prot_k_unbind", type=float, default=None)
    p.add_argument("--prot_k_bind", type=float, default=None)
    p.add_argument("--prot_p_conc", type=float, default=None)
    p.add_argument("--prot_cooperativity", type=float, default=None)

    p.add_argument("--tau_max", type=float, default=None,
                   help="Censoring boundary in dimensionless time.")
    p.add_argument("--n_survival_points", type=int, default=None,
                   help="Resolution of empirical S(tau) grid.")

    p.add_argument("--inf_protamine", action="store_true", default=None)
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_workers", type=int, default=None)
    p.add_argument("--flush_every", type=int, default=None)
    p.add_argument("--save_trajectories", action="store_true", default=None)
    p.add_argument("--no_save_trajectories", dest="save_trajectories",
                   action="store_false", default=None,
                   help="Disable trajectory saving (overrides YAML).")

    p.add_argument("--max_nucs", type=int, default=None)
    return p.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve(args, cfg: dict) -> dict:
    def pick(cli_val, key, fallback=None):
        if cli_val is not None:
            return cli_val
        v = cfg.get(key)
        return v if v is not None else fallback

    def pick_bool(cli_val, key, fallback: bool):
        cfg_val = cfg.get(key)
        if cli_val is True:
            return True
        if cli_val is False:
            return False
        if cfg_val is not None:
            return bool(cfg_val)
        return fallback

    def pick_path(cli_val, key):
        v = cli_val if cli_val is not None else cfg.get(key)
        return Path(v) if v is not None else None

    return {
        "dataset_dir":         pick_path(args.dataset_dir, "dataset_dir"),
        "storage_dir":         pick_path(args.storage_dir, "storage_dir"),
        "dataset":             pick(args.dataset, "dataset"),
        "k_wrap":              pick(args.k_wrap, "k_wrap", 1.0),
        "binding_sites":       pick(args.binding_sites, "binding_sites", 14),
        "prot_k_unbind":       pick(args.prot_k_unbind, "prot_k_unbind", 89.7),
        "prot_k_bind":         pick(args.prot_k_bind, "prot_k_bind", 1.0),
        "prot_p_conc":         pick(args.prot_p_conc, "prot_p_conc", 0.0),
        "prot_cooperativity":  pick(args.prot_cooperativity, "prot_cooperativity", 0.0),
        "tau_max":             pick(args.tau_max, "tau_max", 10000.0),
        "n_survival_points":   pick(args.n_survival_points, "n_survival_points", 1000),
        "inf_protamine":       pick_bool(args.inf_protamine, "inf_protamine", True),
        "replicates":          pick(args.replicates, "replicates", 20),
        "batch_size":          pick(args.batch_size, "batch_size", 10),
        "n_workers":           pick(args.n_workers, "n_workers", 4),
        "flush_every":         pick(args.flush_every, "flush_every", 10000),
        "save_trajectories":   pick_bool(args.save_trajectories, "save_trajectories", True),
        "max_nucs":            pick(args.max_nucs, "max_nucs"),
    }


def main():
    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level="INFO")

    # Worker scratch dir
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)

    args = parse_args()
    cfg = _load_yaml(args.config)
    logger.info(f"Loaded config from: {args.config}")

    p = _resolve(args, cfg)

    if p["dataset_dir"] is None:
        raise SystemExit("dataset_dir is required (set in YAML or via --dataset_dir).")
    if not p["dataset_dir"].exists():
        raise SystemExit(f"dataset_dir not found: {p['dataset_dir']}")
    if p["storage_dir"] is None:
        raise SystemExit("storage_dir is required.")
    p["storage_dir"].mkdir(parents=True, exist_ok=True)

    config = GillespieEventConfig(
        k_wrap=p["k_wrap"],
        binding_sites=p["binding_sites"],
        prot_k_unbind=p["prot_k_unbind"],
        prot_k_bind=p["prot_k_bind"],
        prot_p_conc=p["prot_p_conc"],
        prot_cooperativity=p["prot_cooperativity"],
        tau_max=p["tau_max"],
        n_survival_points=p["n_survival_points"],
        inf_protamine=p["inf_protamine"],
        replicates=p["replicates"],
        batch_size=p["batch_size"],
        n_workers=p["n_workers"],
        flush_every=p["flush_every"],
        save_trajectories=p["save_trajectories"],
    )

    run_gillespie_event(
        dataset_dir=p["dataset_dir"],
        storage_dir=p["storage_dir"],
        config=config,
        dataset_label=p["dataset"],
        max_nucs=p["max_nucs"],
        logger=logger,
    )

    elapsed = time.perf_counter() - start
    logger.info(f"Total time: {dt.timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    main()
