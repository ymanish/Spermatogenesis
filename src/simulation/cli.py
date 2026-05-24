#!/usr/bin/env python3
"""
CLI: Simulation Launcher
========================

Runs Gillespie simulations for nucleosome populations in parallel.

Parameters are loaded from a YAML config file and can be overridden by CLI arguments.
The default config file lives alongside this script:
    src/simulation/simulation_config.yaml

Usage — local (edit the YAML, then just run):
    python -m src.simulation.cli

Usage — cluster (override specific params):
    python -m src.simulation.cli \\
        --dataset_dir SPRM_data/ret_single_nuc \\
        --storage_dir /group/.../output \\
        --tau_stop 10000 --tau_num 1000 \\
        --n_workers 20 --replicates 50
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import sys
from pathlib import Path

import argparse
import time
import datetime as dt

import yaml
import numpy as np

from src.utils.logger_util import get_logger
from src.config.storage import SimulationStorage
from src.config.custom_type import SimulationConfig
from .orchestrator import run_simulation

# Default config file next to this script
_DEFAULT_CONFIG = Path(__file__).parent / "simulation_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nucleosome Gillespie simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="YAML config file. CLI args override values in this file."
    )

    # Input (mutually exclusive)
    parser.add_argument("--dataset_dir", type=Path, default=None,
                        help="SPRM dataset directory (energies.tsv + id_lookup.tsv).")
    parser.add_argument("--infile", type=Path, default=None,
                        help="Old-format nucleosome TSV file.")

    # Output
    parser.add_argument("--storage_dir", type=Path, default=None,
                        help="Root directory for simulation output.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Label prefix for output file names.")

    # Nucleosome parameters
    parser.add_argument("--k_wrap", type=float, default=None)
    parser.add_argument("--binding_sites", type=int, default=None)

    # Protamine parameters
    parser.add_argument("--prot_k_unbind", type=float, default=None)
    parser.add_argument("--prot_k_bind", type=float, default=None)
    parser.add_argument("--prot_p_conc", type=float, default=None)
    parser.add_argument("--prot_cooperativity", type=float, default=None)

    # Time parameters
    parser.add_argument("--tau_stop", type=float, default=None,
                        help="Dimensionless end time tau_max.")
    parser.add_argument("--tau_num", type=int, default=None,
                        help="Number of tau sample points.")

    # Simulation behaviour
    parser.add_argument("--inf_protamine", action="store_true", default=None)
    parser.add_argument("--renucleation", action="store_true", default=None)
    parser.add_argument("--replicates", type=int, default=None)

    # Execution
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--flush_every", type=int, default=None)

    # Trajectories
    parser.add_argument("--save_trajectories", action="store_true", default=None)
    parser.add_argument("--maxpoints_saved_trajectories", type=int, default=None)

    # Testing
    parser.add_argument("--max_nucs", type=int, default=None)
    parser.add_argument("--subids_start", type=int, default=None)
    parser.add_argument("--subids_end", type=int, default=None)

    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_params(args, cfg: dict) -> dict:
    """
    Merge CLI args and YAML config.  CLI values (not None) take priority.
    Falls back to hardcoded defaults when both are absent.
    """
    def pick(cli_val, key, fallback=None):
        if cli_val is not None:
            return cli_val
        v = cfg.get(key)
        return v if v is not None else fallback

    # action="store_true" args return False (not None) when absent — treat False as "not set"
    def pick_bool(cli_val, key, fallback: bool):
        cfg_val = cfg.get(key)
        if cli_val:          # explicitly passed on CLI
            return True
        if cfg_val is not None:
            return bool(cfg_val)
        return fallback

    def pick_path(cli_val, key):
        v = cli_val if cli_val is not None else cfg.get(key)
        return Path(v) if v is not None else None

    return {
        'dataset_dir':                   pick_path(args.dataset_dir,  'dataset_dir'),
        'infile':                        pick_path(args.infile,        'infile'),
        'storage_dir':                   pick_path(args.storage_dir,   'storage_dir'),
        'dataset':                       pick(args.dataset,            'dataset'),
        'k_wrap':                        pick(args.k_wrap,             'k_wrap',             1.0),
        'binding_sites':                 pick(args.binding_sites,      'binding_sites',      14),
        'prot_k_unbind':                 pick(args.prot_k_unbind,      'prot_k_unbind',      89.7),
        'prot_k_bind':                   pick(args.prot_k_bind,        'prot_k_bind',        1.0),
        'prot_p_conc':                   pick(args.prot_p_conc,        'prot_p_conc',        0.0),
        'prot_cooperativity':            pick(args.prot_cooperativity, 'prot_cooperativity', 0.0),
        'tau_stop':                      pick(args.tau_stop,           'tau_stop',           10000.0),
        'tau_num':                       pick(args.tau_num,            'tau_num',            1000),
        'inf_protamine':                 pick_bool(args.inf_protamine,   'inf_protamine',   False),
        'renucleation':                  pick_bool(args.renucleation,    'renucleation',    False),
        'replicates':                    pick(args.replicates,         'replicates',         20),
        'batch_size':                    pick(args.batch_size,         'batch_size',         50),
        'n_workers':                     pick(args.n_workers,          'n_workers',          4),
        'flush_every':                   pick(args.flush_every,        'flush_every',        10000),
        'save_trajectories':             pick_bool(args.save_trajectories, 'save_trajectories', False),
        'maxpoints_saved_trajectories':  pick(args.maxpoints_saved_trajectories,
                                              'maxpoints_saved_trajectories', 100),
        'max_nucs':                      pick(args.max_nucs,           'max_nucs'),
        'subids_start':                  pick(args.subids_start,       'subids_start'),
        'subids_end':                    pick(args.subids_end,         'subids_end'),
    }


def main():
    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level='INFO')

    # Temp directory for worker scratch files
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)

    args = parse_args()
    cfg = _load_yaml(args.config)
    logger.info(f"Loaded config from: {args.config}")

    p = resolve_params(args, cfg)

    # ── Validate input ────────────────────────────────────────────────────────
    if p['dataset_dir'] is None and p['infile'] is None:
        raise ValueError("Provide dataset_dir or infile in the config file or via CLI.")
    if p['dataset_dir'] is not None and p['infile'] is not None:
        raise ValueError("dataset_dir and infile are mutually exclusive.")
    if p['infile'] is not None and not p['infile'].exists():
        raise FileNotFoundError(f"Input file not found: {p['infile']}")
    if p['dataset_dir'] is not None and not p['dataset_dir'].exists():
        raise FileNotFoundError(f"Dataset directory not found: {p['dataset_dir']}")
    if p['storage_dir'] is None:
        raise ValueError("storage_dir is required (set in YAML or via --storage_dir).")

    p['storage_dir'].mkdir(parents=True, exist_ok=True)

    # ── Validate trajectory settings ──────────────────────────────────────────
    if p['save_trajectories'] and p['maxpoints_saved_trajectories'] > p['tau_num']:
        raise ValueError(
            f"maxpoints_saved_trajectories ({p['maxpoints_saved_trajectories']}) "
            f"cannot exceed tau_num ({p['tau_num']})."
        )

    # ── Storage paths ─────────────────────────────────────────────────────────
    storage = SimulationStorage(base_dir=p['storage_dir'], use_index=False)

    params_for_storage = {
        'k_wrap':        p['k_wrap'],
        'prot_params':   {
            'k_unbind':      p['prot_k_unbind'],
            'k_bind':        p['prot_k_bind'],
            'p_conc':        p['prot_p_conc'],
            'cooperativity': p['prot_cooperativity'],
        },
        'binding_sites': p['binding_sites'],
        'tau_max':       p['tau_stop'],
        'tau_steps':     p['tau_num'],
        'inf_protamine': p['inf_protamine'],
        'replicates':    p['replicates'],
    }

    if p['dataset_dir'] is not None:
        file_id = p['dataset_dir'].name
    else:
        file_id = p['infile'].stem
    if p['dataset']:
        file_id = f"{p['dataset']}_{file_id}"

    output_paths = storage.get_output_paths(params_for_storage, file_id)
    traj_outfile = output_paths['trajectory']
    tsv_outfile  = output_paths['summary']

    # ── Build SimulationConfig ────────────────────────────────────────────────
    config = SimulationConfig(
        k_wrap=p['k_wrap'],
        binding_sites=p['binding_sites'],
        prot_k_unbind=p['prot_k_unbind'],
        prot_k_bind=p['prot_k_bind'],
        prot_p_conc=p['prot_p_conc'],
        prot_cooperativity=p['prot_cooperativity'],
        tau_max=p['tau_stop'],
        tau_steps=p['tau_num'],
        inf_protamine=p['inf_protamine'],
        renucleation=p['renucleation'],
        replicates=p['replicates'],
        batch_size=p['batch_size'],
        n_workers=p['n_workers'],
        flush_every=p['flush_every'],
        save_trajectories=p['save_trajectories'],
        maxpoints_saved_trajectories=p['maxpoints_saved_trajectories'],
    )
    logger.info(f"Configuration: {config}")

    subids_range = (
        (p['subids_start'], p['subids_end'])
        if p['subids_start'] is not None and p['subids_end'] is not None
        else None
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    run_simulation(
        traj_outfile=traj_outfile,
        tsv_outfile=tsv_outfile,
        config=config,
        file_path=p['infile'],
        dataset_dir=p['dataset_dir'],
        logger=logger,
        max_nucs=p['max_nucs'],
        subids_range=subids_range,
    )

    elapsed = time.perf_counter() - start
    logger.info(f"Total execution time: {dt.timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    main()
