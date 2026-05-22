"""
Command-Line Interface Module
==============================

CLI for running Markov solver from command line.

Parameters are loaded from a YAML config file and can be overridden by CLI arguments.
The default config file lives alongside this script:
    src/markov_execution/markov_config.yaml

Usage — local (edit the YAML, then just run):
    python -m src.markov_execution.cli

Usage — cluster (override specific params):
    python -m src.markov_execution.cli \\
        --dataset_dir SPRM_data/ret_all_stable147_refined \\
        --storage_dir /group/.../output \\
        --prot_p_conc 100.0 --prot_cooperativity 4.5 \\
        --n_workers 20

Author: MY
Date: 2025-12-11
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import argparse
import time
import datetime as dt
from pathlib import Path

import yaml

from src.utils.logger_util import get_logger
from .config import MarkovConfig
from .orchestrator import run_markov_solver
from .storage import MarkovStorage


# Default config file next to this script
_DEFAULT_CONFIG = Path(__file__).parent / "markov_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Markov solver for nucleosome unwrapping dynamics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="YAML config file. CLI args override values in this file."
    )

    # Input (mutually exclusive — validated after merge)
    parser.add_argument("--dataset_dir", type=Path, default=None,
                        help="SPRM dataset directory (energies.tsv + id_lookup.tsv).")
    parser.add_argument("--infile", type=Path, default=None,
                        help="Old-format nucleosome TSV file.")

    # Output
    parser.add_argument("--storage_dir", type=Path, default=None,
                        help="Root directory for Markov output.")
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

    # Computation parameters
    parser.add_argument("--tau_max", type=float, default=None,
                        help="Maximum dimensionless time tau.")
    parser.add_argument("--tau_steps", type=int, default=None,
                        help="Number of tau sample points.")
    parser.add_argument("--method", type=str, choices=["expm", "ode"], default=None,
                        help="Solver method.")
    parser.add_argument("--sparse", action="store_true", default=None,
                        help="Use sparse matrices.")

    # Execution
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--max_nucs", type=int, default=None,
                        help="Limit number of nucleosomes processed (testing).")

    # Output toggles
    parser.add_argument("--save_survival", action="store_true", default=None)
    parser.add_argument("--save_states", action="store_true", default=None)
    parser.add_argument("--save_mfpt", action="store_true", default=None)

    # Testing / debugging
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
    Merge CLI args and YAML config. CLI values (not None) take priority.
    Falls back to hardcoded defaults when both are absent.

    Unknown YAML keys are silently ignored — only fields listed below are read.
    """
    def pick(cli_val, key, fallback=None):
        if cli_val is not None:
            return cli_val
        v = cfg.get(key)
        return v if v is not None else fallback

    def pick_bool(cli_val, key, fallback: bool):
        cfg_val = cfg.get(key)
        if cli_val:            # explicitly passed on CLI
            return True
        if cfg_val is not None:
            return bool(cfg_val)
        return fallback

    def pick_path(cli_val, key):
        v = cli_val if cli_val is not None else cfg.get(key)
        return Path(v) if v is not None else None

    return {
        'dataset_dir':        pick_path(args.dataset_dir,        'dataset_dir'),
        'infile':             pick_path(args.infile,             'infile'),
        'storage_dir':        pick_path(args.storage_dir,        'storage_dir'),
        'dataset':            pick(args.dataset,                 'dataset'),
        'k_wrap':             pick(args.k_wrap,                  'k_wrap',             1.0),
        'binding_sites':      pick(args.binding_sites,           'binding_sites',      14),
        'prot_k_unbind':      pick(args.prot_k_unbind,           'prot_k_unbind',      89.7),
        'prot_k_bind':        pick(args.prot_k_bind,             'prot_k_bind',        1.0),
        'prot_p_conc':        pick(args.prot_p_conc,             'prot_p_conc',        0.0),
        'prot_cooperativity': pick(args.prot_cooperativity,      'prot_cooperativity', 0.0),
        'tau_max':            pick(args.tau_max,                 'tau_max',            1000.0),
        'tau_steps':          pick(args.tau_steps,               'tau_steps',          500),
        'method':             pick(args.method,                  'method',             'expm'),
        'sparse':             pick_bool(args.sparse,             'sparse',             False),
        'batch_size':         pick(args.batch_size,              'batch_size',         10),
        'n_workers':          pick(args.n_workers,               'n_workers',          10),
        'max_nucs':           pick(args.max_nucs,                'max_nucs'),
        'save_survival':      pick_bool(args.save_survival,      'save_survival',      True),
        'save_states':        pick_bool(args.save_states,        'save_states',        False),
        'save_mfpt':          pick_bool(args.save_mfpt,          'save_mfpt',          True),
        'subids_start':       pick(args.subids_start,            'subids_start'),
        'subids_end':         pick(args.subids_end,              'subids_end'),
    }


def main():
    start = time.perf_counter()
    logger = get_logger(__name__, log_file=None, level='INFO')

    # Temp directory for worker scratch files
    tmp_dir = Path(__file__).parent.parent.parent / "temps"
    tmp_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    logger.info(f"Using temporary directory: {tmp_dir}")

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

    # ── Build MarkovConfig ────────────────────────────────────────────────────
    config = MarkovConfig(
        k_wrap=p['k_wrap'],
        binding_sites=p['binding_sites'],
        prot_k_bind=p['prot_k_bind'],
        prot_k_unbind=p['prot_k_unbind'],
        prot_p_conc=p['prot_p_conc'],
        prot_cooperativity=p['prot_cooperativity'],
        tau_max=p['tau_max'],
        tau_steps=p['tau_steps'],
        method=p['method'],
        sparse=p['sparse'],
        batch_size=p['batch_size'],
        n_workers=p['n_workers'],
        max_nucs=p['max_nucs'],
        save_survival=p['save_survival'],
        save_states=p['save_states'],
        save_mfpt=p['save_mfpt'],
    )

    # ── Storage paths ─────────────────────────────────────────────────────────
    # Indexing maintains a CSV file which can be problematic in array jobs.
    # Use index=False and rebuild from MarkovStorage.rebuild_index() after jobs finish.
    storage = MarkovStorage(base_dir=p['storage_dir'], use_index=False)

    params_for_storage = {
        'k_wrap': config.k_wrap,
        'prot_params': {
            'k_unbind':      config.prot_k_unbind,
            'k_bind':        config.prot_k_bind,
            'p_conc':        config.prot_p_conc,
            'cooperativity': config.prot_cooperativity,
        },
        'binding_sites': config.binding_sites,
        'tau_max':       config.tau_max,
        'tau_steps':     config.tau_steps,
        'method':        config.method,
        'sparse':        config.sparse,
        'dimensionless': config.dimensionless,
    }

    if p['dataset_dir'] is not None:
        file_id = p['dataset_dir'].name
        input_label = str(p['dataset_dir'])
    else:
        file_id = p['infile'].stem
        input_label = str(p['infile'])
    if p['dataset']:
        file_id = f"{p['dataset']}_{file_id}"

    logger.info(f"Running Markov solver on: {input_label} (file_id={file_id})")

    output_paths = storage.get_output_paths(params_for_storage, file_id)
    tsv_outfile = output_paths['summary']
    survival_outfile = output_paths['survivals']

    logger.info(f"Configuration: {config}")

    subids_range = (
        (p['subids_start'], p['subids_end'])
        if p['subids_start'] is not None and p['subids_end'] is not None
        else None
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    run_markov_solver(
        tsv_outfile=tsv_outfile,
        survival_outfile=survival_outfile,
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
