#!/usr/bin/env python3
"""
CLI: Markov Solver Launcher
============================

Runs Markov chain solver for nucleosome populations in parallel.

Parameters are loaded from a YAML config file and can be overridden by CLI arguments.
The default config file lives alongside this script:
    src/markov_execution/markov_config.yaml

Usage — local (edit the YAML, then just run):
    python -m src.markov_execution.cli

Usage — cluster (override specific params):
    python -m src.markov_execution.cli \\
        --dataset_dir SPRM_data/ret_single_nuc \\
        --storage_dir output/markov \\
        --tau_max 1000 --tau_steps 500 \\
        --n_workers 20
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import argparse
import time
import datetime as dt
from pathlib import Path

import yaml
import numpy as np

from src.utils.logger_util import get_logger
from src.core.build_nucleosomes import nucleosome_generator, nucleosome_generator_sprm
from src.analysis.markov_solver.generator import (
    build_full_Q_from_nucleosome,
    matrix_density,
)
from src.analysis.markov_solver.tnp2 import TNP2Config
from .config import MarkovConfig
from .orchestrator import run_markov_solver
from .storage import MarkovStorage

# Default config file next to this script
_DEFAULT_CONFIG = Path(__file__).parent / "markov_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nucleosome Markov chain solver.",
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
                        help="Root directory for output.")
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
    parser.add_argument("--eads_delta", type=float, default=None)
    parser.add_argument("--eads_weight_mode", type=str, default=None,
                        choices=["none", "uniform", "outer8", "inner6"])
    parser.add_argument("--eads_apply", action="store_true", default=None)

    # Computation parameters
    parser.add_argument("--tau_max", type=float, default=None)
    parser.add_argument("--tau_steps", type=int, default=None)
    parser.add_argument("--method", type=str, choices=["expm", "ode"], default=None)
    parser.add_argument("--sparse", action="store_true", default=None)

    # Execution
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=None)

    # Output options
    parser.add_argument("--save_survival", action="store_true", default=None)
    parser.add_argument("--save_states", action="store_true", default=None)
    parser.add_argument("--save_mfpt", action="store_true", default=None)

    # TNP2 v2.0 extension
    parser.add_argument("--tnp2_enabled", action="store_true", default=None,
                        help="Enable TNP2 J_eff layer (CpG-gated cooperativity disruption).")
    parser.add_argument("--tnp2_eps_cpg", type=float, default=None)
    parser.add_argument("--tnp2_mu_t0", type=float, default=None)
    parser.add_argument("--fasta_path", type=Path, default=None,
                        help="FASTA with 147-bp sequences keyed to id_lookup seq_id.")
    parser.add_argument("--fasta_id_style", type=str, default=None, choices=["name", "coord"])

    # Testing
    parser.add_argument("--max_nucs", type=int, default=None)
    parser.add_argument("--max_nucs_seed", type=int, default=None)
    parser.add_argument("--subids_start", type=int, default=None)
    parser.add_argument("--subids_end", type=int, default=None)
    parser.add_argument(
        "--check_matrix_density",
        action="store_true",
        default=None,
        help="Build Q for one nucleosome, print density information, and exit."
    )

    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_params(args, cfg: dict) -> dict:
    """Merge CLI args and YAML config. CLI values (not None) take priority."""
    def pick(cli_val, key, fallback=None):
        if cli_val is not None:
            return cli_val
        v = cfg.get(key)
        return v if v is not None else fallback

    def pick_bool(cli_val, key, fallback: bool):
        cfg_val = cfg.get(key)
        if cli_val:
            return True
        if cfg_val is not None:
            return bool(cfg_val)
        return fallback

    def pick_path(cli_val, key):
        v = cli_val if cli_val is not None else cfg.get(key)
        return Path(v) if v is not None else None

    return {
        'dataset_dir':       pick_path(args.dataset_dir,  'dataset_dir'),
        'infile':            pick_path(args.infile,        'infile'),
        'storage_dir':       pick_path(args.storage_dir,   'storage_dir'),
        'dataset':           pick(args.dataset,            'dataset'),
        'k_wrap':            pick(args.k_wrap,             'k_wrap',             1.0),
        'binding_sites':     pick(args.binding_sites,      'binding_sites',      14),
        'prot_k_unbind':     pick(args.prot_k_unbind,      'prot_k_unbind',      89.7),
        'prot_k_bind':       pick(args.prot_k_bind,        'prot_k_bind',        1.0),
        'prot_p_conc':       pick(args.prot_p_conc,        'prot_p_conc',        0.0),
        'prot_cooperativity':pick(args.prot_cooperativity, 'prot_cooperativity', 0.0),
        'eads_delta':        pick(args.eads_delta,         'eads_delta',         0.0),
        'eads_weight_mode':  pick(args.eads_weight_mode,   'eads_weight_mode',   'none'),
        'eads_apply':        pick_bool(args.eads_apply,    'eads_apply',         False),
        'tau_max':           pick(args.tau_max,            'tau_max',            1000.0),
        'tau_steps':         pick(args.tau_steps,          'tau_steps',          500),
        'method':            pick(args.method,             'method',             'expm'),
        'sparse':            pick_bool(args.sparse,         'sparse',            False),
        'batch_size':        pick(args.batch_size,         'batch_size',         10),
        'n_workers':         pick(args.n_workers,          'n_workers',          10),
        'save_survival':     pick_bool(args.save_survival,  'save_survival',     True),
        'save_states':       pick_bool(args.save_states,    'save_states',       False),
        'save_mfpt':         pick_bool(args.save_mfpt,      'save_mfpt',         True),
        'max_nucs':          pick(args.max_nucs,           'max_nucs'),
        'max_nucs_seed':     pick(args.max_nucs_seed,      'max_nucs_seed',     0),
        'subids_start':      pick(args.subids_start,       'subids_start'),
        'subids_end':        pick(args.subids_end,         'subids_end'),
        'check_matrix_density': pick_bool(args.check_matrix_density, 'check_matrix_density', False),
        # TNP2 v2.0
        'tnp2_enabled':            pick_bool(args.tnp2_enabled,            'tnp2_enabled',            False),
        'tnp2_eps_cpg':            pick(args.tnp2_eps_cpg,                 'tnp2_eps_cpg',            1.0),
        'tnp2_mu_t0':              pick(args.tnp2_mu_t0,                   'tnp2_mu_t0',              -8.0),
        'fasta_path':              pick_path(args.fasta_path,              'fasta_path'),
        'fasta_id_style':          pick(args.fasta_id_style,               'fasta_id_style',          'name'),
    }


def _matrix_nnz(Q) -> int:
    if hasattr(Q, "nnz"):
        return int(Q.nnz)
    return int(np.count_nonzero(Q))


def _matrix_storage_type(Q) -> str:
    if hasattr(Q, "format"):
        return f"sparse/{Q.format}"
    return "dense/numpy"


def _load_first_nucleosome(p: dict, config: MarkovConfig, subids_range):
    if p['dataset_dir'] is not None:
        gen = nucleosome_generator_sprm(
            dataset_dir=p['dataset_dir'],
            k_wrap=config.k_wrap,
            kT=config.kT,
            binding_sites=config.binding_sites
        )
    elif subids_range is not None:
        gen = nucleosome_generator(
            file_path=p['infile'],
            k_wrap=config.k_wrap,
            binding_sites=config.binding_sites,
            subids=np.arange(*subids_range).tolist()
        )
    else:
        gen = nucleosome_generator(
            file_path=p['infile'],
            k_wrap=config.k_wrap,
            binding_sites=config.binding_sites
        )

    try:
        return next(gen)
    except StopIteration as exc:
        raise ValueError("No nucleosomes found for matrix-density check.") from exc


def print_matrix_density_check(p: dict, config: MarkovConfig, subids_range) -> None:
    nuc = _load_first_nucleosome(p, config, subids_range)
    Q_full, Q_TT, Q_AT, states, _, abs_index = build_full_Q_from_nucleosome(
        nuc,
        k_wrap=config.k_wrap,
        protamine_params=config.protamine_params,
        kT=config.kT,
        binding_sites=config.binding_sites,
        sparse=config.sparse,
        dimensionless=config.dimensionless,
        eads_delta=config.eads_delta,
        eads_weight_mode=config.eads_weight_mode,
        eads_apply=config.eads_apply,
    )

    matrices = {
        "Q_full": Q_full,
        "Q_TT": Q_TT,
        "Q_AT": Q_AT,
    }

    print("\nMatrix density check")
    print("====================")
    print(f"nucleosome id: {nuc.id}")
    print(f"nucleosome subid: {nuc.subid}")
    print(f"binding_sites: {config.binding_sites}")
    print(f"transient states: {len(states)}")
    print(f"absorbing index: {abs_index}")
    print(f"sparse requested: {config.sparse}")
    print()

    for name, Q in matrices.items():
        nnz = _matrix_nnz(Q)
        total = Q.shape[0] * Q.shape[1]
        density = matrix_density(Q)
        print(f"{name}:")
        print(f"  storage: {_matrix_storage_type(Q)}")
        print(f"  shape: {Q.shape}")
        print(f"  nonzero entries: {nnz} / {total}")
        print(f"  density: {density:.6f} ({density * 100:.4f}%)")
        print()


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
    if p['storage_dir'] is None and not p['check_matrix_density']:
        raise ValueError("storage_dir is required (set in YAML or via --storage_dir).")

    # ── Build MarkovConfig ────────────────────────────────────────────────────
    tnp2_cfg = TNP2Config(
        enabled=p['tnp2_enabled'],
        eps_cpg=float(p['tnp2_eps_cpg']),
        mu_t0=float(p['tnp2_mu_t0']),
    )

    config = MarkovConfig(
        k_wrap=p['k_wrap'],
        binding_sites=p['binding_sites'],
        prot_k_bind=p['prot_k_bind'],
        prot_k_unbind=p['prot_k_unbind'],
        prot_p_conc=p['prot_p_conc'],
        prot_cooperativity=p['prot_cooperativity'],
        eads_delta=p['eads_delta'],
        eads_weight_mode=p['eads_weight_mode'],
        eads_apply=p['eads_apply'],
        tau_max=p['tau_max'],
        tau_steps=p['tau_steps'],
        method=p['method'],
        sparse=p['sparse'],
        batch_size=p['batch_size'],
        n_workers=p['n_workers'],
        max_nucs=p['max_nucs'],
        max_nucs_seed=p['max_nucs_seed'],
        save_survival=p['save_survival'],
        save_states=p['save_states'],
        save_mfpt=p['save_mfpt'],
        tnp2=tnp2_cfg,
    )
    logger.info(f"Configuration: {config}")

    subids_range = (
        (p['subids_start'], p['subids_end'])
        if p['subids_start'] is not None and p['subids_end'] is not None
        else None
    )

    if p['check_matrix_density']:
        print_matrix_density_check(p, config, subids_range)
        return

    p['storage_dir'].mkdir(parents=True, exist_ok=True)

    # ── Storage paths ─────────────────────────────────────────────────────────
    storage = MarkovStorage(base_dir=p['storage_dir'], use_index=False)

    params_for_storage = {
        'k_wrap':        p['k_wrap'],
        'prot_params':   {
            'k_unbind':      p['prot_k_unbind'],
            'k_bind':        p['prot_k_bind'],
            'p_conc':        p['prot_p_conc'],
            'cooperativity': p['prot_cooperativity'],
        },
        'binding_sites': p['binding_sites'],
        'tau_max':       p['tau_max'],
        'tau_steps':     p['tau_steps'],
        'method':        p['method'],
        'sparse':        p['sparse'],
        'eads_delta':    p['eads_delta'],
        'eads_weight_mode': p['eads_weight_mode'],
        'eads_apply':    p['eads_apply'],
        'max_nucs':      p['max_nucs'],
        'max_nucs_seed': p['max_nucs_seed'],
        'tnp2_enabled':            p['tnp2_enabled'],
        'tnp2_eps_cpg':            float(p['tnp2_eps_cpg']),
        'tnp2_mu_t0':              float(p['tnp2_mu_t0']),
    }

    if p['dataset_dir'] is not None:
        file_id = p['dataset_dir'].name
    else:
        file_id = p['infile'].stem
    if p['dataset']:
        file_id = f"{p['dataset']}_{file_id}"

    output_paths = storage.get_output_paths(params_for_storage, file_id)
    tsv_outfile      = output_paths['summary']
    survival_outfile = output_paths['survivals']

    # ── Run ───────────────────────────────────────────────────────────────────
    fasta_path = p['fasta_path'] if config.tnp2.enabled or p['fasta_path'] is not None else None
    run_markov_solver(
        tsv_outfile=tsv_outfile,
        survival_outfile=survival_outfile,
        config=config,
        file_path=p['infile'],
        dataset_dir=p['dataset_dir'],
        logger=logger,
        max_nucs=p['max_nucs'],
        subids_range=subids_range,
        fasta_path=fasta_path,
        fasta_id_style=p['fasta_id_style'],
    )

    elapsed = time.perf_counter() - start
    logger.info(f"Total execution time: {dt.timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    main()
