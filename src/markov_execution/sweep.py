#!/usr/bin/env python3
"""
Sweep Runner
============

Run the Markov solver over a grid of (dataset × parameter combinations).

Usage:
    python -m src.markov_execution.sweep
    python -m src.markov_execution.sweep --config path/to/sweep.yaml

The sweep YAML defines:
  - sprm_root / datasets  : which SPRM dataset folders to process
  - param_grid            : dict of lists whose Cartesian product is explored
  - fixed params          : applied to every run (k_wrap, tau_max, etc.)
  - skip_existing         : if true, skip runs where output already exists
"""

import os
if os.environ.get("IMPORT_ENV_SETTINGS", "1") == "1":
    from src.config.env_setting import *

import argparse
import itertools
import datetime as dt
import time
from pathlib import Path

import yaml

from src.utils.logger_util import get_logger
from src.analysis.markov_solver.tnp2 import TNP2Config
from .config import MarkovConfig
from .orchestrator import run_markov_solver
from .storage import MarkovStorage


# ── dataset → FASTA registry ─────────────────────────────────────────────────

def _resolve_config_path(path: str, project_root: Path, path_roots: dict | None = None) -> Path:
    """Resolve a YAML path, allowing first-component aliases.

    Example:
        hamnucret_fasta_dir: /abs/path/HAMNucRetSeq_pipeline
        path: hamnucret_fasta_dir/SOM_output/.../stable147.fa
    """
    raw = Path(path)
    if raw.is_absolute():
        return raw
    parts = raw.parts
    if path_roots and parts and parts[0] in path_roots:
        return Path(path_roots[parts[0]]).expanduser().joinpath(*parts[1:])
    return project_root / raw


def _resolve_dataset_fasta(dataset: str, dataset_fasta: dict, project_root: Path, path_roots: dict | None = None):
    """Look up FASTA file + id_style for a dataset.

    Returns (fasta_path, id_style) or (None, 'name') if not configured.
    Entry shape in YAML:
        dataset_fasta:
          ret_single_nuc:
            path: hamnucret_data/SPRM_IN_SEQ/RET_data/single_nuc/pooled_peaks_single_nuc_nuc147.fa
            id_style: name
    """
    if not dataset_fasta or dataset not in dataset_fasta:
        return None, 'name'
    entry = dataset_fasta[dataset]
    if isinstance(entry, str):
        path = entry
        id_style = 'name'
    else:
        path = entry.get('path')
        id_style = entry.get('id_style', 'name')
    if path is None:
        return None, id_style
    fasta = _resolve_config_path(path, project_root, path_roots)
    return fasta, id_style

_DEFAULT_SWEEP = Path(__file__).parent / "markov_sweep.yaml"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _param_combinations(param_grid: dict) -> list[dict]:
    """Return a list of dicts, one per point in the Cartesian product."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], list) else [param_grid[k]]
              for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _normalized_combo_for_dataset(dataset: str, combo: dict, has_fasta: bool = True) -> dict:
    """Apply dataset-level sweep rules and return the effective combo.

    ``has_fasta`` forces TNP2 off when no FASTA file is configured for the
    dataset, since the layer needs sequences to compute site-resolved CpG.
    """
    normalized = dict(combo)
    if dataset.lower().startswith('ret'):
        normalized['eads_apply'] = False
        normalized['eads_delta'] = 0.0
        normalized['eads_weight_mode'] = 'none'
    else:
        normalized['eads_apply'] = True
        normalized['eads_delta'] = float(normalized.get('eads_delta', 0.0))
        normalized['eads_weight_mode'] = normalized.get('eads_weight_mode', 'uniform')
    if not has_fasta:
        normalized['tnp2_enabled'] = False
    return normalized


def _combo_is_meaningful(combo: dict) -> bool:
    """Skip cooperativity repeats when no protamine is present."""
    p_conc = float(combo.get('prot_p_conc', 0.0))
    cooperativity = float(combo.get('prot_cooperativity', 0.0))
    return not (p_conc == 0.0 and cooperativity != 0.0)


def _output_exists(storage: MarkovStorage, params_for_storage: dict, file_id: str) -> bool:
    """Return True if the summary TSV for this (params, file_id) already exists."""
    try:
        paths = storage.get_output_paths(params_for_storage, file_id)
        return Path(paths['summary']).exists()
    except Exception:
        return False


def _build_params_for_storage(fixed: dict, combo: dict) -> dict:
    p = {**fixed, **combo}   # combo values override fixed values
    return {
        'k_wrap':        p.get('k_wrap', 1.0),
        'prot_params': {
            'k_unbind':      p.get('prot_k_unbind', 89.7),
            'k_bind':        p.get('prot_k_bind', 1.0),
            'p_conc':        p.get('prot_p_conc', 0.0),
            'cooperativity': p.get('prot_cooperativity', 0.0),
        },
        'binding_sites': p.get('binding_sites', 14),
        'tau_max':       p.get('tau_max', 1000.0),
        'tau_steps':     p.get('tau_steps', 500),
        'method':        p.get('method', 'ode'),
        'sparse':        p.get('sparse', False),
        'eads_delta':    p.get('eads_delta', 0.0),
        'eads_weight_mode': p.get('eads_weight_mode', 'none'),
        'eads_apply':    p.get('eads_apply', False),
        'max_nucs':      p.get('max_nucs'),
        'max_nucs_seed': p.get('max_nucs_seed', 0),
        # TNP2 v2.0 - storage layer collapses these to a single key when disabled
        'tnp2_enabled':            bool(p.get('tnp2_enabled', False)),
        'tnp2_eps_cpg':            float(p.get('tnp2_eps_cpg', 1.0)),
        'tnp2_mu_t0':              float(p.get('tnp2_mu_t0', -8.0)),
    }


def _build_tnp2_config(p: dict) -> TNP2Config:
    return TNP2Config(
        enabled=bool(p.get('tnp2_enabled', False)),
        eps_cpg=float(p.get('tnp2_eps_cpg', 1.0)),
        mu_t0=float(p.get('tnp2_mu_t0', -8.0)),
    )


def _build_config(fixed: dict, combo: dict) -> MarkovConfig:
    p = {**fixed, **combo}
    return MarkovConfig(
        k_wrap=p.get('k_wrap', 1.0),
        binding_sites=p.get('binding_sites', 14),
        prot_k_bind=p.get('prot_k_bind', 1.0),
        prot_k_unbind=p.get('prot_k_unbind', 89.7),
        prot_p_conc=p.get('prot_p_conc', 0.0),
        prot_cooperativity=p.get('prot_cooperativity', 0.0),
        eads_delta=p.get('eads_delta', 0.0),
        eads_weight_mode=p.get('eads_weight_mode', 'none'),
        eads_apply=p.get('eads_apply', False),
        tau_max=p.get('tau_max', 1000.0),
        tau_steps=p.get('tau_steps', 500),
        method=p.get('method', 'ode'),
        sparse=p.get('sparse', False),
        batch_size=p.get('batch_size', 10),
        n_workers=p.get('n_workers', 10),
        max_nucs=p.get('max_nucs'),
        max_nucs_seed=p.get('max_nucs_seed', 0),
        save_survival=p.get('save_survival', True),
        save_states=p.get('save_states', False),
        save_mfpt=p.get('save_mfpt', True),
        tnp2=_build_tnp2_config(p),
    )


def _combo_label(combo: dict) -> str:
    if not combo:
        return '(fixed params only)'
    return ', '.join(f'{k}={v}' for k, v in combo.items())


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Sweep Markov solver over datasets × parameter grid.')
    parser.add_argument('--config', type=Path, default=_DEFAULT_SWEEP,
                        help='Path to sweep YAML config.')
    args = parser.parse_args()

    logger = get_logger(__name__, log_file=None, level='INFO')

    # Workers need TMPDIR set before the process pool starts (same as cli.py)
    tmp_dir = Path(__file__).parent.parent.parent / 'temps'
    tmp_dir.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(tmp_dir)
    cfg = _load_yaml(args.config)
    logger.info(f'Loaded sweep config from: {args.config}')

    # ── resolve paths ────────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent.parent
    sprm_root    = project_root / cfg.get('sprm_root', 'SPRM_data')
    storage_root = project_root / cfg.get('storage_root', 'markov_output')

    # Datasets: explicit list or all subdirs of sprm_root
    if cfg.get('datasets'):
        datasets = [str(d) for d in cfg['datasets']]
    else:
        datasets = sorted(d.name for d in sprm_root.iterdir() if d.is_dir())

    # FASTA registry (dataset → {path, id_style}) - required when TNP2 is enabled
    dataset_fasta = cfg.get('dataset_fasta', {}) or {}
    path_roots = {
        key: value
        for key, value in cfg.items()
        if key.endswith('_dir') or key.endswith('_root')
    }
    path_root_keys = set(path_roots)

    # Parameter grid and fixed params
    param_grid   = cfg.get('param_grid', {})
    fixed_params = {k: v for k, v in cfg.items()
                    if k not in ('sprm_root', 'storage_root', 'datasets',
                                 'param_grid', 'skip_existing', 'dataset_fasta')
                    and k not in path_root_keys}
    raw_combos   = _param_combinations(param_grid)
    combos       = [combo for combo in raw_combos if _combo_is_meaningful({**fixed_params, **combo})]
    skip_existing = cfg.get('skip_existing', True)

    # ── build job list ────────────────────────────────────────────────────────
    # Dedup uses the storage directory name so that grid points which collapse
    # into the same output dir (e.g. TNP2 grid axes when tnp2_enabled=False)
    # are recognised as duplicates.
    jobs = []
    seen_jobs = set()
    dedup_storage = MarkovStorage(base_dir=storage_root, use_index=False)
    for dataset in datasets:
        fasta_path, fasta_id_style = _resolve_dataset_fasta(
            dataset, dataset_fasta, project_root, path_roots
        )
        has_fasta = fasta_path is not None and fasta_path.exists()
        if fasta_path is not None and not fasta_path.exists():
            logger.warning(f"Configured FASTA missing for {dataset}: {fasta_path} - TNP2 will be forced off.")
        for combo in combos:
            effective_combo = _normalized_combo_for_dataset(dataset, combo, has_fasta=has_fasta)
            params_for_storage = _build_params_for_storage(fixed_params, effective_combo)
            job_key = (dataset, dedup_storage.get_directory_name(params_for_storage))
            if job_key in seen_jobs:
                continue
            seen_jobs.add(job_key)
            jobs.append((dataset, effective_combo, fasta_path, fasta_id_style))

    skipped_combos = len(raw_combos) - len(combos)
    logger.info(f'{len(datasets)} dataset(s) × {len(raw_combos)} raw combo(s)')
    logger.info(f'Skipped {skipped_combos} combo(s) with p_conc=0 and cooperativity>0')
    logger.info(f'Effective job count after RET Eads deduplication: {len(jobs)}')
    logger.info(f'skip_existing = {skip_existing}')

    run_count = skip_count = error_count = 0
    sweep_start = time.perf_counter()

    for job_idx, (dataset, combo, fasta_path, fasta_id_style) in enumerate(jobs, 1):
        dataset_dir = sprm_root / dataset
        storage_dir = storage_root / dataset

        if not dataset_dir.exists():
            logger.warning(f'[{job_idx}/{len(jobs)}] Dataset not found, skipping: {dataset_dir}')
            error_count += 1
            continue

        params_for_storage = _build_params_for_storage(fixed_params, combo)
        storage = MarkovStorage(base_dir=storage_dir, use_index=False)
        file_id = dataset

        if skip_existing and _output_exists(storage, params_for_storage, file_id):
            logger.info(f'[{job_idx}/{len(jobs)}] SKIP (exists): {dataset}  |  {_combo_label(combo)}')
            skip_count += 1
            continue

        logger.info(f'[{job_idx}/{len(jobs)}] RUN: {dataset}  |  {_combo_label(combo)}')
        t0 = time.perf_counter()

        try:
            config = _build_config(fixed_params, combo)
            output_paths = storage.get_output_paths(params_for_storage, file_id)
            attach_fasta = fasta_path if config.tnp2.enabled else None
            run_markov_solver(
                tsv_outfile=output_paths['summary'],
                survival_outfile=output_paths['survivals'],
                config=config,
                dataset_dir=dataset_dir,
                logger=logger,
                fasta_path=attach_fasta,
                fasta_id_style=fasta_id_style,
            )
            elapsed = dt.timedelta(seconds=time.perf_counter() - t0)
            logger.info(f'  Done in {elapsed}')
            run_count += 1

        except Exception as exc:
            logger.error(f'  FAILED: {exc}')
            error_count += 1

    # ── summary ───────────────────────────────────────────────────────────────
    total_elapsed = dt.timedelta(seconds=time.perf_counter() - sweep_start)
    logger.info('=' * 60)
    logger.info(f'Sweep complete in {total_elapsed}')
    logger.info(f'  Ran:     {run_count}')
    logger.info(f'  Skipped: {skip_count}')
    logger.info(f'  Errors:  {error_count}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
