"""
Sweep result loader.

Walks a Markov sweep ``storage_root`` and produces a tidy long-form DataFrame:
one row per (dataset, parameter combination, nucleosome).  Each row carries
the MFPT plus all sweep parameters flattened into columns.

Optionally joins sequence features (GC, rho_CpG, N_CpG) from per-dataset FASTA
files so analyses can bin by composition.

Usage
-----
    from src.analysis.sweep_loader import load_sweep, attach_sequence_features

    df = load_sweep("output/markov_output/eads_weighted_sweep")
    df = attach_sequence_features(df, dataset_fasta=...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def parse_fasta(fa_path: Union[str, Path], id_style: str = "name") -> Dict[str, str]:
    """Parse a FASTA file into ``{key: sequence}``."""
    if id_style not in ("coord", "name"):
        raise ValueError(f"id_style must be 'coord' or 'name', got {id_style!r}")

    records: Dict[str, str] = {}
    current_key: Optional[str] = None
    seq_parts: list[str] = []

    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_key is not None:
                    records[current_key] = "".join(seq_parts).upper()
                header = line[1:]
                parts = header.split("|")
                if id_style == "coord":
                    current_key = parts[0].strip()
                elif len(parts) >= 2:
                    current_key = parts[1].strip()
                else:
                    current_key = parts[0].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_key is not None:
        records[current_key] = "".join(seq_parts).upper()

    return records


def count_cpg(seq: Optional[str]) -> int:
    """Count CpG dinucleotides in a DNA sequence."""
    if not seq:
        return 0
    s = seq.upper()
    return sum(1 for i in range(len(s) - 1) if s[i] == "C" and s[i + 1] == "G")


# ─── flattening parameters.json ───────────────────────────────────────────────

def _flatten_params(params: dict, param_dir_name: str) -> dict:
    """Flatten the nested parameters.json into one row of columns."""
    prot = params.get('prot_params', {}) or {}
    flat = {
        'param_dir': param_dir_name,
        'k_wrap':            params.get('k_wrap'),
        'tau_max':           params.get('tau_max'),
        'tau_steps':         params.get('tau_steps'),
        'method':            params.get('method'),
        'binding_sites':     params.get('binding_sites'),
        'sparse':            params.get('sparse'),
        'dimensionless':     params.get('dimensionless'),
        'prot_k_bind':       prot.get('k_bind'),
        'prot_k_unbind':     prot.get('k_unbind'),
        'prot_p_conc':       prot.get('p_conc'),
        'prot_cooperativity': prot.get('cooperativity'),
        'eads_delta':        params.get('eads_delta'),
        'eads_weight_mode':  params.get('eads_weight_mode'),
        'eads_apply':        params.get('eads_apply'),
        'max_nucs':          params.get('max_nucs'),
        'max_nucs_seed':     params.get('max_nucs_seed'),
    }
    return flat


# ─── walking storage_root ────────────────────────────────────────────────────

def _iter_summary_files(storage_root: Path):
    """Yield (dataset, param_dir_path, summary_tsv_path)."""
    for params_path in sorted(storage_root.glob('*/**/parameters.json')):
        rel = params_path.relative_to(storage_root).parts
        if len(rel) < 2:
            continue
        dataset = rel[0]
        param_dir = params_path.parent
        for tsv in sorted((param_dir / 'summaries').glob('*.tsv')):
            yield dataset, param_dir, tsv


def load_sweep(
    storage_root: Union[str, Path],
    drop_nonfinite: bool = True,
    drop_nonpositive_mfpt: bool = True,
    add_log10: bool = True,
) -> pd.DataFrame:
    """Load every summary TSV under ``storage_root`` into a long DataFrame.

    Columns: dataset, id, subid, mfpt[, log10_mfpt], plus every flattened
    parameter from parameters.json (k_wrap, prot_p_conc, eads_*, ...).

    Returns an empty DataFrame if ``storage_root`` is missing or has no runs.
    """
    storage_root = Path(storage_root)
    if not storage_root.exists():
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    for dataset, param_dir, tsv in _iter_summary_files(storage_root):
        with open(param_dir / 'parameters.json') as f:
            params = json.load(f)
        params_flat = _flatten_params(params, param_dir.name)

        df = pd.read_csv(tsv, sep='\t')
        if 'mfpt' not in df.columns:
            continue
        df['mfpt'] = pd.to_numeric(df['mfpt'], errors='coerce')
        df['dataset'] = dataset
        for k, v in params_flat.items():
            df[k] = v
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if drop_nonfinite:
        out = out[np.isfinite(out['mfpt'])]
    if drop_nonpositive_mfpt:
        out = out[out['mfpt'] > 0.0]
    if add_log10:
        out['log10_mfpt'] = np.log10(out['mfpt'])
    return out.reset_index(drop=True)


# ─── optional sequence-feature join ──────────────────────────────────────────

def _gc_fraction(seq: str) -> float:
    if not seq:
        return np.nan
    s = seq.upper()
    n = len(s)
    return (s.count('G') + s.count('C')) / n if n else np.nan


def attach_sequence_features(
    df: pd.DataFrame,
    dataset_fasta: Dict[str, dict],
    project_root: Optional[Path] = None,
    path_roots: Optional[Dict[str, Union[str, Path]]] = None,
) -> pd.DataFrame:
    """Add ``gc``, ``n_cpg``, ``rho_cpg`` columns by joining FASTA on ``id``.

    ``dataset_fasta`` is the same registry shape used by the sweep YAML, e.g.:

        {'ret_single_nuc': {'path': 'hamnucret_data/.../foo.fa', 'id_style': 'name'},
         'ctrl02_random_genome_gcmatched': {'path': '...', 'id_style': 'coord'}}

    ``path_roots`` optionally expands first-component aliases such as
    ``hamnucret_fasta_dir/...`` from the sweep YAML.

    Datasets without an entry get NaN features (and a warning is printed).
    """
    if df.empty:
        return df.copy()
    project_root = Path(project_root) if project_root is not None else Path.cwd()

    fasta_cache: Dict[str, Dict[str, str]] = {}
    for dataset in df['dataset'].unique():
        entry = dataset_fasta.get(dataset)
        if entry is None:
            print(f"[attach_sequence_features] no FASTA registered for {dataset!r}")
            continue
        if isinstance(entry, str):
            path, id_style = entry, 'name'
        else:
            path = entry.get('path')
            id_style = entry.get('id_style', 'name')
        if path is None:
            continue
        fa = Path(path)
        if not fa.is_absolute():
            parts = fa.parts
            if path_roots and parts and parts[0] in path_roots:
                fa = Path(path_roots[parts[0]]).expanduser().joinpath(*parts[1:])
            else:
                fa = project_root / fa
        if not fa.exists():
            print(f"[attach_sequence_features] FASTA missing for {dataset}: {fa}")
            continue
        fasta_cache[dataset] = parse_fasta(fa, id_style=id_style)

    out = df.copy()
    out['gc'] = np.nan
    out['n_cpg'] = np.nan
    out['rho_cpg'] = np.nan

    for dataset, seqmap in fasta_cache.items():
        mask = out['dataset'] == dataset
        ids = out.loc[mask, 'id'].astype(str)
        seqs = ids.map(seqmap)
        out.loc[mask, 'gc'] = seqs.map(_gc_fraction)
        out.loc[mask, 'n_cpg'] = seqs.map(lambda s: count_cpg(s) if s else np.nan)
        lengths = seqs.map(lambda s: len(s) if s else np.nan)
        out.loc[mask, 'rho_cpg'] = out.loc[mask, 'n_cpg'] / lengths
    return out


# ─── helpers for slicing ─────────────────────────────────────────────────────

SWEEP_AXES = (
    'prot_p_conc',
    'prot_cooperativity',
    'eads_delta',
    'eads_weight_mode',
    'eads_apply',
)


def varying_columns(df: pd.DataFrame, candidates=SWEEP_AXES) -> List[str]:
    """Return the subset of candidate columns that vary across the DataFrame."""
    if df.empty:
        return []
    return [c for c in candidates if c in df.columns and df[c].nunique(dropna=False) > 1]


def dataset_determined_columns(df: pd.DataFrame, candidates=SWEEP_AXES) -> List[str]:
    """Columns that vary across the frame but are constant within each dataset.

    Examples in this codebase: ``eads_apply``, ``eads_delta``, ``eads_weight_mode``
    are normalized by the sweep's per-dataset rule (RET → False/0/none,
    controls → True/<sweep value>/<weight_mode>).  Filtering on them in a
    cross-dataset slice silently drops one class.  Use this list to exclude
    such axes from auto-pickers and example slices.
    """
    if df.empty or 'dataset' not in df.columns:
        return []
    out: List[str] = []
    for c in candidates:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue
        per_dataset = df.groupby('dataset')[c].nunique(dropna=False)
        if per_dataset.max() <= 1:
            out.append(c)
    return out


def cross_dataset_axes(df: pd.DataFrame, candidates=SWEEP_AXES) -> List[str]:
    """Real sweep axes: columns that vary AND vary independently of dataset.

    Use this when picking what the user can filter on in a cross-dataset
    comparison without accidentally excluding RET or controls.
    """
    determined = set(dataset_determined_columns(df, candidates))
    return [c for c in varying_columns(df, candidates) if c not in determined]


# ─── dataset class helper (RET vs ctrl) ──────────────────────────────────────

def add_dataset_class(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``dataset_class`` column ('RET' or 'CTRL')."""
    if df.empty or 'dataset' not in df.columns:
        return df.copy()
    out = df.copy()
    out['dataset_class'] = out['dataset'].apply(
        lambda d: 'RET' if str(d).lower().startswith('ret') else 'CTRL'
    )
    return out


def filter_df(df: pd.DataFrame, **eq) -> pd.DataFrame:
    """Filter ``df`` to rows where each kwarg matches the column value.

    A list value is treated as ``isin``; a scalar is exact equality.
    Columns missing from ``df`` are silently ignored so the same call works
    on partial sweeps.
    """
    out = df
    for k, v in eq.items():
        if k not in out.columns:
            continue
        if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
            out = out[out[k].isin(list(v))]
        else:
            out = out[out[k] == v]
    return out
