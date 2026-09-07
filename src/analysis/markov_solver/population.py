"""Population-level survival for the Markov sweep.

The sweep stores one survival curve per sequence. The figures need the population
average

    S_D(tau; c, J) = (1 / N_D) * sum_{i in D} S_i(tau; c, J)

and its half-eviction time t50, where S_D(t50) = 0.5. Averaging 20k-by-1000 arrays
over 126 sweep cells costs about two minutes and 20 GB of reads, so the reduced
form is cached to a single npz.

On t50 and the simulation window
--------------------------------
The grid stops at tau_max = 1e7 (5.5 days at k_wrap = 21/s). Only 17 of the 126
cells have S_D falling below 0.5 inside it, so reading t50 off the grid leaves the
concentration scans almost entirely right-censored and makes the cooperative
speed-up A_D undefined nearly everywhere.

Absorption is far slower than relaxation within the 105 transient (l, r) states, so
each sequence quasi-equilibrates long before it escapes and its first-passage law
collapses to a single exponential, S_i(t) = exp(-t / m_i) with m_i its MFPT. The
population average is then a mixture of exponentials over the (uncensored, GTH-exact)
MFPTs, which gives t50 at any depth. `check_exponential` measures the error rather
than assuming it: across all 126 cells the median max |S_approx - S_measured| is
0.00000 and the worst is 0.099, confined to the saturating cells (c = 1000, J = 4.5)
where escape stops being slow. Where the grid resolves t50 on its own, the two
estimates agree within 0.08 decades.
"""
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0

Cell = Tuple[str, float, float]          # (dataset dir name, c, cooperativity)
_FIELDS = ('mean', 'se', 'lo', 'hi', 'logmfpt')


def aggregate_survival(sweep_root, datasets: Optional[Iterable[str]] = None,
                       cache=None, rebuild: bool = False, envelope=(10.0, 90.0),
                       verbose: bool = True) -> Tuple[np.ndarray, Dict[Cell, dict]]:
    """Reduce every sweep cell to its population survival. Returns (tau, cells).

    Each entry of `cells` holds

        mean     population-averaged S_D(tau)
        se       standard error of that mean (sequence sd / sqrt(N))
        lo, hi   `envelope` percentiles of S_i across sequences -- heterogeneity
                 between sequences, which is not an uncertainty on the mean
        logmfpt  log10 MFPT for every sequence in the cell
        n        number of sequences
    """
    sweep_root = Path(sweep_root)
    cache = Path(cache) if cache is not None else None
    allowed = None if datasets is None else set(datasets)
    if cache is not None and cache.exists() and not rebuild:
        tau, cells = _load_cache(cache)
        # A sweep that has grown since the cache was written would otherwise be
        # half-visible: callers reading concentrations from the summaries would
        # index cells that are not here, and fail far from the cause.
        missing = _sweep_cells(sweep_root, allowed) - set(cells)
        if not missing:
            return tau, cells
        # Announced even when quiet: this is a minutes-long rebuild, not a no-op.
        cs = sorted({c for _, c, _ in missing})
        print(f'  cache holds {len(cells)} cells, sweep has '
              f'{len(cells) + len(missing)}; rebuilding. New c: {cs}', flush=True)

    tau, cells = None, {}
    for dsdir in sorted(sweep_root.iterdir()):
        if not dsdir.is_dir() or (allowed is not None and dsdir.name not in allowed):
            continue
        for run in sorted(dsdir.iterdir()):
            pf = run / 'parameters.json'
            if not pf.exists():
                continue
            pp = json.loads(pf.read_text())['prot_params']
            # Run directory names round the concentration, so trust the json only.
            key = (dsdir.name, float(pp['p_conc']), float(pp['cooperativity']))
            if key in cells:
                raise RuntimeError(f'duplicate sweep cell {key}')
            f = next((run / 'survivals').glob('*.parquet'))
            d = pd.read_parquet(f, columns=['tau_grid', 'survival', 'mfpt'])
            g = np.asarray(d['tau_grid'].iloc[0], dtype=float)
            if tau is None:
                tau = g
            elif not np.allclose(g, tau):
                raise RuntimeError(f'{key}: tau grid differs from the others')
            S = np.vstack(d['survival'].to_numpy()).astype(float)
            m = d['mfpt'].to_numpy(float)
            cells[key] = dict(
                mean=S.mean(0),
                se=S.std(0, ddof=1) / np.sqrt(S.shape[0]),
                lo=np.percentile(S, envelope[0], axis=0),
                hi=np.percentile(S, envelope[1], axis=0),
                logmfpt=np.log10(m[np.isfinite(m) & (m > 0)]),
                n=S.shape[0])
            if verbose:
                print(f'  {dsdir.name[:34]:34s} c={key[1]:<8g} J={key[2]:<4g} '
                      f'n={S.shape[0]}', flush=True)
    if tau is None:
        raise RuntimeError(f'no sweep cells found under {sweep_root}')
    if cache is not None:
        _save_cache(cache, tau, cells)
    return tau, cells


def _sweep_cells(sweep_root, allowed=None):
    """Every (dataset, c, J) the sweep holds, from parameters.json alone.

    Cheap enough to check on every load: it reads no survival data.
    """
    keys = set()
    for dsdir in sorted(Path(sweep_root).iterdir()):
        if not dsdir.is_dir() or (allowed is not None and dsdir.name not in allowed):
            continue
        for run in sorted(dsdir.iterdir()):
            pf = run / 'parameters.json'
            if pf.exists():
                pp = json.loads(pf.read_text())['prot_params']
                keys.add((dsdir.name, float(pp['p_conc']),
                          float(pp['cooperativity'])))
    return keys


def _save_cache(cache, tau, cells):
    flat = {'tau_grid': tau}
    for (ds, c, J), v in cells.items():
        stem = f'{ds}|{c!r}|{J!r}'
        for name in _FIELDS:
            flat[f'{stem}|{name}'] = v[name]
        flat[f'{stem}|n'] = np.array(v['n'])
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **flat)


def _load_cache(cache):
    z = np.load(cache, allow_pickle=False)
    tau = z['tau_grid']
    cells = {}
    for k in z.files:
        if k == 'tau_grid' or not k.endswith('|mean'):
            continue
        ds, c, J, _ = k.rsplit('|', 3)
        stem = k[:-len('|mean')]
        cells[(ds, float(c), float(J))] = {
            **{name: z[f'{stem}|{name}'] for name in _FIELDS},
            'n': int(z[f'{stem}|n'])}
    return tau, cells


def t50_exponential(logmfpt: np.ndarray, level: float = 0.5) -> float:
    """t where the exponential mixture over per-sequence MFPTs falls to `level`.

    Solves mean_i exp(-t / 10**logmfpt_i) = level for log10 t. Uncensored: it does
    not care where the simulation grid stopped.
    """
    from scipy.optimize import brentq
    lm = np.asarray(logmfpt, float)
    lm = lm[np.isfinite(lm)]
    if len(lm) == 0:
        return np.nan
    f = lambda L: np.exp(-np.exp((L - lm) * np.log(10))).mean() - level
    return 10.0 ** brentq(f, lm.min() - 8, lm.max() + 8, xtol=1e-10)


def t50_resolved(mean: np.ndarray, tau: np.ndarray, level: float = 0.5) -> float:
    """t50 read straight off the grid, or nan when the curve never gets there."""
    mean = np.asarray(mean, float)
    if mean[-1] >= level:
        return np.nan
    i = int(np.argmax(mean < level))
    if i == 0:
        return float(tau[0])
    x0, x1 = np.log10(tau[i - 1]), np.log10(tau[i])
    y0, y1 = mean[i - 1], mean[i]
    return 10.0 ** (x0 + (level - y0) / (y1 - y0) * (x1 - x0))


def population_t50(tau, cells, level: float = 0.5) -> pd.DataFrame:
    """t50 for every cell, by both routes.

    Columns: dataset, c, J, n, log_t50 (exponential mixture, always defined),
    log_t50_grid (nan when right-censored) and `resolved`.
    """
    rows = []
    for (ds, c, J), v in cells.items():
        g = t50_resolved(v['mean'], tau, level)
        rows.append(dict(dataset=ds, c=c, J=J, n=v['n'],
                         log_t50=np.log10(t50_exponential(v['logmfpt'], level)),
                         log_t50_grid=np.log10(g) if np.isfinite(g) else np.nan,
                         resolved=np.isfinite(g)))
    return pd.DataFrame(rows).sort_values(['dataset', 'J', 'c'], ignore_index=True)


def check_exponential(tau, cells) -> pd.DataFrame:
    """Per-cell error of the exponential-mixture approximation to S_D(tau)."""
    t = np.maximum(np.asarray(tau, float), 1e-12)
    rows = []
    for (ds, c, J), v in cells.items():
        lm = v['logmfpt']
        approx = np.exp(-np.exp(np.log10(t)[:, None] * np.log(10)
                                - lm[None, :] * np.log(10))).mean(1)
        d = np.abs(approx - v['mean'])
        rows.append(dict(dataset=ds, c=c, J=J, max_abs_dS=d.max(),
                         tau_at_max=t[int(d.argmax())],
                         S_end_meas=v['mean'][-1], S_end_approx=approx[-1]))
    return pd.DataFrame(rows).sort_values('max_abs_dS', ascending=False,
                                          ignore_index=True)


def tau_to_days(tau, k_wrap: float = 21.0):
    """Dimensionless tau -> days of real time."""
    return np.asarray(tau, float) / (k_wrap * SECONDS_PER_DAY)


def days_to_tau(days, k_wrap: float = 21.0):
    return np.asarray(days, float) * k_wrap * SECONDS_PER_DAY
