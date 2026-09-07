"""
Single-molecule trajectories for the reduced (l, r) CTMC.

The Markov sweep stores the MFPT and S(tau); it stores no paths. This module
samples exact Gillespie paths from the same generator, so a trajectory figure and
the sweep's own numbers cannot disagree.

Two things matter for long horizons. A path to tau = 1e7 is of order 1e7 jumps, so
the jump list is never materialised: the wrapped-site count is written straight onto
the output grid and the path is forgotten. And because that path still costs seconds,
results are cached to .npz keyed by every argument that changes them.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def tau_grid(tau_max: float, n_points: int = 1000, tau_log_min: float = 1e-2,
             spacing: str = 'log') -> np.ndarray:
    """The sweep's grid convention: tau = 0 followed by log-spaced points."""
    if spacing == 'linear':
        return np.linspace(0.0, tau_max, n_points)
    return np.concatenate([[0.0], np.logspace(np.log10(tau_log_min),
                                              np.log10(tau_max), n_points - 1)])


def _walk(cum: np.ndarray, tot: np.ndarray, nb: np.ndarray, start: int,
          grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One Gillespie path, written onto `grid` without storing the jumps.

    `cum[j]` is the cumulative outgoing rate row for state j, with the absorbing
    state last. Between jumps the count is constant, so every grid point inside
    the waiting interval takes the current value.
    """
    M = cum.shape[0]
    G = len(grid)
    y = np.zeros(G)
    t, j, gi = 0.0, start, 0
    block = 1 << 16
    exps = rng.exponential(size=block)
    unis = rng.random(size=block)
    bi = 0
    while gi < G:
        if bi == block:
            exps = rng.exponential(size=block)
            unis = rng.random(size=block)
            bi = 0
        t_next = t + exps[bi] / tot[j]
        n_cur = nb[j]
        while gi < G and grid[gi] < t_next:
            y[gi] = n_cur
            gi += 1
        if gi >= G:
            break
        i = int(np.searchsorted(cum[j], unis[bi] * tot[j]))
        bi += 1
        if i >= M:              # absorbed: fully unwrapped from here on
            y[gi:] = 0.0
            break
        j, t = i, t_next
    return y


def _walk_task(args):
    cum, tot, nb, start, grid, entropy = args
    return _walk(cum, tot, nb, start, grid, np.random.default_rng(entropy))


def sample_wrapped_paths(nucleosomes: Sequence, protamine_params: Dict[str, float],
                         grid: np.ndarray, n_paths: int = 1, seed: int = 0,
                         binding_sites: int = 14, dimensionless: bool = True,
                         n_workers: int = 1) -> np.ndarray:
    """Wrapped-site count on `grid` for each (nucleosome, path).

    Returns an array of shape (len(nucleosomes) * n_paths, len(grid)), ordered
    nucleosome-major. Paths are independent, so they are handed to a process pool
    when `n_workers > 1`; each gets its own spawned seed sequence, so the result
    does not depend on the number of workers or on scheduling order.
    """
    from src.analysis.markov_solver.generator import build_full_Q_from_nucleosome

    n_tasks = len(nucleosomes) * n_paths
    seeds = np.random.SeedSequence(seed).spawn(n_tasks)
    tasks, i = [], 0
    for nuc in nucleosomes:
        _, Q_TT, Q_AT, states, sidx, _ = build_full_Q_from_nucleosome(
            nuc, k_wrap=None, sparse=False, protamine_params=protamine_params,
            dimensionless=dimensionless)
        M = len(states)
        a = np.asarray(Q_AT).ravel()[:M]
        R = np.array(Q_TT, dtype=float)
        np.fill_diagonal(R, 0.0)
        cum = np.cumsum(np.vstack([R, a[None, :]]), axis=0).T
        nb = np.array([binding_sites - l - r for l, r in states], dtype=float)
        for _ in range(n_paths):
            tasks.append((cum, cum[:, -1], nb, sidx[(0, 0)], grid, seeds[i]))
            i += 1

    if n_workers and n_workers > 1 and n_tasks > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(n_workers, n_tasks)) as pool:
            rows = list(pool.map(_walk_task, tasks))
    else:
        rows = [_walk_task(t) for t in tasks]
    return np.vstack(rows)


def cached_paths(cache_dir, dataset_dir, protamine_params: Dict[str, float],
                 grid: np.ndarray, n_nucs: int = 10, n_paths: int = 1,
                 k_wrap: float = 21.0, seed: int = 0, binding_sites: int = 14,
                 dimensionless: bool = True, rebuild: bool = False,
                 n_workers: int = 1) -> Tuple[np.ndarray, bool]:
    """`sample_wrapped_paths` with an on-disk cache. Returns (paths, was_cached)."""
    from src.analysis.markov_solver.nucleosome_utils import load_nucleosomes_from_sprm

    cache_dir = Path(cache_dir)
    dataset_dir = Path(dataset_dir)
    pp = protamine_params
    key = (f'{dataset_dir.name}__p{pp["p_conc"]:g}_c{pp["cooperativity"]:g}'
           f'_kb{pp["k_bind"]:g}_ku{pp["k_unbind"]:g}'
           f'_n{n_nucs}x{n_paths}_g{len(grid)}_t{grid[-1]:.6g}_s{seed}.npz')
    path = cache_dir / key
    if path.exists() and not rebuild:
        with np.load(path) as z:
            if np.array_equal(z['grid'], grid):
                return z['paths'], True

    nucs = load_nucleosomes_from_sprm(dataset_dir, k_wrap=k_wrap, kT=1.0,
                                      binding_sites=binding_sites, max_nucs=n_nucs)
    if len(nucs) < n_nucs:
        raise RuntimeError(f'{dataset_dir.name}: asked for {n_nucs} nucleosomes, '
                           f'got {len(nucs)}')
    paths = sample_wrapped_paths(nucs, pp, grid, n_paths=n_paths, seed=seed,
                                 binding_sites=binding_sites,
                                 dimensionless=dimensionless, n_workers=n_workers)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, paths=paths, grid=grid)
    return paths, False
