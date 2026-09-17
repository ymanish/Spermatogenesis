"""Free-energy landscape of nucleosome breathing, with and without protamine.

The sweep integrates a CTMC on the transient states (l, r), l + r < N. This module
reduces that 2D chain to a free-energy profile along the unwrapping coordinate

    n = l + r        (number of unwrapped contacts, 0 ... N)

so the kinetics elsewhere in the notebook can be read off a landscape.

Why a landscape exists at all
----------------------------
The rates are not obviously derived from a potential: opening uses bare nucleosome
free-energy differences,

    k_open(l, r -> l+1, r) = k_wrap * exp(-[F(l+1,r) - F(l,r)] / kT),

while closing is gated by the protamine lattice gas on the exposed arm,

    k_close(l, r -> l-1, r) = k_wrap * p_free(l).

They are, though. Every elementary square of the (l, r) lattice satisfies
Kolmogorov's criterion: going round (l,r) -> (l+1,r) -> (l+1,r+1) -> (l,r+1) ->
(l,r) the F terms telescope and the two p_free factors are the same in both
directions, so the product of rates round the loop is the same either way. The
transient chain is therefore reversible and has an equilibrium measure.

Writing detailed balance for one step, pi(l+1,r) / pi(l,r) = k_open / k_close =
exp(-[F(l+1,r) - F(l,r)]) / p_free(l+1), and using the transfer-matrix identity

    p_free(n) = Z_{n-1} / Z_n

(the first row of T equals the boundary vector b, so (T^{n-1} b)_0 = b^T T^{n-2} b),
the product telescopes and

    G(l, r) = F_nuc(l, r) - kT ln Z_l - kT ln Z_r,        pi ~ exp(-G / kT).

The protamine term is just the grand potential of the 1D lattice gas on each
exposed arm: -kT ln Z_n. It is sequence-independent, so protamine tilts every
nucleosome's landscape by exactly the same amount; datasets differ only through
where their bare barrier already sits. `check_detailed_balance` verifies G against
the generator the sweep actually integrates, edge by edge.

The profile along n is then the exact potential of mean force,

    F(n) = -kT ln sum_{l+r=n} exp(-G(l, r) / kT).

The zero
--------
energies.tsv stores dF_total = (F - F_freeDNA) + adsorption, so both terms vanish
for fully unwrapped DNA and the released state is zero by construction. Profiles
are reported on that zero, which makes them absolute free energies against free
DNA; `reference='wrapped'` shifts instead to F(0) = 0, which puts the kinetic
barrier straight on the axis.

The released state
------------------
n = N carries F_nuc = 0, so only the protamine term survives and the freed DNA is
one coated chain: F(N) = -kT ln Z_N (`released_level`). With no protamine that is
exactly zero, the convention energies.tsv is written in, and the profile runs
continuously into it.

One caveat on that point. It is absorbing, so the chain never leaves it, no
detailed balance fixes its free energy, and nothing in the kinetics depends on it
either -- the final opening rate is k_wrap * exp(F_nuc(l,r)/kT), with the bare
F_nuc and no p_free factor. Read n = N as where the landscape is heading, not as a
barrier height.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

N_SITES = 14


def transient_states(N: int = N_SITES) -> List[Tuple[int, int]]:
    """The (l, r) states with l + r < N, in the generator's own order."""
    return [(l, r) for l in range(N) for r in range(N - l)]


def beta_params(prot_params: Dict[str, float], kT: float = 1.0) -> Tuple[float, float]:
    """(beta*mu, beta*J) from a sweep's prot_params, matching generator.py."""
    if prot_params['p_conc'] <= 0.0:
        return -np.inf, 0.0
    return (np.log(prot_params['p_conc'] * prot_params['k_bind']
                   / prot_params['k_unbind']),
            prot_params['cooperativity'] / kT)


def log_partition(n_max: int, betamu: float, betaJ: float) -> np.ndarray:
    """ln Z_n for an open chain of n sites, n = 0 ... n_max.

    Z_n = b^T T^{n-1} b with the transfer matrix of src.core.ising_model. Carried
    forward with the vector rescaled each step, so large beta*mu cannot overflow.
    """
    lnZ = np.zeros(n_max + 1)
    if betamu == -np.inf:                      # no protamine: Z_n = 1
        return lnZ
    ef2 = np.exp(0.5 * betamu)
    T = np.array([[1.0, ef2], [ef2, np.exp(betamu + betaJ)]])
    b = np.array([1.0, ef2])
    v, shift = b.copy(), 0.0                   # v = T^{n-1} b, up to exp(shift)
    for n in range(1, n_max + 1):
        if n > 1:
            v = T @ v
            s = v.max()
            v, shift = v / s, shift + np.log(s)
        lnZ[n] = np.log(b @ v) + shift
    return lnZ


def load_state_energies(dataset_dir, N: int = N_SITES, cache=None,
                        rebuild: bool = False) -> np.ndarray:
    """Bare F_nuc(l, r) for every sequence in an SPRM dataset.

    Returns an (n_sequences, n_states) array whose columns follow
    `transient_states(N)`. energies.tsv is ~60 MB per dataset, so the parsed array
    is cached as an .npy when `cache` is given.
    """
    cache = Path(cache) if cache is not None else None
    if cache is not None and cache.exists() and not rebuild:
        return np.load(cache)

    states = transient_states(N)
    index = np.full((N, N), -1, dtype=np.int32)
    for k, (l, r) in enumerate(states):
        index[l, r] = k

    df = pd.read_csv(Path(dataset_dir) / 'energies.tsv', sep='\t',
                     usecols=['global_id', 'left_open', 'right_open', 'dF_total'],
                     dtype={'global_id': np.int64, 'left_open': np.int16,
                            'right_open': np.int16, 'dF_total': np.float64})
    col = index[df.left_open.to_numpy(), df.right_open.to_numpy()]
    row, ids = pd.factorize(df.global_id.to_numpy(), sort=False)
    if (col < 0).any():
        raise RuntimeError(f'{dataset_dir}: rows outside l + r < {N}')

    F = np.full((len(ids), len(states)), np.nan)
    F[row, col] = df.dF_total.to_numpy()
    if np.isnan(F).any():
        bad = int(np.isnan(F).any(1).sum())
        raise RuntimeError(f'{dataset_dir}: {bad} sequences miss some (l, r) state')
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, F)
    return F


def effective_energies(F: np.ndarray, betamu: float, betaJ: float,
                       N: int = N_SITES, kT: float = 1.0) -> np.ndarray:
    """G(l, r) = F_nuc(l, r) - kT [ln Z_l + ln Z_r], broadcast over sequences."""
    lnZ = log_partition(N, betamu, betaJ)
    tilt = np.array([-kT * (lnZ[l] + lnZ[r]) for l, r in transient_states(N)])
    return F + tilt


def released_level(betamu: float, betaJ: float, N: int = N_SITES, kT: float = 1.0,
                   mode: str = 'chain') -> float:
    """F(N) for fully released DNA. F_nuc = 0 there, so only protamine is left.

    mode='chain' (default): the freed N-site DNA coated as one chain, -kT ln Z_N.
        With no protamine this is exactly zero, which is the convention energies.tsv
        is written in, and it lets protamines cooperate across the dyad.
    mode='shell': the same shell sum as every other n, -kT ln sum_{l+r=N} Z_l Z_r.
        Uniform in formula, but it counts the N+1 left/right splits of one physical
        state as N+1 states, which costs -kT ln(N+1) = -2.71 kT even at c = 0.

    The two differ by 2.71 kT at J = 0 and 0.78 kT at c = 50 uM, J = 4.5: the size
    of the convention at the one point of the landscape that is not pinned down.
    """
    lnZ = log_partition(N, betamu, betaJ)
    if mode == 'chain':
        return -kT * lnZ[N]
    if mode != 'shell':
        raise ValueError(f"mode must be 'chain' or 'shell', got {mode!r}")
    x = np.array([lnZ[l] + lnZ[N - l] for l in range(N + 1)])
    return -kT * (x.max() + np.log(np.exp(x - x.max()).sum()))


def shell_profile(G: np.ndarray, N: int = N_SITES, kT: float = 1.0,
                  reference: str = 'released', released: Optional[float] = None
                  ) -> np.ndarray:
    """F(n) = -kT ln sum_{l+r=n} exp(-G/kT), per sequence.

    Covers n = 0 ... N-1, plus the released shell n = N when `released` is given
    (see `released_shell`), which keeps the profile continuous to the end.

    reference : 'released' keeps the zero energies.tsv is written in -- dF_total is
        measured against free DNA, so bare fully unwrapped DNA is zero by
        construction and every curve is an absolute free energy on that scale.
        'wrapped' shifts each sequence so its fully wrapped state is zero, which
        puts the barrier straight on the axis.
    """
    if reference not in ('released', 'wrapped'):
        raise ValueError(f"reference must be 'released' or 'wrapped', got {reference!r}")
    shells = [np.array([k for k, (l, r) in enumerate(transient_states(N))
                        if l + r == n]) for n in range(N)]
    out = np.empty((G.shape[0], N))
    for n, cols in enumerate(shells):
        x = -G[:, cols] / kT
        m = x.max(axis=1, keepdims=True)
        out[:, n] = -kT * (m[:, 0] + np.log(np.exp(x - m).sum(axis=1)))
    if released is not None:
        out = np.hstack([out, np.full((out.shape[0], 1), released)])
    if reference == 'wrapped':
        out = out - out[:, [0]]
    return out


def dataset_profile(dataset_dir, prot_params: Dict[str, float], N: int = N_SITES,
                    kT: float = 1.0, quantiles=(0.25, 0.5, 0.75), cache_dir=None,
                    reference: str = 'released', include_released: bool = True,
                    released_mode: str = 'chain') -> Dict[str, np.ndarray]:
    """Landscape statistics across the sequences of one dataset, at one condition.

    Returns n (0 ... N), the mean and sample standard deviation of F(n) across
    sequences, the requested quantiles as an array of shape (len(quantiles), N+1),
    the released level, and n_seq. The spread either way is heterogeneity between
    sequences, not an uncertainty on the mean. Pass include_released=False to stop
    at the last reversible shell, N-1.
    """
    dataset_dir = Path(dataset_dir)
    cache = None if cache_dir is None else Path(cache_dir) / f'{dataset_dir.name}.npy'
    F = load_state_energies(dataset_dir, N=N, cache=cache)
    betamu, betaJ = beta_params(prot_params, kT)
    rel = (released_level(betamu, betaJ, N, kT, released_mode)
           if include_released else None)
    prof = shell_profile(effective_energies(F, betamu, betaJ, N, kT), N, kT,
                         reference, rel)
    q = np.asarray(quantiles, float)
    return dict(n=np.arange(prof.shape[1]),
                mean=prof.mean(axis=0),
                sd=prof.std(axis=0, ddof=1),
                q=np.quantile(prof, q, axis=0),
                released=rel,
                released_mode=released_mode,
                reference=reference,
                quantiles=q,
                n_seq=F.shape[0])


def barrier(n: np.ndarray, profile: np.ndarray) -> Tuple[float, int]:
    """(height, position) of the highest point of a reversible profile."""
    i = int(np.argmax(profile))
    return float(profile[i]), int(n[i])


def check_detailed_balance(nucleosome, prot_params: Dict[str, float],
                           k_wrap: float = 21.0, kT: float = 1.0,
                           N: int = N_SITES) -> float:
    """Largest |ln| violation of pi ~ exp(-G) against the generator the sweep uses.

    Builds Q for one nucleosome and checks pi_i Q_ji = pi_j Q_ij on every transient
    edge. Returns the worst absolute log ratio; it should be at round-off.
    """
    from .generator import build_full_Q_from_nucleosome

    _, Q_TT, _, states, _, _ = build_full_Q_from_nucleosome(
        nucleosome, k_wrap=k_wrap, protamine_params=prot_params, kT=kT,
        binding_sites=N, dimensionless=True)
    Q_TT = np.asarray(Q_TT)
    F = np.array([[nucleosome.G_mat[l, (N - 1) - r] for l, r in states]])
    betamu, betaJ = beta_params(prot_params, kT)
    logpi = -effective_energies(F, betamu, betaJ, N, kT)[0] / kT

    worst = 0.0
    for j in range(len(states)):
        for i in range(len(states)):
            if i == j or Q_TT[i, j] <= 0 or Q_TT[j, i] <= 0:
                continue
            # pi_j q_{j->i} = pi_i q_{i->j}, with Q[i, j] = rate(j -> i).
            d = abs((logpi[j] + np.log(Q_TT[i, j])) - (logpi[i] + np.log(Q_TT[j, i])))
            worst = max(worst, d)
    return worst
