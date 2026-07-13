"""Regression test for the absorbing-transition double-count bug in
build_full_Q_from_nucleosome (src/analysis/markov_solver/generator.py).

Bug (fixed): from a breathing state with a single remaining wrapped contact
(l + r == N - 1), BOTH "open left" and "open right" were sent to the absorbing
state. Those describe the same single contact releasing, so the final eviction
rate was doubled and every MFPT/survival was biased low by a landscape-dependent
amount (up to ~20%).

Correct behaviour: exactly one absorbing transition from each l+r == N-1 state,
at rate k_wrap * exp(-(0 - F(l,r))/kT). This mirrors the Gillespie's
`if left != right` guard in Nucleosome.unwrapping()/rewrapping().
"""
import numpy as np
from types import SimpleNamespace

from src.analysis.markov_solver import (
    build_full_Q_from_nucleosome,
    compute_mfpt_from_Q_TT,
)


def _nuc(G, N, k_wrap=1.0):
    return SimpleNamespace(G_mat=G, k_wrap=k_wrap, kT=1.0, binding_sites=N)


NO_PROT = {"k_bind": 1.0, "k_unbind": 89.7, "p_conc": 0.0, "cooperativity": 0.0}


def test_last_step_absorbing_rate_is_single_not_double():
    """N=3 flat landscape: every l+r==N-1 state has absorbing rate exactly
    k_wrap*exp(F)=1, not 2 (the old double-counted value)."""
    N = 3
    G = np.zeros((N, N))  # flat -> every open/close rate = k_wrap = 1
    Qf, _, _, states, sidx, absi = build_full_Q_from_nucleosome(
        _nuc(G, N), k_wrap=1.0, protamine_params=NO_PROT,
        sparse=False, dimensionless=True)
    last_step = [(l, r) for (l, r) in states if l + r == N - 1]
    assert last_step, "expected some l+r==N-1 states"
    for (l, r) in last_step:
        rate = Qf[absi, sidx[(l, r)]]
        assert np.isclose(rate, 1.0), (
            f"state ({l},{r}) absorbing rate = {rate} (expected 1.0; "
            f"2.0 would be the double-count bug)")


def test_generator_columns_sum_to_zero():
    """A valid CTMC generator (column convention) has every column summing to 0."""
    N = 5
    rng = np.random.default_rng(0)
    G = rng.normal(size=(N, N))
    Qf, _, _, _, _, _ = build_full_Q_from_nucleosome(
        _nuc(G, N), k_wrap=2.0, protamine_params=NO_PROT,
        sparse=False, dimensionless=True)
    assert np.allclose(Qf.sum(axis=0), 0.0, atol=1e-9)


def _reference_mfpt_single_channel(G, N, k_wrap, betamu, betaJ):
    """Independent reduced-model MFPT with a SINGLE absorbing channel from every
    l+r==N-1 state (open-left only), for cross-checking build_full_Q."""
    from src.core.ising_model import p_free

    def F(l, r):
        if l < 0 or r < 0 or l >= N or r >= N or l + r >= N:
            return 0.0
        i, j = l, (N - 1) - r
        return G[i, j] if (0 <= i < N and 0 <= j < N and i <= j) else 0.0

    states = [(l, r) for l in range(N) for r in range(N - l)]
    idx = {s: k for k, s in enumerate(states)}
    M = len(states)
    Q = np.zeros((M, M))  # column convention: Q[to, from]
    for s, (l, r) in enumerate(states):
        Fc = F(l, r)
        out = 0.0
        # closing (gated by p_free)
        if l > 0:
            rr = k_wrap * p_free(l, betamu, betaJ); Q[idx[(l-1, r)], s] += rr; out += rr
        if r > 0:
            rr = k_wrap * p_free(r, betamu, betaJ); Q[idx[(l, r-1)], s] += rr; out += rr
        # open left (single absorbing channel when it reaches l+r==N)
        rate = k_wrap * np.exp(-(F(l+1, r) - Fc))
        if l + 1 + r < N:
            Q[idx[(l+1, r)], s] += rate
        out += rate
        # open right ONLY to a transient state (never to absorbing -> no duplicate)
        if l + r + 1 < N:
            rate = k_wrap * np.exp(-(F(l, r+1) - Fc))
            Q[idx[(l, r+1)], s] += rate
            out += rate
        Q[s, s] -= out
    tau = np.linalg.solve(Q.T, -np.ones(M))
    return tau[idx[(0, 0)]]


def test_matches_independent_single_channel_reference():
    """build_full_Q MFPT must equal an independent single-absorbing-channel
    generator on a random landscape, with and without cooperative protamine."""
    N = 6
    rng = np.random.default_rng(7)
    G = rng.normal(scale=1.5, size=(N, N))
    k_wrap = 3.0
    for prot in (
        {"k_bind": 1.0, "k_unbind": 89.7, "p_conc": 10.0, "cooperativity": 0.0},
        {"k_bind": 1.0, "k_unbind": 89.7, "p_conc": 10.0, "cooperativity": 4.5},
    ):
        _, Q_TT, _, _, sidx, _ = build_full_Q_from_nucleosome(
            _nuc(G, N, k_wrap), k_wrap=k_wrap, protamine_params=prot,
            sparse=False, dimensionless=True)
        code_mfpt = compute_mfpt_from_Q_TT(Q_TT, sidx, (0, 0))[0]
        betamu = np.log(prot["p_conc"] * prot["k_bind"] / prot["k_unbind"])
        # build_full_Q factors out k_wrap (dimensionless): reference uses k_wrap=1
        ref_mfpt = _reference_mfpt_single_channel(
            G, N, 1.0, betamu, prot["cooperativity"])
        assert np.isclose(code_mfpt, ref_mfpt, rtol=1e-9), (
            f"code {code_mfpt} != reference {ref_mfpt} (coop={prot['cooperativity']})")
