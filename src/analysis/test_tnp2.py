#!/usr/bin/env python3
"""Focused tests for the TNP2 J_eff extension."""

import math

import numpy as np

from src.core.ising_model import p_free, p_free_site_dependent
from src.core.nucleosomes import Nucleosome
from src.analysis.markov_solver.generator import build_full_Q_from_nucleosome
from src.analysis.markov_solver.mfpt import compute_mfpt_from_Q_TT
from src.analysis.markov_solver.tnp2 import (
    TNP2Config,
    compute_jeff_profile,
    compute_tnp2_occupancy_profile,
    count_cpg,
    count_cpg_per_site,
    get_site_ranges,
)


PROT_PARAMS = {
    "k_bind": 1.0,
    "k_unbind": 89.7,
    "p_conc": 10.0,
    "cooperativity": 4.5,
}


def _make_nucleosome(seq, binding_sites=14):
    g_mat = np.zeros((binding_sites, binding_sites), dtype=float)
    for i in range(binding_sites):
        for j in range(i, binding_sites):
            n_open = i + (binding_sites - 1 - j)
            g_mat[i, j] = -0.3 * n_open
    return Nucleosome(
        nuc_id="synthetic",
        subid=0,
        sequence=seq,
        G_mat=g_mat,
        k_wrap=1.0,
        kT=1.0,
        binding_sites=binding_sites,
    )


def _build_q(nuc, tnp2_config=None):
    return build_full_Q_from_nucleosome(
        nuc,
        k_wrap=1.0,
        protamine_params=PROT_PARAMS,
        kT=1.0,
        sparse=False,
        dimensionless=True,
        tnp2_config=tnp2_config,
    )


def _mfpt(nuc, tnp2_config=None):
    _, q_tt, _, _, state_index, _ = _build_q(nuc, tnp2_config=tnp2_config)
    mfpt, _ = compute_mfpt_from_Q_TT(q_tt, state_index, start_state=(0, 0))
    return mfpt


def test_count_cpg():
    assert count_cpg("") == 0
    assert count_cpg(None) == 0
    assert count_cpg("CGCGCG") == 3
    assert count_cpg("cgcg") == 2
    assert count_cpg("CNG") == 0
    print("  test_count_cpg: OK")


def test_site_ranges_and_counts():
    ranges = get_site_ranges()
    assert len(ranges) == 14
    assert ranges[0] == (0, 14)
    assert ranges[-1] == (139, 147)
    assert sum(end - start for start, end in ranges) == 147

    right_ranges = get_site_ranges(side="right")
    assert len(right_ranges) == 14
    assert right_ranges[0] == (132, 147)
    assert right_ranges[1] == (122, 132)
    assert right_ranges[-1] == (0, 7)
    assert sum(end - start for start, end in right_ranges) == 147

    seq = "A" * 147
    seq = seq[:14] + "CG" + seq[16:]
    counts = count_cpg_per_site(seq)
    assert counts.shape == (14,)
    assert counts[1] == 1
    assert counts.sum() == 1

    seq = "A" * 147
    seq = seq[:132] + "CG" + seq[134:]
    right_counts = count_cpg_per_site(seq, side="right")
    assert right_counts[0] == 1
    assert right_counts.sum() == 1
    print("  test_site_ranges_and_counts: OK")


def test_occupancy_and_jeff_limits():
    seq = "CG" * 73 + "C"
    p_low, cpg_counts = compute_tnp2_occupancy_profile(seq, eps_cpg=1.0, mu_t0=-100.0)
    p_high, _ = compute_tnp2_occupancy_profile(seq, eps_cpg=1.0, mu_t0=100.0)
    assert np.all(p_high > p_low)
    assert cpg_counts.sum() == count_cpg(seq)

    j_bare = 4.5
    jeff_off, _, _ = compute_jeff_profile(seq, eps_cpg=0.0, mu_t0=-100.0, j_bare=j_bare)
    jeff_on, _, _ = compute_jeff_profile(seq, eps_cpg=0.0, mu_t0=100.0, j_bare=j_bare)
    assert np.allclose(jeff_off, j_bare)
    assert np.all(jeff_on < 1e-80)
    print("  test_occupancy_and_jeff_limits: OK")


def test_site_dependent_p_free_matches_uniform():
    beta_mu = math.log(PROT_PARAMS["p_conc"] * PROT_PARAMS["k_bind"] / PROT_PARAMS["k_unbind"])
    beta_j = PROT_PARAMS["cooperativity"]
    for n in range(1, 15):
        got = p_free_site_dependent(n, beta_mu, np.full(max(n - 1, 0), beta_j))
        expected = p_free(n, beta_mu, beta_j)
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)
    print("  test_site_dependent_p_free_matches_uniform: OK")


def test_regression_v1_match_when_disabled_or_no_sequence():
    nuc = _make_nucleosome("CG" * 73 + "C")
    baseline = _mfpt(nuc, tnp2_config=None)
    disabled = _mfpt(nuc, TNP2Config(enabled=False, eps_cpg=1.0, mu_t0=100.0))
    assert math.isclose(baseline, disabled, rel_tol=1e-12)

    no_seq = _make_nucleosome(None)
    no_seq_baseline = _mfpt(no_seq, tnp2_config=None)
    no_seq_enabled = _mfpt(no_seq, TNP2Config(enabled=True, eps_cpg=1.0, mu_t0=100.0))
    assert math.isclose(no_seq_baseline, no_seq_enabled, rel_tol=1e-12)
    print("  test_regression_v1_match_when_disabled_or_no_sequence: OK")


def test_opening_rates_unchanged_and_closing_uses_jeff():
    seq = "CG" * 73 + "C"
    nuc = _make_nucleosome(seq)
    cfg = TNP2Config(enabled=True, eps_cpg=0.0, mu_t0=0.0)

    q_base, _, _, _, idx, _ = _build_q(nuc, tnp2_config=None)
    q_tnp2, _, _, _, _, _ = _build_q(nuc, tnp2_config=cfg)

    # Opening rates from the same source state are unchanged.
    src = idx[(1, 0)]
    assert math.isclose(q_base[idx[(2, 0)], src], q_tnp2[idx[(2, 0)], src], rel_tol=1e-12)
    assert math.isclose(q_base[idx[(1, 1)], src], q_tnp2[idx[(1, 1)], src], rel_tol=1e-12)

    # Closing rate changes through the p_free cooperativity term only.
    src = idx[(2, 0)]
    close_left_tnp2 = q_tnp2[idx[(1, 0)], src]
    close_left_base = q_base[idx[(1, 0)], src]
    assert not math.isclose(close_left_base, close_left_tnp2, rel_tol=1e-6)

    beta_mu = math.log(PROT_PARAMS["p_conc"] * PROT_PARAMS["k_bind"] / PROT_PARAMS["k_unbind"])
    jeff, _, _ = compute_jeff_profile(seq, eps_cpg=cfg.eps_cpg, mu_t0=cfg.mu_t0, j_bare=4.5)
    expected = p_free_site_dependent(2, beta_mu, jeff[0:1])
    assert math.isclose(close_left_tnp2, expected, rel_tol=1e-12)

    src = idx[(0, 2)]
    close_right_tnp2 = q_tnp2[idx[(0, 1)], src]
    jeff_right, _, _ = compute_jeff_profile(
        seq, eps_cpg=cfg.eps_cpg, mu_t0=cfg.mu_t0, j_bare=4.5, side="right"
    )
    expected_right = p_free_site_dependent(2, beta_mu, jeff_right[0:1])
    assert math.isclose(close_right_tnp2, expected_right, rel_tol=1e-12)
    print("  test_opening_rates_unchanged_and_closing_uses_jeff: OK")


def main():
    print("=" * 70)
    print("TNP2 J_eff Tests")
    print("=" * 70)
    test_count_cpg()
    test_site_ranges_and_counts()
    test_occupancy_and_jeff_limits()
    test_site_dependent_p_free_matches_uniform()
    test_regression_v1_match_when_disabled_or_no_sequence()
    test_opening_rates_unchanged_and_closing_uses_jeff()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
