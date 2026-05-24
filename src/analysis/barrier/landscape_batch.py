"""
Batch computation of energy-landscape descriptors across nucleosome cohorts.

Typical usage
-------------
    from pathlib import Path
    from src.analysis.barrier.landscape_batch import compare_cohorts
    from src.analysis.barrier.drift_reversal_plots import plot_landscape_descriptor_comparison

    df = compare_cohorts({
        'ret_single_nuc': Path('/path/to/ret_single_nuc_sprm'),
        'ctrl04':         Path('/path/to/ctrl04_sprm'),
    }, k_wrap=1.0, kT=1.0)

    print(df.groupby('cohort')[['dE_firstbreath', 'dE_barrier', 'sigma_dE']].describe())
    plot_landscape_descriptor_comparison(df)
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .drift_reversal import DriftReversalAnalyzer

# Dummy protamine params — descriptors are purely G_mat observables
_NO_PROT: Dict[str, float] = {
    'k_bind': 1.0,
    'k_unbind': 1.0,
    'p_conc': 0.0,
    'cooperativity': 0.0,
}


def compute_cohort_descriptors(
    sprm_dir: Path,
    k_wrap: float = 1.0,
    kT: float = 1.0,
    binding_sites: int = 14,
    max_nucs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute landscape descriptors for every nucleosome in an SPRM dataset.

    Parameters
    ----------
    sprm_dir : Path
        Directory containing ``energies.tsv`` and ``id_lookup.tsv``
        (SPRM format).
    k_wrap : float
        Wrapping rate (used only for Analyzer init; descriptors are
        k_wrap-independent in dimensionless units).
    kT : float
        Thermal energy in the same units as G_mat values.
    binding_sites : int
        Number of DNA binding sites (default 14).
    max_nucs : int, optional
        Cap the number of nucleosomes loaded (useful for quick tests).

    Returns
    -------
    pd.DataFrame
        Columns: ``id``, ``subid``, ``dE_firstbreath``, ``dE_barrier``,
        ``sigma_dE``.
    """
    from src.analysis.markov_solver.nucleosome_utils import load_nucleosomes_from_sprm

    nucs = load_nucleosomes_from_sprm(
        sprm_dir,
        k_wrap=k_wrap,
        kT=kT,
        binding_sites=binding_sites,
        max_nucs=max_nucs,
    )

    rows = []
    for nuc in nucs:
        analyzer = DriftReversalAnalyzer(
            nuc,
            k_wrap=k_wrap,
            kT=kT,
            binding_sites=binding_sites,
            protamine_params=_NO_PROT,
        )
        ld = analyzer.compute_landscape_descriptors()
        rows.append({'id': nuc.id, 'subid': nuc.subid, **ld})

    return pd.DataFrame(rows)


def compare_cohorts(
    cohort_dirs: Dict[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load multiple SPRM cohorts and return a combined DataFrame.

    Parameters
    ----------
    cohort_dirs : dict
        Mapping of cohort name → SPRM directory path.
        Example: ``{'ret_single_nuc': Path(...), 'ctrl04': Path(...)}``.
    **kwargs
        Forwarded to ``compute_cohort_descriptors`` (k_wrap, kT,
        binding_sites, max_nucs).

    Returns
    -------
    pd.DataFrame
        Like ``compute_cohort_descriptors`` but with an extra ``cohort``
        column identifying the source group.
    """
    frames = []
    for name, d in cohort_dirs.items():
        df = compute_cohort_descriptors(Path(d), **kwargs)
        df['cohort'] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_batch_drift_reversal(
    sprm_dir: Path,
    protamine_params: Optional[Dict] = None,
    k_wrap: float = 1.0,
    kT: float = 1.0,
    binding_sites: int = 14,
    max_nucs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full drift-reversal analysis for every nucleosome in an SPRM
    dataset and return a summary DataFrame.

    This computes all scalar outputs from ``DriftReversalAnalyzer.analyze()``:
    MFPT, quasi-potential barrier, critical nucleus, committor at n*, and the
    three landscape descriptors.  Survival curves and per-shell data are not
    stored (call ``analyze()`` directly if you need them).

    Parameters
    ----------
    sprm_dir : Path
        SPRM dataset directory (``energies.tsv`` + ``id_lookup.tsv``).
    protamine_params : dict, optional
        Keys: k_bind, k_unbind, p_conc, cooperativity.
        Defaults to no protamine (p_conc = 0).
    k_wrap : float
        Wrapping rate.
    kT : float
        Thermal energy.
    binding_sites : int
        Number of binding sites.
    max_nucs : int, optional
        Load at most this many nucleosomes (for quick tests).

    Returns
    -------
    pd.DataFrame
        One row per nucleosome.  Columns:
        ``id``, ``subid``,
        ``mfpt_1d``, ``mfpt_nucleation``,
        ``n_star``, ``n_star_refined``, ``delta_phi``,
        ``committor_at_nstar``,
        ``dE_firstbreath``, ``dE_barrier``, ``sigma_dE``.
    """
    from src.analysis.markov_solver.nucleosome_utils import load_nucleosomes_from_sprm

    if protamine_params is None:
        protamine_params = _NO_PROT

    nucs = load_nucleosomes_from_sprm(
        sprm_dir,
        k_wrap=k_wrap,
        kT=kT,
        binding_sites=binding_sites,
        max_nucs=max_nucs,
    )

    rows = []
    for nuc in nucs:
        analyzer = DriftReversalAnalyzer(
            nuc,
            k_wrap=k_wrap,
            kT=kT,
            binding_sites=binding_sites,
            protamine_params=protamine_params,
        )
        res = analyzer.analyze()

        # Committor value at n* (None if no critical nucleus found)
        committor_nstar = (
            float(res.committor[res.n_star])
            if res.n_star is not None and res.n_star < len(res.committor)
            else None
        )

        rows.append({
            'id':                nuc.id,
            'subid':             nuc.subid,
            # 1D kinetics
            'mfpt_1d':           res.mfpt_1d,
            'mfpt_nucleation':   res.mfpt_nucleation,
            # Landscape shape
            'n_star':            res.n_star,
            'n_star_refined':    res.n_star_refined,
            'delta_phi':         res.delta_phi,
            'committor_at_nstar': committor_nstar,
            # Sequence-intrinsic descriptors
            'dE_firstbreath':    res.dE_firstbreath,
            'dE_barrier':        res.dE_barrier,
            'sigma_dE':          res.sigma_dE,
        })

    return pd.DataFrame(rows)
