"""Shared id-list path convention for reachable-nucleosome sampling.

Both select_reachable_ids.py (which WRITES the id-lists) and
generate_sweep_grid.py (which references them in sweep_grid.tsv) import this so
the per-(dataset, conc, coop) filename can never drift between the two.

A reachable id-list is keyed by (dataset, prot_p_conc, prot_cooperativity) only,
NOT by k_bind: the Markov MFPT that defines "reachable" depends solely on
c0 = k_unbind/k_bind (held fixed across the ladder), so one id-list is sampled
per (dataset, conc, coop) and reused for every k_bind rung. This is what makes
the ladder a paired comparison (same nucleosomes at k_bind = 1, 10, 100 ...).
"""

from pathlib import Path


def ids_relpath(dataset: str, conc: float, coop: float) -> Path:
    """Relative id-list path for one (dataset, conc, coop) cell.

    Floats are formatted with ``:g`` so 1.0 -> "1", 0.01 -> "0.01", 4.5 -> "4.5"
    — compact and unambiguous for the concentration/cooperativity grids in use.
    """
    return Path(dataset) / f"p{conc:g}_c{coop:g}.ids.txt"
