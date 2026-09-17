"""Shared dataset-name and id-list path conventions for reachable sampling.

Both select_reachable_ids.py (which WRITES the id-lists) and
generate_sweep_grid.py (which references them in sweep_grid.tsv) import this so
the dataset name and the per-(dataset, conc, coop) filename can never drift
between the two.

A reachable id-list is keyed by (dataset, prot_p_conc, prot_cooperativity) only,
NOT by k_bind: the Markov MFPT that defines "reachable" depends solely on
c0 = k_unbind/k_bind (held fixed across the ladder), so one id-list is sampled
per (dataset, conc, coop) and reused for every k_bind rung. This is what makes
the ladder a paired comparison (same nucleosomes at k_bind = 1, 10, 100 ...).
"""

from pathlib import Path


def resolve_datasets(cfg: dict) -> list:
    """Expand ``{E_out:.2f}``-style placeholders in ``cfg['sweep']['datasets']``.

    Dataset directory names carry the adsorption energies they were generated
    with, so the sweep YAML templates them against its own top-level ``E_out`` /
    ``E_in`` and stays the single source of truth when those change.

    EVERY consumer must resolve them this way. An unexpanded name matches no
    directory on disk and therefore reports as missing DATA (empty cell grids,
    ``reachable=MISSING``) rather than as the config error it actually is — so a
    bad template raises here instead of propagating a literal-brace path.
    """
    try:
        return [dataset.format_map(cfg) for dataset in cfg["sweep"]["datasets"]]
    except (KeyError, ValueError, IndexError) as exc:
        raise SystemExit(
            f"ERROR: invalid dataset template: {exc!r}. Dataset names may only "
            f"reference top-level keys of the sweep YAML (e.g. {{E_out:.2f}})."
        )


def ids_relpath(dataset: str, conc: float, coop: float) -> Path:
    """Relative id-list path for one (dataset, conc, coop) cell.

    Floats are formatted with ``:g`` so 1.0 -> "1", 0.01 -> "0.01", 4.5 -> "4.5"
    — compact and unambiguous for the concentration/cooperativity grids in use.
    """
    return Path(dataset) / f"p{conc:g}_c{coop:g}.ids.txt"
