#!/usr/bin/env python3
"""Sample "reachable" nucleosomes from a Markov MFPT sweep for Gillespie validation.

The event-driven Gillespie is affordable only for nucleosomes that actually
detach inside the observation window: cost per replicate scales as
(event rate) x (tau at detachment), so a stiff nucleosome that never evicts runs
to tau_max accumulating orders of magnitude more events while producing a
censored (uninformative) survival curve. This script reads the Markov MFPT
summaries — a near-free linear solve, and the very quantity the Gillespie is
validated against — and, per (dataset, conc, coop) cell, samples up to N
nucleosomes whose MFPT is below a reachability cap.

Because the Markov MFPT depends only on c0 = k_unbind/k_bind (held fixed across
the k_bind ladder), one id-list per (dataset, conc, coop) is sampled here and
reused for every k_bind rung by the sweep — a paired comparison.

Reachability is strongly cell-dependent: high-concentration cells have thousands
of reachable nucleosomes, while conc=0 / low-conc cells (intrinsic unwrapping
only) may have far fewer than N. This script takes min(N, n_reachable) and
records both counts in manifest.tsv so the shortfall is visible before you run.

Workflow (run AFTER the Markov sweep finishes, BEFORE generate_sweep_grid.py):
    python cluster_sim_scripts/gillespie_event/select_reachable_ids.py \
        --config cluster_sim_scripts/gillespie_event/gillespie_event_sweep.yaml \
        --markov_root <markov output root>   # overrides yaml markov_root
    # then: python .../generate_sweep_grid.py   (reads the id-lists)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sampling_paths import ids_relpath  # noqa: E402


# ── Pure core (unit-tested) ──────────────────────────────────────────────────
def select_ids(summary: pd.DataFrame, cap: float, n: int, seed: int):
    """Select up to ``n`` reachable subids from a Markov summary DataFrame.

    Keeps rows with mfpt_flag == 'ok', finite mfpt, and mfpt < cap, then draws a
    seeded sample of min(n, n_reachable) subids WITHOUT replacement. Returns
    (sorted_selected_subids, n_reachable). Selection is deterministic in ``seed``
    so re-running reproduces the exact set and the ladder reuse is trivially
    identical.
    """
    ok = summary["mfpt_flag"] == "ok"
    finite = np.isfinite(summary["mfpt"])
    reachable = summary[ok & finite & (summary["mfpt"] < cap)]
    subids = reachable["subid"].to_numpy()
    n_reachable = len(subids)

    take = min(n, n_reachable)
    if take == 0:
        return [], 0
    rng = np.random.default_rng(seed)
    chosen = rng.choice(subids, size=take, replace=False)
    return sorted(int(x) for x in chosen), n_reachable


def _seed_for(seed_base: int, dataset: str, conc: float, coop: float) -> int:
    """Stable per-cell seed so each (dataset, conc, coop) samples reproducibly."""
    key = f"{seed_base}|{dataset}|{conc:g}|{coop:g}"
    return int.from_bytes(__import__("hashlib").sha256(key.encode()).digest()[:8], "big")


# ── IO / orchestration ───────────────────────────────────────────────────────
def _index_markov_cells(markov_root: Path, dataset: str):
    """Map (conc, coop) -> summary TSV path by reading each cell's parameters.json.

    Robust to the tau_max/hash portion of the cell directory name changing
    between the template (tau10000) and production (tau100000) sweeps.
    """
    cells = {}
    for pj in sorted((markov_root / dataset).glob("*/parameters.json")):
        params = json.loads(pj.read_text())
        pp = params["prot_params"]
        conc = float(pp["p_conc"])
        coop = float(pp["cooperativity"])
        summ = pj.parent / "summaries" / f"{dataset}.tsv"
        if summ.exists():
            cells[(conc, coop)] = summ
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=HERE / "gillespie_event_sweep.yaml")
    ap.add_argument("--markov_root", type=Path, default=None,
                    help="Markov sweep output root (overrides yaml markov_root).")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where id-lists are written (overrides yaml ids_root).")
    ap.add_argument("--n", type=int, default=None, help="Target nucleosomes per cell.")
    ap.add_argument("--cap", type=float, default=None,
                    help="Reachability cap: keep nucleosomes with Markov mfpt < cap.")
    ap.add_argument("--seed_base", type=int, default=20250713)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sweep = cfg["sweep"]
    datasets = sweep["datasets"]
    concs = [float(c) for c in sweep["prot_p_conc"]]
    coops = [float(c) for c in sweep["prot_cooperativity"]]

    markov_root = args.markov_root or Path(cfg["markov_root"])
    out_dir = args.out_dir or Path(cfg["ids_root"])
    n = args.n if args.n is not None else int(cfg.get("sampling_n", 100))
    cap = args.cap if args.cap is not None else float(cfg.get("sampling_cap", 5000.0))

    if not markov_root.is_dir():
        sys.exit(f"ERROR: markov_root not found: {markov_root}")

    print(f"markov_root = {markov_root}")
    print(f"out_dir     = {out_dir}")
    print(f"n = {n}   cap = {cap:g}\n")

    manifest = []
    kbind0_conc_collapsed = 0
    for dataset in datasets:
        cells = _index_markov_cells(markov_root, dataset)
        for conc, coop in itertools.product(concs, coops):
            # Same conc=0 collapse the sweep uses: no protamine -> coop irrelevant.
            if conc == 0.0 and coop != 0.0:
                kbind0_conc_collapsed += 1
                continue
            summ = cells.get((conc, coop))
            rel = ids_relpath(dataset, conc, coop)
            if summ is None:
                print(f"  WARN: no Markov cell for {dataset} conc={conc:g} coop={coop:g}",
                      file=sys.stderr)
                manifest.append((dataset, conc, coop, -1, 0, str(rel)))
                continue
            df = pd.read_csv(summ, sep="\t")
            seed = _seed_for(args.seed_base, dataset, conc, coop)
            ids, n_reach = select_ids(df, cap=cap, n=n, seed=seed)

            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("".join(f"{i}\n" for i in ids))
            flag = "" if len(ids) >= n else f"  <-- SHORT ({len(ids)}/{n})"
            print(f"  {dataset[:28]:28s} conc={conc:<7g} coop={coop:<4g} "
                  f"reachable={n_reach:6d} selected={len(ids):4d}{flag}")
            manifest.append((dataset, conc, coop, n_reach, len(ids), str(rel)))

    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.tsv"
    with open(man_path, "w") as f:
        f.write("dataset\tprot_p_conc\tprot_cooperativity\tn_reachable\tn_selected\tids_file\n")
        for row in manifest:
            f.write("\t".join(str(x) for x in row) + "\n")

    short = [m for m in manifest if m[4] < n]
    print(f"\nWrote {len(manifest)} id-lists + {man_path}")
    print(f"({kbind0_conc_collapsed} conc=0/coop>0 cells collapsed, matching the sweep)")
    if short:
        print(f"{len(short)} cell(s) below N={n} (mostly low/zero conc — expected):")
        for d, c, j, nr, ns, _ in short:
            print(f"    {d[:28]:28s} conc={c:<7g} coop={j:<4g} -> {ns} "
                  f"(reachable={nr if nr >= 0 else 'MISSING'})")


if __name__ == "__main__":
    main()
