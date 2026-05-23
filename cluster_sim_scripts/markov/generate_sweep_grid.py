#!/usr/bin/env python3
"""
Generate sweep_grid.tsv from markov_sweep.yaml.

Reads the sweep specification, computes the full cross product of
(datasets x prot_p_conc x prot_cooperativity), validates that each
dataset directory exists, and writes a tab-delimited grid the SLURM
array job consumes.

Usage:
    python cluster_sim_scripts/markov/generate_sweep_grid.py
    python cluster_sim_scripts/markov/generate_sweep_grid.py --config <path>
"""

import argparse
import itertools
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=HERE / "markov_sweep.yaml",
                        help="Sweep YAML config (default: markov_sweep.yaml next to this script).")
    parser.add_argument("--output", type=Path, default=HERE / "sweep_grid.tsv",
                        help="Output TSV path (default: sweep_grid.tsv next to this script).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sprm_root = Path(cfg["sprm_root"])
    storage_root = Path(cfg["storage_root"])
    sweep = cfg["sweep"]
    datasets = sweep["datasets"]
    concs = sweep["prot_p_conc"]
    coops = sweep["prot_cooperativity"]

    missing = [d for d in datasets if not (sprm_root / d).is_dir()]
    if missing:
        print(f"ERROR: {len(missing)} dataset directory(ies) not found under {sprm_root}:",
              file=sys.stderr)
        for d in missing:
            print(f"  - {d}", file=sys.stderr)
        sys.exit(1)

    raw_rows = list(itertools.product(datasets, concs, coops))

    # Rail guard: skip (conc=0, coop>0) combinations. With no protamine present,
    # cooperativity has no effect, so those runs would be redundant duplicates
    # of (conc=0, coop=0).
    rows = [(d, c, j) for (d, c, j) in raw_rows if not (c == 0.0 and j != 0.0)]
    skipped = len(raw_rows) - len(rows)

    with open(args.output, "w") as f:
        f.write("task_id\tdataset_name\tdataset_dir\tstorage_dir\tprot_p_conc\tprot_cooperativity\n")
        for i, (dataset, conc, coop) in enumerate(rows, start=1):
            dataset_dir = sprm_root / dataset
            storage_dir = storage_root / dataset
            f.write(f"{i}\t{dataset}\t{dataset_dir}\t{storage_dir}\t{conc}\t{coop}\n")

    print(f"Wrote {len(rows)} tasks ({len(datasets)} datasets x {len(concs)} concs x {len(coops)} coops, "
          f"{skipped} skipped by conc=0/coop>0 guard) to {args.output}")


if __name__ == "__main__":
    main()
