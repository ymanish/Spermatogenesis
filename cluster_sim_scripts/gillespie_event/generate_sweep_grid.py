#!/usr/bin/env python3
"""
Generate sweep_grid.tsv from gillespie_event_sweep.yaml.

Reads the sweep specification, computes the full cross product of
(datasets x prot_p_conc x prot_cooperativity), validates that each
dataset directory exists, and writes a tab-delimited grid the SLURM
array job consumes.

Usage:
    python cluster_sim_scripts/gillespie_event/generate_sweep_grid.py
    python cluster_sim_scripts/gillespie_event/generate_sweep_grid.py --config <path>
    python cluster_sim_scripts/gillespie_event/generate_sweep_grid.py --no-validate
"""

import argparse
import itertools
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=HERE / "gillespie_event_sweep.yaml",
                        help="Sweep YAML config (default: gillespie_event_sweep.yaml next to this script).")
    parser.add_argument("--output", type=Path, default=HERE / "sweep_grid.tsv",
                        help="Output TSV path (default: sweep_grid.tsv next to this script).")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the dataset_dir existence check. Use when generating "
                             "offline (e.g., on a laptop) with paths that only resolve on "
                             "the cluster.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sprm_root = Path(cfg["sprm_root"])
    storage_root = Path(cfg["storage_root"])
    sweep = cfg["sweep"]
    datasets = sweep["datasets"]
    concs = sweep["prot_p_conc"]
    coops = sweep["prot_cooperativity"]
    kbinds_phys = sweep["prot_k_bind_phys"]

    # Dimensionless conversion (pipeline runs at k_wrap=1):
    #   k_bind_run   = k_bind_phys / k_wrap_phys
    #   k_unbind_run = c0 * k_bind_run
    # keeps c0 = k_unbind/k_bind fixed while sweeping the on-rate toward the fast limit.
    k_wrap_phys = float(cfg["k_wrap_phys"])
    c0 = float(cfg["prot_c0"])

    if args.no_validate:
        print(f"WARN: --no-validate set; not checking that dataset_dirs exist under {sprm_root}",
              file=sys.stderr)
    else:
        missing = [d for d in datasets if not (sprm_root / d).is_dir()]
        if missing:
            print(f"ERROR: {len(missing)} dataset directory(ies) not found under {sprm_root}:",
                  file=sys.stderr)
            for d in missing:
                print(f"  - {d}", file=sys.stderr)
            print("Pass --no-validate to skip this check (e.g., generating offline for "
                  "cluster paths).", file=sys.stderr)
            sys.exit(1)

    raw_rows = list(itertools.product(datasets, concs, coops, kbinds_phys))

    # Rail guards (both drop physically redundant runs):
    #   1. (conc=0, coop>0): with no protamine, cooperativity has no effect.
    #   2. (conc=0, k_bind != ladder[0]): with no protamine, k_bind is irrelevant,
    #      so the ladder collapses to a single run per (dataset, conc=0, coop=0).
    kbind0 = kbinds_phys[0]
    rows = [
        (d, c, j, kb) for (d, c, j, kb) in raw_rows
        if not (c == 0.0 and j != 0.0)
        and not (c == 0.0 and kb != kbind0)
    ]
    skipped = len(raw_rows) - len(rows)

    with open(args.output, "w") as f:
        f.write("task_id\tdataset_name\tdataset_dir\tstorage_dir\tprot_p_conc\t"
                "prot_cooperativity\tprot_k_bind\tprot_k_unbind\tprot_k_bind_phys\n")
        for i, (dataset, conc, coop, kb_phys) in enumerate(rows, start=1):
            dataset_dir = sprm_root / dataset
            storage_dir = storage_root / dataset
            k_bind_run = kb_phys / k_wrap_phys
            k_unbind_run = c0 * k_bind_run
            f.write(f"{i}\t{dataset}\t{dataset_dir}\t{storage_dir}\t{conc}\t{coop}\t"
                    f"{k_bind_run:.10g}\t{k_unbind_run:.10g}\t{kb_phys:g}\n")

    print(f"Wrote {len(rows)} tasks ({len(datasets)} datasets x {len(concs)} concs x "
          f"{len(coops)} coops x {len(kbinds_phys)} k_bind, {skipped} skipped by "
          f"conc=0 guards) to {args.output}")


if __name__ == "__main__":
    main()
