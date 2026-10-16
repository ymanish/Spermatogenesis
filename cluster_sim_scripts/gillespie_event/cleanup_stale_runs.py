#!/usr/bin/env python3
"""Delete stale Gillespie run directories for a superseded tau_max.

The run directory is named ``k{k_wrap}_p{conc}_c{coop}_inf__{hash}`` where the
hash covers (k_wrap, inf_protamine, tau_max, tau_steps, prot_params,
binding_sites) -- see src/config/storage.py. Changing tau_max therefore moves
the output to a NEW directory and leaves the old one behind.

Deleting purely by old hash is unsafe: a superseded sweep usually contains some
COMPLETED cells, and those results are real (and, for a larger tau_max, cover a
longer window than the run replacing them). This script classifies each old
directory first and by default removes only the incomplete ones -- the tasks
that were cancelled or killed, which wrote no merged output because
orchestrator.py merges only after every batch finishes.

DRY RUN BY DEFAULT. Nothing is deleted unless --apply is passed.

    # see what exists and how it classifies
    python cluster_sim_scripts/gillespie_event/cleanup_stale_runs.py --old_tau_max 1e6

    # delete only the incomplete ones
    python .../cleanup_stale_runs.py --old_tau_max 1e6 --apply

    # delete completed ones too (destroys real results -- be sure)
    python .../cleanup_stale_runs.py --old_tau_max 1e6 --include_complete --apply

    # locally: the yaml storage_root is the cluster path, so pass the local copy
    python .../cleanup_stale_runs.py --old_tau_max 1e6 \\
        --storage_root output/gillespie_event_sweep_stable147_v1_two_adsorption_gth
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sampling_paths import resolve_datasets  # noqa: E402


def param_hash(conc, coop, k_bind, k_unbind, tau_max, tau_steps,
               k_wrap, binding_sites, inf_protamine, length=6):
    """Reproduce SimulationStorage._get_param_hash exactly."""
    repr_ = {
        "k_wrap": k_wrap,
        "inf_protamine": inf_protamine,
        "tau_max": tau_max,
        "tau_steps": tau_steps,
        "prot_params": {"k_unbind": k_unbind, "k_bind": k_bind,
                        "p_conc": conc, "cooperativity": coop},
        "binding_sites": binding_sites,
    }
    return hashlib.sha256(
        json.dumps(repr_, sort_keys=True).encode()
    ).hexdigest()[:length]


def run_dirname(conc, coop, k_bind, k_unbind, tau_max, **kw):
    h = param_hash(conc, coop, k_bind, k_unbind, tau_max, **kw)
    return f"k{kw['k_wrap']:.1f}_p{conc:.1f}_c{coop:.1f}_inf__{h}"


def classify(d: Path, dataset: str):
    """complete if the merged per-dataset output landed; else incomplete."""
    if not d.is_dir():
        return "absent", 0
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    done = (d / "survival" / f"{dataset}.parquet").exists()
    return ("complete" if done else "incomplete"), size


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=HERE / "gillespie_event_sweep.yaml")
    ap.add_argument("--grid", type=Path, default=HERE / "sweep_grid.tsv")
    ap.add_argument("--old_tau_max", type=float, required=True,
                    help="tau_max of the SUPERSEDED run whose dirs you want gone.")
    ap.add_argument("--storage_root", type=Path, default=None,
                    help="Overrides yaml storage_root.")
    ap.add_argument("--include_complete", action="store_true",
                    help="Also delete directories that hold finished results.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this it is a dry run.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    root = args.storage_root or Path(cfg["storage_root"])
    cur_tau = float(cfg["tau_max"])
    if abs(cur_tau - args.old_tau_max) < 1e-9:
        sys.exit(f"REFUSING: --old_tau_max {args.old_tau_max:g} equals the CURRENT "
                 f"config tau_max. That would delete the run you are about to do.")

    common = dict(tau_steps=int(cfg["tau_steps"]), k_wrap=float(cfg.get("k_wrap", 1.0)),
                  binding_sites=int(cfg["binding_sites"]),
                  inf_protamine=bool(cfg["inf_protamine"]))

    rows = list(csv.DictReader(args.grid.open(), delimiter="\t"))
    datasets = set(resolve_datasets(cfg))

    print(f"storage_root = {root}")
    print(f"old tau_max  = {args.old_tau_max:g}   (current config tau_max = {cur_tau:g})")
    print(f"mode         = {'APPLY (deleting)' if args.apply else 'DRY RUN'}\n")

    seen, targets = set(), []
    for r in rows:
        ds = r["dataset_name"]
        if ds not in datasets:
            continue
        conc, coop = float(r["prot_p_conc"]), float(r["prot_cooperativity"])
        kb, ku = float(r["prot_k_bind"]), float(r["prot_k_unbind"])
        name = run_dirname(conc, coop, kb, ku, args.old_tau_max, **common)
        d = root / ds / name
        if d in seen:
            continue
        seen.add(d)
        state, size = classify(d, ds)
        targets.append((d, ds, conc, coop, float(r["prot_k_bind_phys"]), state, size))

    for d, ds, conc, coop, kbp, state, size in sorted(targets, key=lambda x: x[5]):
        mark = {"complete": "KEEP ", "incomplete": "DEL  ", "absent": "  -  "}[state]
        if state == "complete" and args.include_complete:
            mark = "DEL* "
        print(f"  {mark} {state:10s} {size/2**20:9.1f} MB  {ds[:22]:22s} "
              f"p={conc:<6g} c={coop:<4g} kb={kbp:<5g} {d.name}")

    todel = [t for t in targets if t[5] == "incomplete"
             or (t[5] == "complete" and args.include_complete)]
    freed = sum(t[6] for t in todel)
    print(f"\n{len(todel)} directory(ies) to delete, {freed/2**30:.2f} GB")
    kept = [t for t in targets if t[5] == "complete" and not args.include_complete]
    if kept:
        print(f"{len(kept)} completed directory(ies) KEPT (pass --include_complete "
              f"to remove them too).")

    if not args.apply:
        print("\nDRY RUN - nothing deleted. Re-run with --apply.")
        return
    for d, *_ in todel:
        shutil.rmtree(d)
        print(f"  deleted {d}")
    print(f"\nDeleted {len(todel)} directory(ies).")


if __name__ == "__main__":
    main()
