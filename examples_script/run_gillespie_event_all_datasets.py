"""Driver: run the event-driven Gillespie pipeline over the six target
SPRM datasets, one CLI invocation per dataset. Continues past per-dataset
failures.
"""

import subprocess
import sys
import time
from pathlib import Path

from src.config.path import SPRM_DATA_DIR


DATASETS = [
    "ctrl01_random_genome_safe_stable147_refined",
    "ctrl02_random_genome_gcmatched_stable147_refined",
    "ctrl03_som_gcmatched_stable147_refined",
    "ctrl04_bound_prom_evicted_stable147_refined",
    "ctrl05_unbound_prom_yazdi_stable147_refined",
    "ret_all_stable147_refined",
]


def main():
    repo_root = Path(__file__).parent.parent
    storage_dir = repo_root / "SPRM_output" / "gillespie_event"
    config = repo_root / "src" / "gillespie_event" / "gillespie_event_config.yaml"
    storage_dir.mkdir(parents=True, exist_ok=True)

    exit_codes = {}
    for name in DATASETS:
        dataset_dir = SPRM_DATA_DIR / name
        if not dataset_dir.exists():
            print(f"[SKIP] dataset_dir does not exist: {dataset_dir}")
            exit_codes[name] = None
            continue

        cmd = [
            sys.executable, "-m", "src.gillespie_event.cli",
            "--config",      str(config),
            "--dataset_dir", str(dataset_dir),
            "--storage_dir", str(storage_dir),
        ]
        print(f"\n=== Running: {name} ===")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd)
        dt_s = time.perf_counter() - t0
        exit_codes[name] = proc.returncode
        print(f"=== Done: {name}  exit={proc.returncode}  in {dt_s:.1f}s ===")

    print("\nSummary:")
    for name, code in exit_codes.items():
        marker = "SKIP" if code is None else ("OK" if code == 0 else f"FAIL({code})")
        print(f"  {marker:>9}  {name}")


if __name__ == "__main__":
    main()
