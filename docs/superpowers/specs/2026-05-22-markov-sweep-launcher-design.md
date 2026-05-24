# Markov Sweep Launcher: YAML + Generated TSV + SLURM Array

**Date:** 2026-05-22
**Branch:** `markov-new-dataset-cluster`
**Depends on:** `2026-05-22-markov-yaml-loading-design.md` (CLI must accept `--dataset_dir` and read `markov_config.yaml`)

## Problem

The existing parameter-grid launchers (`cluster_sim_scripts/launch_gillespie_param_grid.job`, `launch_rmst_estimation_modular.sh`) have four problems:

1. The parameter grid is hardcoded inside the `.job` file as a `PARAMS=("dataset:conc:coop" ...)` bash array.
2. Colon-separated parsing is fragile and hard to extend.
3. `#SBATCH --array=1-N%N` must be kept in sync with `${#PARAMS[@]}` by hand.
4. Input file selection uses hardcoded `if/elif` branches per dataset.

We need a launcher for the Markov execution sweep over `*_stable147_refined` SPRM datasets crossed with grids of `prot_p_conc` and `prot_cooperativity` that avoids all four problems.

## Goals

- Externalize the sweep specification from the SLURM job script.
- Use structured (tab-delimited) data instead of colon-encoded tuples.
- Auto-compute array size — no manual `--array=1-N` sync.
- Eliminate hardcoded per-dataset branches in the launcher.
- Make the grid inspectable before submitting (`cat sweep_grid.tsv`).
- Support partial re-runs (`sbatch --array=5,9,12 ...`) without changing any code.

## Non-Goals

- No retry/resume mechanism beyond `sbatch --array=<ids>`.
- No automatic post-sweep merging or analysis.
- No global cross-dataset index (each dataset has its own `MarkovStorage` root).
- No changes to `src/markov_execution/` — the CLI already supports `--dataset_dir` and YAML defaults from the previous task.

## Design

### File layout

All new files under `cluster_sim_scripts/markov/`:

```
cluster_sim_scripts/markov/
├── markov_sweep.yaml          # Sweep specification
├── generate_sweep_grid.py     # YAML → sweep_grid.tsv
├── sweep_grid.tsv             # Generated artifact (gitignored)
├── launch_markov_sweep.job    # Thin SLURM array template
└── submit_sweep.sh            # Wrapper that counts rows + sbatches
```

### `markov_sweep.yaml`

Explicit dataset list (no globs) + lists of swept parameter values. Full cross product is computed by the generator.

```yaml
sprm_root: /home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/SPRM_data
storage_root: /home/pol_schiessel/maya620d/pol/Projects/Codebase/Spermatogensis/output/markov_sweep_stable147

datasets:
  - ctrl01_random_genome_safe_stable147_refined
  - ctrl02_random_genome_gcmatched_stable147_refined
  - ctrl03_som_gcmatched_stable147_refined
  - ctrl04_bound_prom_evicted_stable147_refined
  - ctrl05_unbound_prom_yazdi_stable147_refined
  - ret_all_stable147_refined

prot_p_conc:        [0.0, 100.0, 500.0, 1000.0]
prot_cooperativity: [0.0, 4.5]
```

6 datasets × 4 concentrations × 2 cooperativities = **48 array tasks**.

### `generate_sweep_grid.py`

Standalone helper, ~40 lines. Reads the YAML, validates every dataset directory exists under `sprm_root`, then writes `sweep_grid.tsv`:

```
task_id  dataset_name  dataset_dir  storage_dir  prot_p_conc  prot_cooperativity
1        ctrl01_...    /.../SPRM_data/ctrl01_...   /.../markov_sweep_stable147/ctrl01_...   0.0      0.0
2        ctrl01_...    /.../SPRM_data/ctrl01_...   /.../markov_sweep_stable147/ctrl01_...   0.0      4.5
...
48       ret_all_...   /.../SPRM_data/ret_all_...  /.../markov_sweep_stable147/ret_all_...  1000.0   4.5
```

Uses `itertools.product(datasets, prot_p_conc, prot_cooperativity)` for the cross product. Tab-delimited so dataset names with hyphens or underscores parse cleanly.

Behaviour:
- If any `dataset_dir` is missing, abort with a clear error listing what's missing.
- Print the number of tasks generated to stdout.
- Idempotent — re-running overwrites `sweep_grid.tsv`.
- Optional `--config <path>` flag; default is the sibling `markov_sweep.yaml`.

### `launch_markov_sweep.job`

SLURM template. Key behaviours:

- Standard SBATCH header for the cluster (matches `launch_gillespie_param_grid.job` resources: `--ntasks=20`, `--mem-per-cpu=1G`, `--time=48:00:00`, `--exclude=compute-0-[13-15]`).
- **No `#SBATCH --array=...` directive** — provided by `submit_sweep.sh` on the command line.
- Accepts the grid file path as `$1`, defaulting to `cluster_sim_scripts/markov/sweep_grid.tsv`.
- Reads row `$SLURM_ARRAY_TASK_ID + 1` (header skip) with `awk`, splits on tabs with `IFS=$'\t' read`.
- Creates the per-dataset `storage_dir` if missing.
- Invokes `singularity exec ... python3 -m src.markov_execution.cli` with **only** these overrides:
  - `--dataset_dir`
  - `--storage_dir`
  - `--prot_p_conc`
  - `--prot_cooperativity`
  - `--n_workers "$SLURM_NTASKS"`

Everything else (k_wrap, binding_sites, tau_max, tau_steps, method, batch_size, save_*) comes from `src/markov_execution/markov_config.yaml`.

Exit code is propagated so SLURM marks the array task as failed/succeeded correctly.

### `submit_sweep.sh`

Wrapper that auto-sizes the array:

```bash
#!/bin/bash
set -e
GRID="${1:-cluster_sim_scripts/markov/sweep_grid.tsv}"
if [ ! -f "$GRID" ]; then
    echo "ERROR: grid file not found: $GRID"
    echo "Run: python cluster_sim_scripts/markov/generate_sweep_grid.py"
    exit 1
fi
N=$(($(wc -l < "$GRID") - 1))   # subtract header
echo "Submitting array of $N tasks (concurrency cap 20)..."
sbatch --array=1-${N}%20 cluster_sim_scripts/markov/launch_markov_sweep.job "$GRID"
```

Concurrency cap `%20` mirrors the gillespie launcher's setting. Adjustable.

### Workflow

```
1. Edit markov_sweep.yaml (add/remove datasets, change param values)
2. python cluster_sim_scripts/markov/generate_sweep_grid.py
   → writes sweep_grid.tsv (inspect with `cat` or `column -t -s $'\t'`)
3. ./cluster_sim_scripts/markov/submit_sweep.sh
   → sbatch with auto-sized array
```

For partial re-runs:
```
sbatch --array=5,9,12 cluster_sim_scripts/markov/launch_markov_sweep.job
```

### Output layout

```
output/markov_sweep_stable147/
├── ctrl01_random_genome_safe_stable147_refined/      # one MarkovStorage root per dataset
│   ├── k1.0_p0.0_c0.0_tau1000_ode__abc123/
│   │   ├── parameters.json
│   │   ├── summaries/ctrl01_random_genome_safe_stable147_refined.tsv
│   │   └── survivals/ctrl01_random_genome_safe_stable147_refined.parquet
│   └── ... (one subdir per parameter combo)
├── ctrl02_random_genome_gcmatched_stable147_refined/
└── ... (one tree per dataset)
```

Each dataset has its own `MarkovStorage` root → independent `parameters.json` files → no cross-dataset collisions.

### `.gitignore`

Add `cluster_sim_scripts/markov/sweep_grid.tsv` (generated artifact).

## Validation

After generating the grid, sanity checks:

1. `wc -l sweep_grid.tsv` → 49 (48 tasks + header)
2. `cut -f2 sweep_grid.tsv | tail -n+2 | sort -u | wc -l` → 6 (unique datasets)
3. `cut -f5 sweep_grid.tsv | tail -n+2 | sort -un` → `0.0 100.0 500.0 1000.0`
4. `cut -f6 sweep_grid.tsv | tail -n+2 | sort -un` → `0.0 4.5`

Dry-run a single task locally (without SLURM) before cluster submission:

```bash
SLURM_ARRAY_TASK_ID=1 SLURM_NTASKS=2 bash cluster_sim_scripts/markov/launch_markov_sweep.job
```

(Requires temporarily disabling the singularity exec wrapper or running on a host where the .sif exists.)

## Risks & Mitigations

- **Stale TSV after editing YAML:** the workflow requires re-running the generator. The submit_sweep.sh script checks file existence but does not check freshness — easy to forget. Acceptable trade-off; the alternative (auto-regenerate) couples the submit step to a Python invocation that may fail silently.
- **Singularity image path:** mirrors `nucleosome.sif` from working dir, same as existing scripts. If the image lives elsewhere, the `singularity exec` line needs adjustment.
- **Concurrency cap `%20`:** 48 tasks × 20 cores each = up to 400 cores reserved simultaneously. May need lowering depending on cluster policy.
- **Dataset directory missing at submit time:** the generator's pre-check catches this before any sbatch call.
