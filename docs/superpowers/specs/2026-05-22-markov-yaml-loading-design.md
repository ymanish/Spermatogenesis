# Markov Execution: YAML Config Loading & SPRM Dataset Support

**Date:** 2026-05-22
**Branch:** `markov-new-dataset-cluster`
**Status:** Approved design, ready for implementation plan

## Problem

The Markov execution CLI (`src/markov_execution/cli.py`) currently uses pure argparse and only accepts an old-format TSV file via `--infile`. The `markov_config.yaml` file sitting next to it is **not actually read by any code** — it is dead documentation. As a result:

1. There is no way to pass an SPRM-format dataset (`energies.tsv` + `id_lookup.tsv`) into the Markov pipeline, even though the underlying loader `nucleosome_generator_sprm` already exists in `src/core/build_nucleosomes.py`.
2. Local runs require typing every parameter as a CLI flag instead of editing a config file.
3. The sister module `src/simulation/cli.py` already has a clean YAML + CLI override pattern that should be mirrored for consistency.

The downstream goal (separate task) is a parameter sweep over the `*_stable147_refined` SPRM datasets crossed with a grid of `prot_p_conc` and `prot_cooperativity` values. That sweep needs SPRM-input support in the CLI as a prerequisite.

## Goals

- Bring the Markov CLI in line with the simulation CLI: YAML defaults + CLI overrides, supporting both `--dataset_dir` (SPRM) and `--infile` (old TSV) as mutually exclusive inputs.
- Make `markov_config.yaml` a real, working config file.
- Delete the unused `analyze_delta_scan.py` script (references a non-existent `markov_sweep.yaml`).

## Non-Goals

- No changes to `MarkovConfig` schema or the solver/batch/output internals.
- No wiring of half-implemented YAML fields (`eads_delta`, `eads_weight_mode`, `eads_apply`, `check_matrix_density`, `max_nucs_seed`) — these are stripped from the YAML.
- No launcher / sweep script in this task — that is a follow-up.
- No touching of unrelated modules.

## Design

### File-level changes

#### `src/markov_execution/cli.py` — restructure to YAML + CLI pattern

Mirror the structure of `src/simulation/cli.py:108-165` exactly. Three helpers:

```python
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}

def resolve_params(args, cfg: dict) -> dict:
    # CLI value wins if not None, else YAML value, else hardcoded fallback
    ...
```

Changes to `parse_args`:

- Add `--config` flag, default = `Path(__file__).parent / "markov_config.yaml"`.
- Add `--dataset_dir` flag (mutually exclusive with `--infile`; validation happens after merging).
- Change every existing flag's `default=...` to `default=None` so "not passed" is detectable.
- Boolean flags (`--save_survival`, `--save_states`, `--save_mfpt`, `--sparse`) keep `action="store_true"` but the merge uses a `pick_bool` helper that treats `False` as "not set".

Changes to `main`:

- Load YAML via `_load_yaml(args.config)`.
- Merge with `resolve_params(args, cfg)`.
- Validate input: exactly one of `dataset_dir` / `infile` must be set; `storage_dir` is required; paths must exist.
- Compute `file_id`:
  - `dataset_dir.name` when SPRM input is used (e.g. `ret_all_stable147_refined`).
  - `infile.stem` when old-format input is used.
  - If `--dataset` is set, it prefixes: `f"{dataset}_{file_id}"`.
- Pass both `file_path` and `dataset_dir` through to `run_markov_solver` (the orchestrator handles the dispatch).
- Unknown YAML keys are silently ignored — only fields explicitly listed in `resolve_params` are merged.

#### `src/markov_execution/orchestrator.py` — accept SPRM input

Signature change for `run_markov_solver`:

```python
def run_markov_solver(
    tsv_outfile: Path,
    survival_outfile: Path,
    config: MarkovConfig,
    file_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    max_nucs: Optional[int] = None,
    subids_range: Optional[tuple] = None
) -> None:
```

Validation block at the top: `if (file_path is None) == (dataset_dir is None): raise ValueError(...)`.

Dispatch in the body — mirror `src/simulation/orchestrator.py:61-78`:

```python
from src.core.build_nucleosomes import nucleosome_generator, nucleosome_generator_sprm

if dataset_dir is not None:
    gen = nucleosome_generator_sprm(
        dataset_dir=str(dataset_dir),
        k_wrap=config.k_wrap,
        binding_sites=config.binding_sites,
    )
else:
    if subids_range is not None:
        gen = nucleosome_generator(
            file_path=file_path,
            k_wrap=config.k_wrap,
            binding_sites=config.binding_sites,
            subids=np.arange(*subids_range).tolist(),
        )
    else:
        gen = nucleosome_generator(
            file_path=file_path,
            k_wrap=config.k_wrap,
            binding_sites=config.binding_sites,
        )
```

Note: `nucleosome_generator_sprm` does not currently expose a `subids` filter, so the `subids_range` argument is ignored when `dataset_dir` is set. This matches the simulation orchestrator's behavior.

Logging block updated to print `Input: <file_path or dataset_dir>` instead of just `Input file: ...`.

#### `src/markov_execution/markov_config.yaml` — clean up

Keep only fields that map to real `MarkovConfig` / CLI options. Remove orphans:

- `eads_delta`, `eads_weight_mode`, `eads_apply` (no MarkovConfig fields; not wired through)
- `check_matrix_density` (no plumbing exists)
- `max_nucs_seed` (no MarkovConfig field)

Keep the leading comment block explaining `dataset_dir` vs `infile` mutual exclusion. Final field list:

```yaml
# Input / Output
dataset_dir: <path or null>
infile: <path or null>
storage_dir: <path>
dataset: null

# Nucleosome parameters
k_wrap: 1.0
binding_sites: 14

# Protamine parameters
prot_k_unbind: 89.7
prot_k_bind: 1.0
prot_p_conc: 100.0
prot_cooperativity: 4.5

# Computation
tau_max: 1000.0
tau_steps: 500
method: ode
sparse: false

# Execution
batch_size: 10
n_workers: 20

# Output
save_survival: true
save_states: false
save_mfpt: true

# Testing / debugging
max_nucs: null
subids_start: null
subids_end: null
```

#### Delete `src/markov_execution/analyze_delta_scan.py`

Standalone analysis script referencing a non-existent `markov_sweep.yaml`. No other code imports it.

### Data flow

```
markov_config.yaml ──┐
                     ├──> resolve_params() ──> param dict ──> MarkovConfig + paths ──> run_markov_solver()
CLI flags ───────────┘                                                                      │
                                                                                            ├── nucleosome_generator_sprm(dataset_dir)
                                                                                            └── nucleosome_generator(file_path)
                                                                                                          │
                                                                                            ProcessPoolExecutor → batches → merged TSV + Parquet
```

### Backwards compatibility

- Existing scripts that pass `--infile <path>` continue to work unchanged. Every previously-required CLI flag remains accepted.
- The `storage` setup in `cli.py` (computing `tsv_outfile` and `survival_outfile` via `MarkovStorage`) is preserved; only the inputs to that block change.
- `run_markov_solver` callers that pass `file_path=...` as a keyword argument continue to work. Positional callers (none exist in-tree) would need updating — verified by grep.

## Validation

Smoke test once code compiles:

```bash
python -m src.markov_execution.cli \
    --dataset_dir SPRM_data/ctrl01_random_genome_safe_stable147_refined \
    --storage_dir output/markov_smoke \
    --max_nucs 5 \
    --n_workers 2 \
    --method ode \
    --tau_max 100 --tau_steps 50
```

Expected:
- A directory like `output/markov_smoke/k1.0_p100.0_c4.5_tau100_ode__XXXXXX/` exists.
- It contains `summaries/ctrl01_random_genome_safe_stable147_refined.tsv` and `survivals/ctrl01_random_genome_safe_stable147_refined.parquet`.
- The summary TSV has 5 rows (one per nucleosome).

A second test runs the same command with no CLI overrides except `--dataset_dir` and `--storage_dir`, confirming the YAML defaults are picked up.

## Risks & Mitigations

- **Risk:** `nucleosome_generator_sprm` could yield nucleosomes with a different attribute shape than `nucleosome_generator`, breaking downstream `run_batch_markov`. **Mitigation:** the simulation pipeline already uses both generators interchangeably — same risk would have surfaced there.
- **Risk:** The YAML `tau_max`/`tau_steps` keys in the current file match the CLI flag names, but `MarkovConfig` field names are also `tau_max`/`tau_steps` — confirmed by `src/markov_execution/config.py:90-91`. No translation needed.
- **Risk:** Removing `analyze_delta_scan.py` could break a user workflow. **Mitigation:** verified zero imports across the repo; the YAML it references doesn't exist; safe.
