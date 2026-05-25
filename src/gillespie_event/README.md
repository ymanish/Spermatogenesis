# Gillespie Event-Driven Pipeline

Event-driven Gillespie simulation of nucleosome unwrapping + protamine binding.
No fixed sampling grid: each replicate runs until detachment or `tau > tau_max`
(right-censoring). Produces MFPT/RMST, empirical survival S(tau), and event-only
trajectories per SPRM dataset.

## Quick start

```bash
python -m src.gillespie_event.cli \
    --dataset_dir SPRM_data/ret_all_stable147_refined \
    --storage_dir SPRM_output/gillespie_event \
    --tau_max 50000 --n_survival_points 1000 \
    --replicates 100 --n_workers 20 --batch_size 1
```

## Outputs

For each dataset, three files under
`<storage_dir>/<param_hash_dir>/`:

- `summaries/<file_id>.tsv` — one row per (id, subid). Columns include
  `mfpt_uncensored`, `rmst`, `half_life`, `final_survival`,
  `censored_fraction`, time-weighted ensemble means in two conventions.
- `survival/<file_id>.parquet` — `tau_grid` + `survival` arrays + raw
  `detach_times` per (id, subid).
- `trajectories/<file_id>.parquet` — one row per (id, subid, replicate);
  `traj_tau` and `traj_n_closed` lists, recorded only at `n_closed`-change
  events plus endpoints.

## Differences vs `src/simulation/`

- No fixed `tau_points` grid: replicate ends on detachment or `tau_max`.
- Trajectories record only `n_closed`-change events.
- Outputs include empirical survival S(tau) and right-censored MFPT
  estimators (RMST + uncensored mean + censoring fraction).
- Renucleation and old-format TSV input are not supported.

## Running all six target datasets

```bash
python examples_script/run_gillespie_event_all_datasets.py
```

See `docs/superpowers/specs/2026-05-25-gillespie-event-driven-design.md`
for the full design.
