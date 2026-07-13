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
    --tau_max 50000 --tau_steps 1000 --tau_spacing log \
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

## Restricting to specific nucleosomes (`--global_ids_file`)

Pass a text file of nucleosome `global_id`s (one per line, `#` comments allowed)
to run only those nucleosomes:

```bash
python -m src.gillespie_event.cli ... --global_ids_file reachable.ids.txt
```

The Markov-validation sweep uses this to run only *reachable* nucleosomes —
those whose Markov MFPT falls below a cap, so they evict inside the window
instead of running to `tau_max` (per-sim cost scales with eviction time, not the
ceiling). Generate the id-lists from the Markov MFPT sweep with:

```bash
python cluster_sim_scripts/gillespie_event/select_reachable_ids.py \
    --config cluster_sim_scripts/gillespie_event/gillespie_event_sweep.yaml
```

which writes one id-list per (dataset, conc, coop) cell under `ids_root` plus a
`manifest.tsv` of reachable/selected counts. Run it after the Markov sweep and
before `generate_sweep_grid.py`.

## Running all six target datasets

```bash
python examples_script/run_gillespie_event_all_datasets.py
```

See `docs/superpowers/specs/2026-05-25-gillespie-event-driven-design.md`
for the full design.
