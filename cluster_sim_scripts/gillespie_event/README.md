# Gillespie-event validation sweep

This directory runs the **event-driven Gillespie** simulation as a validation
counterpart to the Markov solver sweep. It runs the same concentration /
cooperativity / log-τ grid as Markov, and sweeps the protamine on-rate `k_bind`
(the ladder `1, 10, 100`) to show the Gillespie survival curve **converging onto
the Markov fast-protamine limit** as `k_bind` increases.

The trick that makes this affordable: instead of running every nucleosome, we
run only the **reachable** ones — nucleosomes whose Markov MFPT is small enough
that they actually evict inside the observation window. A Gillespie run costs
roughly `(event rate) × (τ at eviction)`, so a stiff nucleosome that never
evicts runs all the way to `tau_max` burning 100–1000× more events and produces
a flat, useless survival curve. Sampling by Markov MFPT keeps only the cheap,
informative ones. See `notes/fast_protamine_review.md` for the physics.

---

## The pipeline (4 steps)

```
[1] Markov sweep            -> MFPT per nucleosome         (you run; separate job)
        │
        ▼
[2] select_reachable_ids.py -> id-lists + manifest.tsv     (reads Markov output)
        │
        ▼
[3] generate_sweep_grid.py  -> sweep_grid.tsv              (one row per task)
        │
        ▼
[4] submit_sweep.sh         -> Gillespie survival curves   (SLURM array job)
```

You run step 1 (the Markov sweep) and step 4 (the Gillespie sweep). Steps 2–3
are quick prep in between. **Step 2 must run after the Markov sweep finishes** —
it reads the Markov MFPT output.

| step | script | reads | produces |
|------|--------|-------|----------|
| 1 | *(Markov sweep, `cluster_sim_scripts/markov/`)* | SPRM datasets | `<markov_root>/<dataset>/<cell>/summaries/<dataset>.tsv` — one MFPT per nucleosome |
| 2 | `select_reachable_ids.py` | the Markov summaries | `<ids_root>/<dataset>/p{conc}_c{coop}.ids.txt` (≤ `sampling_n` global_ids each) + `<ids_root>/manifest.tsv` |
| 3 | `generate_sweep_grid.py` | `gillespie_event_sweep.yaml` + the id-lists' paths | `sweep_grid.tsv` — one row per (dataset, conc, coop, k_bind), each pointing at its id-list |
| 4 | `submit_sweep.sh` → `launch_gillespie_event_sweep.job` → `src.gillespie_event.cli` | `sweep_grid.tsv` | Gillespie output under `<storage_root>/<dataset>/`: survival `S(τ)`, MFPT/RMST, trajectories — **only for the sampled nucleosomes** |

### Commands (all on the cluster, after the Markov sweep finishes)

```bash
git pull

# [2] sample reachable nucleosomes from the Markov MFPT output
python cluster_sim_scripts/gillespie_event/select_reachable_ids.py
#     inspect what each cell got before spending compute (path is `ids_root` from the yaml):
column -t <ids_root>/manifest.tsv

# [3] build the task grid (references the id-lists)
python cluster_sim_scripts/gillespie_event/generate_sweep_grid.py

# [4] submit the SLURM array
./cluster_sim_scripts/gillespie_event/submit_sweep.sh
```

---

## Configuration — `gillespie_event_sweep.yaml`

Single source of truth for all four steps. The knobs you are most likely to
touch:

| key | meaning |
|-----|---------|
| `sweep.datasets` | which SPRM datasets to run |
| `sweep.prot_p_conc`, `sweep.prot_cooperativity` | the grid (match the Markov sweep) |
| `sweep.prot_k_bind_phys` | the convergence ladder `[1, 10, 100]` (physical µM⁻¹s⁻¹) |
| `markov_root` | where step 1's MFPT output lives (step 2 reads this) |
| `ids_root` | where step 2 writes id-lists and step 4 reads them |
| `sampling_n` | target nucleosomes per cell (default 100) |
| `sampling_cap` | reachability cap: keep nucleosomes with Markov MFPT `< cap` in τ units (default 5000) |

### What `sampling_cap` means

τ is dimensionless time (`τ = k_wrap · t`, `k_wrap ≈ 21/s`). `sampling_cap = 5000`
keeps nucleosomes whose **mean eviction time (MFPT) is below 5000 τ** (≈ 240 s
real time). Lower cap → cheaper, earlier-evicting sims but fewer qualify at low
concentration; higher cap → more samples but longer runs. 5000 yields ≥ 100 in
every mid/high-concentration cell.

### Sample counts are not uniform

Each cell takes `min(sampling_n, n_reachable)`. High-concentration cells hit the
full 100; low/zero-concentration cells (intrinsic unwrapping only — little or no
protamine) may yield far fewer (e.g. ~10). This is expected and recorded per
cell in `manifest.tsv` (`n_reachable`, `n_selected`). The convergence figure
lives in the mid/high-concentration cells, so thin low-conc cells don't hurt it.

Because the Markov MFPT depends only on `c0 = k_unbind/k_bind` (held fixed across
the ladder), **one id-list per (dataset, conc, coop) is reused for every `k_bind`
rung** — so the ladder compares the *same* nucleosomes at `k_bind = 1, 10, 100`.

---

## Sampling on your laptop instead of the cluster

The id-lists contain only integer `global_id`s, so their **contents are
machine-independent** — sampling locally gives the identical nucleosome set. Two
things to get right:

1. Point the sampler at your local paths (the yaml defaults are cluster paths):
   ```bash
   python cluster_sim_scripts/gillespie_event/select_reachable_ids.py \
       --markov_root /path/to/local/markov_output \
       --out_dir     /path/to/local/ids
   ```
2. The cluster job reads id-lists from `ids_root` (baked into `sweep_grid.tsv` by
   step 3). So after sampling locally, **copy the id-lists up to the cluster
   `ids_root`** before submitting:
   ```bash
   rsync -av /path/to/local/ids/ <cluster>:<ids_root>/
   ```

Simplest is to run step 2 on the cluster where the Markov output already sits —
then there is nothing to copy.

---

## Artifacts: what is committed vs regenerated

- **Committed:** `gillespie_event_sweep.yaml`, `sweep_grid.tsv` (regenerate with
  step 3 whenever you edit the yaml, then commit + push + `git pull` on cluster).
- **Not committed:** the id-lists under `ids_root` and `manifest.tsv` —
  regenerate on the cluster with step 2 after each Markov run.
