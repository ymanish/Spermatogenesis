# RMST Replicate Estimator

**Estimate required simulation replicates using Restricted Mean Survival Time (RMST) analysis.**

---

## Why RMST?

Traditional replicate estimation uses **detachment time** (when nucleosome fully unwraps). This fails when nucleosomes are stable:

- ❌ High protamine → many NaN values (no detachment)
- ❌ Loses partial unwrapping information
- ❌ Can't use 30-50% of data

**RMST solves this** by integrating the entire survival curve:

```
RMST = ∫₀^τₘₐₓ S(τ) dτ
```

- ✅ Always defined (100% data usage)
- ✅ Captures full dynamics
- ✅ Works for stable nucleosomes

---

## Quick Start

### Python API

```python
from pathlib import Path
from src.config.custom_type import PilotConfig
from src.analysis.rmst_estimator import estimate_replicates_rmst

config = PilotConfig(
    k_wrap=1.0,
    prot_p_conc=100.0,
    prot_cooperativity=4.5,
    n_pilot_nucleosomes=50,
    n_pilot_replicates=20,
    tau_max=10000.0,
    tau_steps=1000
)

results = estimate_replicates_rmst(
    file_path=Path("hamnucret_data/.../001.tsv"),
    config=config,
    condition_label="prot100_coop4.5",
    save_path=Path("output/rmst"),
    n_workers=20,
    tolerance=0.05,
    seed=42
)

print(f"R = {results.R:.4f}")
print(f"Required replicates (ε=0.05): {results.n_reps_required}")
```

### Command Line

```bash
python -m src.analysis.rmst_estimator.cli \
    --dataset bound \
    --prot-p-conc 100.0 \
    --prot-cooperativity 4.5 \
    --n-nucs 50 \
    --n-reps 20 \
    --n-workers 20 \
    --seed 42
```

### Run Example Script

```bash
cd examples
python example_rmst_replicate_estimation.py
```

---

## HPC Cluster Usage

### SLURM Script

File: `cluster_sim_scripts/launch_rmst_estimation_modular.sh`

```bash
#!/bin/bash
#SBATCH --job-name=rmst_est
#SBATCH --array=1-4
#SBATCH --ntasks=20
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Parameter combinations
PARAMS=(
    "bound:100.0:0.0"
    "unbound:100.0:0.0"
    "bound:500.0:4.5"
    "unbound:500.0:4.5"
)

# Parse parameters
PARAM_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
IFS=':' read -r DATASET PROT_CONC COOP <<< "${PARAMS[$PARAM_INDEX]}"

# Run analysis
python -m src.analysis.rmst_estimator.cli \
    --dataset "$DATASET" \
    --prot-p-conc "$PROT_CONC" \
    --prot-cooperativity "$COOP" \
    --n-nucs 100 \
    --n-reps 50 \
    --n-workers "$SLURM_NTASKS" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --tolerance 0.05 \
    --no-plot
```

### Submit Job

```bash
mkdir -p logs
sbatch cluster_sim_scripts/launch_rmst_estimation_modular.sh
```

---

## Module Structure

```
src/analysis/rmst_estimator/
├── __init__.py          # Main API: estimate_replicates_rmst()
├── core.py              # RMST computation, survival curves
├── sampling.py          # Memory-efficient nucleosome sampling
├── simulation.py        # Parallel simulation execution
├── variance.py          # Statistical variance analysis
├── visualization.py     # Plotting functions
├── io.py                # File I/O and reports
└── cli.py               # Command-line interface
```

**Import any component:**
```python
from src.analysis.rmst_estimator import (
    estimate_replicates_rmst,      # Main function
    sample_nucleosomes,             # Sampling
    run_rmst_pilot_study,           # Simulation
    compute_variance_components,    # Analysis
    plot_rmst_analysis              # Plotting
)
```

---

## CLI Options

```bash
# Required
--dataset {bound,unbound}    # Dataset type
--n-nucs N                   # Number of nucleosomes
--n-reps R                   # Replicates per nucleosome
--seed SEED                  # Random seed

# Protamine parameters
--k-wrap K                   # Wrapping energy (default: 1.0)
--prot-p-conc C              # Concentration μM (default: 0.0)
--prot-cooperativity X       # Cooperativity k_BT (default: 0.0)

# Analysis
--tolerance ε                # Tolerance for N_rep (default: 0.05)
--n-workers N                # Parallel workers (default: 1)

# Optional
--tau-max T                  # Max time (default: 10000.0)
--tau-steps S                # Time steps (default: 1000)
--batch-size B               # Batch size (default: 10)
--no-plot                    # Disable plots (for cluster)
--output-dir DIR             # Output directory
```

---

## Output Files

```
output/rmst_<condition>_<timestamp>/
├── rmst_analysis_<condition>.txt       # Human-readable report
├── rmst_analysis_<condition>.json      # Machine-readable data
├── rmst_plot_<condition>.png           # 4-panel analysis plot
├── config.json                         # Run configuration
└── metadata.json                       # Run metadata
```

### Example Report

```
RMST STATISTICS
────────────────────────────────────────
Mean RMST:                  5432.18 ± 234.56
Median RMST:                5401.23

VARIANCE DECOMPOSITION
────────────────────────────────────────
Within-nucleosome (σ²_w):   45123.45
Between-nucleosome (σ²_b):  12345.67
Variance ratio (R):         0.7852

REPLICATE RECOMMENDATION
────────────────────────────────────────
Tolerance (ε):              0.05
Required replicates:        146
```

---

## Theory (Brief)

### RMST Definition

Area under survival curve from 0 to τ_max:

```
RMST_i,r = ∫₀^τₘₐₓ S_i,r(τ) dτ

where S(τ) = CS(τ) / CS_initial
```

### Variance Decomposition

```
σ²_total = σ²_within + σ²_between

σ²_within  = average replicate variance (stochastic noise)
σ²_between = variance across nucleosomes (biological)
```

### Variance Ratio

```
R = σ²_within / (σ²_within + σ²_between)

R → 0: Low noise, few replicates needed
R → 1: High noise, many replicates needed
```

### Required Replicates

```
N_rep = R / [ε² × (1 - R)]

Example: R = 0.8, ε = 0.05
→ N_rep = 0.8 / (0.0025 × 0.2) = 1600
```

---

## Troubleshooting

**Import Error**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Memory Error**
```python
# Reduce batch size or nucleosome count
estimate_replicates_rmst(..., batch_size=5)
config = PilotConfig(n_pilot_nucleosomes=20)
```

**Slow Performance**
```python
# Increase workers, reduce tau_steps
estimate_replicates_rmst(..., n_workers=40)
config = PilotConfig(tau_steps=500)
```

**Cluster Plotting Error**
```bash
# Add --no-plot flag
python -m src.analysis.rmst_estimator.cli --no-plot ...
```

---

## When to Use RMST vs Detachment Time

| Scenario | Detachment Time | RMST |
|----------|----------------|------|
| No protamine | ❌ Many NaN | ✅ Recommended |
| Low protamine (< 200 μM) | ⚠️ Some NaN | ✅ Recommended |
| High protamine (> 500 μM) | ❌ Mostly NaN | ✅ Required |
| High cooperativity | ❌ Stabilized | ✅ Required |
| Quick analysis | ✅ Simple | ✅ More robust |

**Default recommendation:** Use RMST unless you specifically need endpoint analysis.

---

## Examples

See `examples/example_rmst_replicate_estimation.py` for:

1. **No protamine** - Stable nucleosomes (RMST essential)
2. **Medium protamine** - Mixed behavior (RMST better)
3. **RMST vs detachment** - Direct comparison
4. **High protamine** - Very stable (RMST required)

---

## References

**Variance Components:**  
Searle, S.R. et al. (2009). *Variance Components*. Wiley.

**RMST Theory:**  
Royston, P. & Parmar, M.K. (2013). "Restricted mean survival time." *BMC Med Res Methodol*, 13(1), 152.

**Project Documentation:**
- Main README: `README.md`
- Example script: `examples/example_rmst_replicate_estimation.py`
- Cluster script: `cluster_sim_scripts/launch_rmst_estimation_modular.sh`

---

**Author:** MY  
**Date:** November 2025  
