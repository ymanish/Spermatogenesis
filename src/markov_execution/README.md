# Markov Execution Module

Modular package for running Markov solver calculations on multiple nucleosomes with parallelization and output management.

## Overview

This module provides a high-level interface for executing Markov chain calculations across multiple nucleosomes, mirroring the design of the `simulation` module for consistency.

### Key Features

✅ **Parallel processing**: Batch execution with configurable workers  
✅ **Memory efficient**: Streaming output with temporary file merging  
✅ **Flexible configuration**: Dataclass-based config similar to SimulationConfig  
✅ **Multiple outputs**: Summary TSV + detailed Parquet with survival/state data  
✅ **CLI support**: Command-line interface for easy execution  
✅ **Consistent style**: Matches simulation module architecture  

## Quick Start

### Python API

```python
from src.markov_execution import run_markov_solver, MarkovConfig
from pathlib import Path

# Create configuration
config = MarkovConfig(
    k_wrap=1.0,
    prot_p_conc=10.0,
    prot_cooperativity=0.0,
    t_max=1000.0,
    t_steps=500,
    n_workers=10,
    save_survival=True,
    save_mfpt=True
)

# Run solver
run_markov_solver(
    file_path=Path("data/nucleosomes.tsv"),
    output_dir=Path("output/markov"),
    config=config,
    max_nucs=50
)
```

### Command Line

```bash
python -m src.markov_execution.cli \
    data/nucleosomes.tsv \
    output/markov \
    --k-wrap 1.0 \
    --prot-p-conc 10.0 \
    --t-max 1000.0 \
    --t-steps 500 \
    --n-workers 10 \
    --max-nucs 50
```

## Module Structure

```
markov_execution/
├── __init__.py          # Public API exports
├── config.py           # MarkovConfig dataclass
├── solver_runner.py    # Individual nucleosome solver
├── batch.py            # Batch processing with parallelization
├── output.py           # File I/O and merging
├── orchestrator.py     # Main orchestration function
├── cli.py              # Command-line interface
└── README.md           # This file
```

### Component Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration dataclass with validation |
| `solver_runner.py` | Solve single nucleosome Markov chain |
| `batch.py` | Process batches in parallel workers |
| `output.py` | Save and merge results to files |
| `orchestrator.py` | Main entry point, coordinates execution |
| `cli.py` | Command-line interface |

## Configuration

### MarkovConfig

```python
@dataclass
class MarkovConfig:
    # Nucleosome parameters
    k_wrap: float = 1.0
    binding_sites: int = 14
    kT: float = 1.0
    
    # Protamine parameters
    prot_k_bind: float = 1.0
    prot_k_unbind: float = 89.7
    prot_p_conc: float = 0.0
    prot_cooperativity: float = 0.0
    
    # Computation parameters
    t_max: float = 1000.0
    t_steps: int = 500
    method: str = 'expm'  # 'expm' or 'ode'
    sparse: bool = False
    
    # Execution parameters
    batch_size: int = 10
    n_workers: int = 10
    max_nucs: Optional[int] = None
    
    # Output parameters
    save_survival: bool = True
    save_states: bool = False
    save_mfpt: bool = True
```

## Output Files

### 1. Summary TSV (`markov_summary.tsv`)

Tab-separated file with one row per nucleosome:

```
id  subid  n_states  binding_sites  mfpt  half_life  final_survival  mean_survival
0   0      105       14             125.3  98.2      0.001          0.452
0   1      105       14             132.1  102.5     0.002          0.468
...
```

**Columns:**
- `id`, `subid`: Nucleosome identifiers
- `n_states`: Number of transient states in Markov chain
- `binding_sites`: Number of DNA-histone contacts
- `mfpt`: Mean first passage time (dimensionless τ)
- `half_life`: Time when S(t) = 0.5
- `final_survival`: S(t_max)
- `mean_survival`: Average survival over time

### 2. Detailed Parquet (`markov_results.parquet`)

Columnar format with arrays per nucleosome:

```python
import polars as pl

df = pl.read_parquet("markov_results.parquet")
# Columns:
# - id, subid
# - t_grid: Time grid (array)
# - survival: S(t) (array)
# - state_probs: P(t) (2D array, if save_states=True)
# - mfpt: Scalar MFPT value
# - mfpt_vec: MFPT from all states (array)
```

### 3. Configuration (`markov_config.txt`)

Plain text file with all configuration parameters for reproducibility.

## Examples

### Example 1: Basic Usage

```python
from src.markov_execution import run_markov_solver, MarkovConfig
from pathlib import Path

config = MarkovConfig(
    k_wrap=1.0,
    prot_p_conc=0.0,  # No protamine
    t_max=5000.0,
    t_steps=1000,
    n_workers=20
)

run_markov_solver(
    file_path=Path("data/nucleosomes.tsv"),
    output_dir=Path("output/no_protamine"),
    config=config
)
```

### Example 2: With Protamine

```python
config = MarkovConfig(
    k_wrap=1.0,
    prot_k_bind=1.0,
    prot_k_unbind=89.7,
    prot_p_conc=10.0,
    prot_cooperativity=0.0,
    t_max=1000.0,
    t_steps=500,
    method='expm',
    n_workers=10,
    save_survival=True,
    save_states=False
)

run_markov_solver(
    file_path=Path("data/nucleosomes.tsv"),
    output_dir=Path("output/with_protamine"),
    config=config,
    max_nucs=100
)
```

### Example 3: Large System (ODE solver)

```python
config = MarkovConfig(
    k_wrap=1.0,
    prot_p_conc=50.0,
    t_max=2000.0,
    t_steps=800,
    method='ode',  # Faster for large systems
    sparse=True,   # Memory efficient
    n_workers=20,
    batch_size=5
)

run_markov_solver(
    file_path=Path("data/nucleosomes.tsv"),
    output_dir=Path("output/large_scale"),
    config=config
)
```

### Example 4: Parameter Scan

```python
from pathlib import Path

concentrations = [0.0, 1.0, 10.0, 100.0]

for conc in concentrations:
    config = MarkovConfig(
        k_wrap=1.0,
        prot_p_conc=conc,
        t_max=1000.0,
        t_steps=500,
        n_workers=10
    )
    
    run_markov_solver(
        file_path=Path("data/nucleosomes.tsv"),
        output_dir=Path(f"output/scan_c{conc:.0f}"),
        config=config,
        max_nucs=50
    )
```

### Example 5: Save Full State Probabilities

```python
config = MarkovConfig(
    k_wrap=1.0,
    prot_p_conc=10.0,
    t_max=500.0,
    t_steps=200,
    save_survival=True,
    save_states=True,  # Save P(t) for all states
    save_mfpt=True,
    n_workers=5
)

run_markov_solver(
    file_path=Path("data/nucleosomes.tsv"),
    output_dir=Path("output/with_states"),
    config=config,
    max_nucs=10  # Limit for memory
)
```

## Analysis Examples

### Load and Analyze Results

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Load summary
summary = pl.read_csv("output/markov/markov_summary.tsv", separator='\t', comment_prefix='#')

# Statistics
print(f"Mean MFPT: {summary['mfpt'].mean():.2f} ± {summary['mfpt'].std():.2f}")
print(f"Mean half-life: {summary['half_life'].mean():.2f}")

# Load detailed results
results = pl.read_parquet("output/markov/markov_results.parquet")

# Plot survival function for first nucleosome
row = results.row(0, named=True)
t_grid = np.array(row['t_grid'])
S = np.array(row['survival'])

plt.figure(figsize=(8, 5))
plt.plot(t_grid, S, 'b-', lw=2)
plt.xlabel('Dimensionless Time (τ)')
plt.ylabel('Survival S(t)')
plt.title(f"Nucleosome {row['id']}-{row['subid']}")
plt.grid(True, alpha=0.3)
plt.show()
```

### Compare Multiple Nucleosomes

```python
# Plot all survival curves
fig, ax = plt.subplots(figsize=(10, 6))

for row in results.iter_rows(named=True):
    t_grid = np.array(row['t_grid'])
    S = np.array(row['survival'])
    ax.plot(t_grid, S, alpha=0.3, lw=1)

# Average
t_grid = np.array(results['t_grid'][0])
all_S = np.array([np.array(s) for s in results['survival']])
S_mean = all_S.mean(axis=0)
S_std = all_S.std(axis=0)

ax.plot(t_grid, S_mean, 'k-', lw=3, label='Mean')
ax.fill_between(t_grid, S_mean - S_std, S_mean + S_std, 
                alpha=0.3, color='gray', label='±1 SD')

ax.set_xlabel('Time (τ)')
ax.set_ylabel('Survival S(t)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Performance Tips

### For Large Datasets

1. **Use ODE solver** for systems with many states:
   ```python
   config.method = 'ode'
   config.sparse = True
   ```

2. **Adjust batch size** based on memory:
   ```python
   config.batch_size = 5  # Smaller batches for large nucleosomes
   ```

3. **Don't save state probabilities** unless needed:
   ```python
   config.save_states = False
   ```

### For Parameter Scans

1. **Reuse nucleosome loading**:
   ```python
   # Load once
   nucs = load_nucleosomes_from_file(file_path, k_wrap=1.0)
   
   # Process with different configs
   for config in configs:
       run_markov_solver(...)
   ```

2. **Parallel parameter scanning**:
   ```python
   from concurrent.futures import ProcessPoolExecutor
   
   def run_with_config(conc):
       config = MarkovConfig(prot_p_conc=conc, ...)
       run_markov_solver(...)
   
   with ProcessPoolExecutor(max_workers=4) as executor:
       executor.map(run_with_config, concentrations)
   ```

## Comparison with Simulation Module

| Feature | Simulation | Markov Execution |
|---------|-----------|------------------|
| Input | Nucleosome data | Nucleosome data |
| Computation | Gillespie (stochastic) | Markov (analytical) |
| Config class | `SimulationConfig` | `MarkovConfig` |
| Main function | `run_simulation()` | `run_markov_solver()` |
| Output | Trajectories + summary | Survival + summary |
| Parallelization | ✅ Batch + workers | ✅ Batch + workers |
| Time convention | Dimensionless τ | Dimensionless τ |
| CLI | ✅ Yes | ✅ Yes |

Both modules share:
- Similar architecture (orchestrator, batch, output, cli)
- Dataclass-based configuration
- Parallel batch processing
- Polars-based file I/O
- Consistent naming conventions

## Troubleshooting

### Memory Issues

**Problem:** Out of memory with large nucleosome sets

**Solutions:**
```python
# 1. Use sparse matrices
config.sparse = True

# 2. Smaller batches
config.batch_size = 5

# 3. Don't save state probabilities
config.save_states = False

# 4. Process fewer nucleosomes at once
run_markov_solver(..., max_nucs=100)
```

### Slow Execution

**Problem:** Matrix exponential taking too long

**Solutions:**
```python
# 1. Use ODE solver
config.method = 'ode'

# 2. Reduce time points
config.t_steps = 200  # Instead of 1000

# 3. Increase workers
config.n_workers = 20
```

### Failed Batches

**Problem:** Some batches fail due to errors

**Check:**
- Look for error rows in TSV (n_states = -1, mfpt = NaN)
- Check worker logs for detailed errors
- Validate nucleosome G_mat data

## See Also

- `src/analysis/markov_solver/README.md` - Core solver module documentation
- `src/simulation/README.md` - Simulation module (Gillespie)
- `compare_gillespie_vs_markov.py` - Comparison script
- `docs/dimensionless_time_convention.md` - Time convention details

## Authors

- MY (2025)

## License

MIT License - See LICENSE file for details.
