# Simulation Package

**Modular package for running Gillespie simulations of nucleosome dynamics with parallelization.**

---

## 📦 What's New

✨ **SimulationConfig Class** - All simulation parameters encapsulated in a single object  
✨ **Modular Design** - Refactored from 600+ line monolith into focused modules  
✨ **Better API** - Clean interface with 4 parameters instead of 15+  
✨ **Type Safety** - Dataclass validation and computed properties  
✨ **Backward Compatible** - Old API still works with deprecation warnings  

---

## 🚀 Quick Start

### Using SimulationConfig (Recommended)

```python
from pathlib import Path
from src.simulation import run_simulation
from src.config.custom_type import SimulationConfig

# Create configuration
config = SimulationConfig(
    # Nucleosome parameters
    k_wrap=1.0,
    binding_sites=14,
    
    # Protamine parameters
    prot_p_conc=100.0,
    prot_cooperativity=4.5,
    prot_k_unbind=89.7,
    prot_k_bind=1.0,
    
    # Time parameters
    tau_max=10000.0,
    tau_steps=1000,
    
    # Simulation parameters
    inf_protamine=True,
    renucleation=False,
    replicates=20,
    
    # Execution parameters
    batch_size=10,
    n_workers=4,
    
    # Trajectory parameters
    save_trajectories=True,
    maxpoints_saved_trajectories=100
)

# Run simulation
run_simulation(
    file_path=Path("data/nucleosomes.tsv"),
    traj_outfile=Path("output/trajectories.parquet"),
    tsv_outfile=Path("output/summary.tsv"),
    config=config
)
```

### Command-Line Interface

```bash
python -m src.simulation.cli \
    --infile data/nucleosomes.tsv \
    --storage_dir output/simulations \
    --k_wrap 1.0 \
    --prot_p_conc 100.0 \
    --prot_cooperativity 4.5 \
    --replicates 20 \
    --n_workers 10 \
    --save_trajectories
```

---

## 📚 SimulationConfig

### Overview

`SimulationConfig` encapsulates all simulation parameters in a validated dataclass with computed properties.

**Benefits:**
- **Single source of truth** - All parameters in one place
- **Validation** - Checks parameter combinations on creation
- **Computed properties** - Auto-calculates derived values
- **Serializable** - Easy to save/load configurations
- **Type safe** - Clear parameter types and defaults

### Parameters

#### Nucleosome Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_wrap` | float | 1.0 | Nucleosome wrapping rate |
| `binding_sites` | int | 14 | Number of binding sites per nucleosome |

#### Protamine Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prot_k_unbind` | float | 89.7 | Protamine unbinding rate |
| `prot_k_bind` | float | 1.0 | Protamine binding rate |
| `prot_p_conc` | float | 0.0 | Protamine concentration |
| `prot_cooperativity` | float | 0.0 | Cooperativity parameter |

#### Time Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau_max` | float | 10000.0 | Maximum simulation time |
| `tau_steps` | int | 1000 | Number of time points |

#### Simulation Behavior
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inf_protamine` | bool | True | Infinite protamine pool |
| `renucleation` | bool | False | Enable renucleation |
| `replicates` | int | 20 | Number of replicates |

#### Execution Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 10 | Nucleosomes per batch |
| `n_workers` | int | 4 | Parallel workers |
| `flush_every` | int | 10000 | Save frequency |

#### Trajectory Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_trajectories` | bool | False | Save full trajectories |
| `maxpoints_saved_trajectories` | int | 100 | Max trajectory points |

### Computed Properties

These are automatically calculated from the input parameters:

- **`tau_points`**: `np.ndarray` - Array of time points from 0 to `tau_max`
- **`tau_min`**: `Optional[float]` - Minimum time for renucleation (if enabled)
- **`build_params`**: `dict` - Factory functions for creating objects

### Methods

#### `to_dict() -> dict`
Convert configuration to dictionary for serialization:

```python
config = SimulationConfig(k_wrap=1.0, prot_p_conc=100.0)
config_dict = config.to_dict()
# Save to JSON/YAML/etc
```

#### `from_dict(config_dict: dict) -> SimulationConfig`
Create configuration from dictionary:

```python
config_dict = {
    'k_wrap': 1.0,
    'prot_p_conc': 100.0,
    'tau_max': 10000.0,
    'replicates': 20
}
config = SimulationConfig.from_dict(config_dict)
```

#### `__repr__() -> str`
Pretty-print configuration:

```python
print(config)
# Output:
# SimulationConfig(
#   k_wrap=1.0, binding_sites=14,
#   prot_k_unbind=89.7, prot_k_bind=1.0,
#   prot_p_conc=100.0, prot_cooperativity=4.5,
#   ...
# )
```

---

## 🧩 Module Structure

```
src/simulation/
├── __init__.py          # Package exports
├── simulator.py         # Simulator creation
├── replicate.py         # Single replicate execution
├── batch.py             # Batch processing
├── trajectory.py        # Trajectory data handling
├── io.py                # File I/O operations
├── orchestrator.py      # Main coordination
└── cli.py               # Command-line interface
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **simulator.py** | Create GillespieSimulator instances | `create_simulator()`, `calculate_stride()` |
| **replicate.py** | Run single replicates | `run_single_replicate()`, `process_simulation_states()` |
| **batch.py** | Process batches in parallel | `run_batch_simulations()` |
| **trajectory.py** | Handle trajectory data | `store_trajectory_data()`, `save_trajectories_to_parquet()` |
| **io.py** | Merge output files | `merge_output_files()` |
| **orchestrator.py** | Coordinate everything | `run_simulation()` |
| **cli.py** | Command-line interface | `main()`, argument parsing |

---

## 💡 Usage Examples

### 1. Basic Simulation

```python
from pathlib import Path
from src.simulation import run_simulation
from src.config.custom_type import SimulationConfig

# Quick configuration with defaults
config = SimulationConfig(
    prot_p_conc=100.0,
    prot_cooperativity=4.5,
    replicates=10
)

# Run simulation
run_simulation(
    file_path=Path("data/nucleosomes.tsv"),
    traj_outfile=Path("output/traj.parquet"),
    tsv_outfile=Path("output/summary.tsv"),
    config=config
)
```

### 2. High-Performance Simulation

```python
# Configuration for HPC cluster
config = SimulationConfig(
    # Many replicates
    replicates=1000,
    
    # Large batches for efficiency
    batch_size=50,
    
    # Use many cores
    n_workers=32,
    
    # Flush frequently for safety
    flush_every=5000,
    
    # Save trajectories
    save_trajectories=True,
    maxpoints_saved_trajectories=200
)

run_simulation(
    file_path=Path("data/large_dataset.tsv"),
    traj_outfile=Path("output/hpc_traj.parquet"),
    tsv_outfile=Path("output/hpc_summary.tsv"),
    config=config
)
```

### 3. Renucleation Study

```python
# Study with renucleation enabled
config = SimulationConfig(
    k_wrap=1.0,
    prot_p_conc=50.0,
    prot_cooperativity=3.0,
    
    # Enable renucleation
    renucleation=True,
    
    # Long simulation time
    tau_max=50000.0,
    tau_steps=2000,
    
    replicates=50
)

run_simulation(
    file_path=Path("data/nucleosomes.tsv"),
    traj_outfile=Path("output/renuc_traj.parquet"),
    tsv_outfile=Path("output/renuc_summary.tsv"),
    config=config
)
```

### 4. Parameter Sweep

```python
from pathlib import Path
from src.simulation import run_simulation
from src.config.custom_type import SimulationConfig

# Sweep over protamine concentrations
for prot_conc in [10, 50, 100, 200, 500]:
    config = SimulationConfig(
        prot_p_conc=prot_conc,
        prot_cooperativity=4.5,
        replicates=20,
        n_workers=8
    )
    
    # Create output directory
    output_dir = Path(f"output/sweep_conc_{prot_conc}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run simulation
    run_simulation(
        file_path=Path("data/nucleosomes.tsv"),
        traj_outfile=output_dir / "trajectories.parquet",
        tsv_outfile=output_dir / "summary.tsv",
        config=config
    )
    
    print(f"Completed concentration: {prot_conc}")
```

### 5. Custom Logger

```python
from src.utils.logger_util import get_logger

# Create custom logger
logger = get_logger(
    __name__,
    log_file=Path("logs/simulation.log"),
    level='DEBUG'
)

# Run with custom logger
config = SimulationConfig(prot_p_conc=100.0, replicates=10)

run_simulation(
    file_path=Path("data/nucleosomes.tsv"),
    traj_outfile=Path("output/traj.parquet"),
    tsv_outfile=Path("output/summary.tsv"),
    config=config,
    logger=logger  # Pass custom logger
)
```

### 6. Save/Load Configuration

```python
import json
from pathlib import Path
from src.config.custom_type import SimulationConfig

# Create configuration
config = SimulationConfig(
    k_wrap=1.0,
    prot_p_conc=100.0,
    prot_cooperativity=4.5,
    replicates=20,
    n_workers=10
)

# Save to JSON
config_dict = config.to_dict()
with open("configs/my_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Load from JSON
with open("configs/my_config.json", "r") as f:
    loaded_dict = json.load(f)
loaded_config = SimulationConfig.from_dict(loaded_dict)

# Use loaded configuration
from src.simulation import run_simulation

run_simulation(
    file_path=Path("data/nucleosomes.tsv"),
    traj_outfile=Path("output/traj.parquet"),
    tsv_outfile=Path("output/summary.tsv"),
    config=loaded_config
)
```

---

## 🔧 Advanced Usage

### Direct Module Access

For custom workflows, import specific modules:

```python
from src.simulation.simulator import create_simulator
from src.simulation.replicate import run_single_replicate
from src.simulation.batch import run_batch_simulations

# Create custom pipeline
simulator = create_simulator(...)
result = run_single_replicate(...)
```

### Batch Processing

```python
from src.simulation.batch import run_batch_simulations
from src.core.build_nucleosomes import nucleosome_generator
import src.core.helper.bkeep as bk

# Generate nucleosomes
gen = nucleosome_generator(file_path, k_wrap=1.0, binding_sites=14)

# Create batches
batches = bk.batcher(gen, batch_size=10)

# Process batches
for batch in batches:
    result = run_batch_simulations(
        batch=batch,
        build_params=config.build_params,
        tau_points=config.tau_points,
        ...
    )
```

---

## 🎯 Performance Tips

### 1. Batch Size
- **Small files (<1000 nucleosomes)**: `batch_size=10-20`
- **Medium files (1000-10000)**: `batch_size=50-100`
- **Large files (>10000)**: `batch_size=100-200`

### 2. Worker Count
- **Local machine**: `n_workers=CPU_count - 2`
- **HPC cluster**: `n_workers=allocated_cores`
- **Memory-limited**: Reduce workers to avoid OOM

### 3. Trajectory Saving
- **Development**: `save_trajectories=True`, `maxpoints_saved_trajectories=100`
- **Production**: `save_trajectories=False` (much faster)
- **Analysis**: Save only for subset of replicates

### 4. Flush Frequency
- **Fast storage**: `flush_every=10000`
- **Slow storage/network**: `flush_every=5000`
- **Reliability critical**: `flush_every=1000`

---

## 📊 Output Files

### Summary TSV
Contains aggregated results:
- `nucleosome_id`: Identifier
- `avg_rmst`: Average RMST
- `std_rmst`: Standard deviation
- `n_replicates`: Number of replicates

### Trajectory Parquet
Contains full trajectories (if enabled):
- `nucleosome_id`: Identifier
- `replicate`: Replicate number
- `time`: Time point
- `state`: Nucleosome state

---

## 🔄 Migration Guide

### From Old exec_sim.py

**Old way:**
```python
from src.scripts.exec_sim import run_simulation

run_simulation(
    file_path=path,
    traj_outfile=traj_path,
    tsv_outfile=tsv_path,
    k_wrap=1.0,
    binding_sites=14,
    prot_k_unbind=89.7,
    prot_k_bind=1.0,
    prot_p_conc=100.0,
    prot_cooperativity=4.5,
    tau_max=10000.0,
    tau_steps=1000,
    inf_protamine=True,
    renucleation=False,
    replicates=20,
    batch_size=10,
    n_workers=4,
    flush_every=10000,
    save_trajectories=False,
    maxpoints_saved_trajectories=100
)
```

**New way:**
```python
from src.simulation import run_simulation
from src.config.custom_type import SimulationConfig

config = SimulationConfig(
    k_wrap=1.0,
    prot_p_conc=100.0,
    prot_cooperativity=4.5,
    replicates=20,
    n_workers=4
    # Other parameters use defaults
)

run_simulation(
    file_path=path,
    traj_outfile=traj_path,
    tsv_outfile=tsv_path,
    config=config
)
```

---

## 📖 Documentation

Detailed documentation available in `docs/`:

- **`simulation_config_guide.md`** - Complete SimulationConfig guide
- **`simulation_refactoring_summary.md`** - Refactoring details
- **`exec_sim_refactoring.md`** - Quick reference

---

## ❓ FAQ

### Why use SimulationConfig?

**Benefits:**
- Reduces function signatures from 15+ to 4 parameters
- Provides validation and computed properties
- Makes configurations reusable and serializable
- Improves code readability and maintainability

### Is the old API still supported?

Yes! The old `exec_sim.py` still works but shows deprecation warnings. It delegates to the new modular package.

### How do I run on HPC cluster?

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=24:00:00

python -m src.simulation.cli \
    --infile data/nucleosomes.tsv \
    --storage_dir output \
    --prot_p_conc 100.0 \
    --prot_cooperativity 4.5 \
    --replicates 1000 \
    --n_workers 32 \
    --batch_size 50
```

### How do I debug simulations?

Enable debug logging:

```python
from src.utils.logger_util import get_logger

logger = get_logger(__name__, level='DEBUG')

run_simulation(..., logger=logger)
```

---

## 🐛 Troubleshooting

### Out of Memory

**Symptoms:** Process killed, memory errors

**Solutions:**
- Reduce `n_workers`
- Increase `flush_every` 
- Reduce `batch_size`
- Disable trajectory saving

### Slow Performance

**Symptoms:** Simulations take too long

**Solutions:**
- Increase `n_workers` (up to CPU count)
- Increase `batch_size`
- Disable trajectory saving
- Check disk I/O bottleneck

### Missing Output Files

**Symptoms:** Output files not created

**Solutions:**
- Check file paths exist
- Verify write permissions
- Check disk space
- Review logs for errors

---

## 📝 License

See main repository LICENSE file.

## 👥 Authors

- MY - Refactoring and modularization
- Original implementation contributors

---

## 🔗 Related Packages

- **`src.core`** - Core simulation logic
- **`src.config`** - Configuration management
- **`src.analysis`** - Analysis tools (RMST estimator)
- **`src.utils`** - Utility functions

---

**Last Updated:** 2025-01-16
