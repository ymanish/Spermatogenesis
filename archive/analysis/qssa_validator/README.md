# QSSA Validator Module

A modular package for validating the Quasi-Steady-State Approximation (QSSA) in nucleosome-protamine simulations.

## Overview

The QSSA is valid when protamine binding/unbinding equilibrates much faster than nucleosome wrapping/unwrapping dynamics. This module computes the timescale ratio:

```
epsilon = tau_prot / tau_slow
```

Where:
- `tau_prot`: Fast timescale for protamine binding/unbinding
- `tau_slow(i,j)`: Slow timescale for nucleosome wrapping/unwrapping at state (i,j)

**QSSA is valid when epsilon << 1** (typically epsilon <= 0.1).

## Module Structure

```
qssa_validator/
├── __init__.py          # Main API exports
├── core.py              # Timescale computation functions
├── validation.py        # QSSA validation logic
├── io.py                # File I/O and reporting
├── visualization.py     # Plotting functions
└── cli.py               # Command-line interface
```

## Quick Start

### Python API

```python
from pathlib import Path
from src.analysis.qssa_validator import validate_qssa
from src.core.build_nucleosomes import build_nucleosomes_from_file
from src.core.protamine import protamines

# Load nucleosomes
nucs = build_nucleosomes_from_file(
    file_path="data.tsv",
    k_wrap=21.0,
    kT=1.0,
    binding_sites=14
)

# Create protamine instance
prot = protamines(
    P_conc=100.0,
    cooperativity=0.0,
    k_unbind=100.0,
    k_bind=1.0,
    binding_sites=14
)

# Validate QSSA
result = validate_qssa(
    nucleosomes=nucs,
    protamines=prot,
    threshold=0.1,
    output_dir=Path("output/qssa"),
    verbose=True
)

# Check result
if result.system_qssa_valid:
    print("✓ QSSA is valid! Can use hybrid rejection simulator.")
else:
    print("✗ QSSA is invalid. Must use full Gillespie simulator.")
```

### Command-Line Interface

```bash
# Basic usage
python -m src.analysis.qssa_validator.cli \
    data/nucleosomes.tsv \
    --output-dir output/qssa \
    --verbose

# With custom parameters
python -m src.analysis.qssa_validator.cli \
    data/nucleosomes.tsv \
    --k-wrap 21.0 \
    --kT 1.0 \
    --binding-sites 14 \
    --prot-conc 100.0 \
    --prot-k-bind 1.0 \
    --prot-k-unbind 100.0 \
    --prot-cooperativity 0.0 \
    --threshold 0.1 \
    --output-dir output/qssa \
    --verbose
```

## Main API Functions

### High-Level

```python
from src.analysis.qssa_validator import validate_qssa

result = validate_qssa(
    nucleosomes,      # Nucleosome or Nucleosomes instance
    protamines,       # protamines instance
    threshold=0.1,    # QSSA validity threshold
    beta=1.0,         # Inverse temperature
    output_dir=None,  # Optional: save reports/data
    verbose=True      # Print summary
)
```

### Core Functions

```python
from src.analysis.qssa_validator import (
    compute_protamine_fast_timescale,
    validate_qssa_for_nucleosome,
    validate_qssa_for_system
)

# Compute tau_prot
tau_prot = compute_protamine_fast_timescale(prot, beta=1.0)

# Validate single nucleosome
nuc_result = validate_qssa_for_nucleosome(nuc, tau_prot, threshold=0.1)

# Validate entire system
sys_result = validate_qssa_for_system(nucs, prot, threshold=0.1, beta=1.0)
```

### I/O Functions

```python
from src.analysis.qssa_validator import (
    print_qssa_summary,
    generate_qssa_report,
    save_qssa_data
)

# Print to console
print_qssa_summary(result, verbose=True)

# Save text report
generate_qssa_report(result, Path("output/report.txt"), include_details=True)

# Save TSV data
save_qssa_data(result, Path("output/data.tsv"))
```

### Visualization

```python
from src.analysis.qssa_validator.visualization import (
    plot_epsilon_heatmap,
    plot_system_overview,
    plot_timescale_comparison,
    plot_all_visualizations
)

# Individual nucleosome heatmap
plot_epsilon_heatmap(nuc_result, threshold=0.1, save_path="epsilon_heatmap.png")

# System overview
plot_system_overview(sys_result, save_path="system_overview.png")

# Timescale comparison
plot_timescale_comparison(nuc_result, save_path="timescales.png")

# Generate all plots
plot_all_visualizations(sys_result, output_dir="output/plots", max_nucleosomes=5)
```

## Output Files

When `output_dir` is specified, the following files are generated:

```
output_dir/
├── qssa_validation_report.txt      # Detailed text report
├── qssa_validation_data.tsv        # Raw validation data
└── plots/                          # Visualization plots (optional)
    ├── system_overview.png
    └── individual_nucleosomes/
        ├── {nuc_id}_heatmap.png
        └── {nuc_id}_timescales.png
```

### TSV Data Columns

The `qssa_validation_data.tsv` file contains:

| Column | Description |
|--------|-------------|
| `nuc_id` | Nucleosome ID |
| `subid` | Nucleosome sub-ID |
| `i` | Left contact index |
| `j` | Right contact index |
| `n_open` | Total open contacts (i + L-1-j) |
| `tau_prot` | Protamine fast timescale (s) |
| `tau_slow` | Nucleosome slow timescale (s) |
| `epsilon` | Timescale ratio (tau_prot/tau_slow) |
| `threshold` | QSSA threshold used |
| `qssa_valid_local` | State-level QSSA validity |
| `qssa_valid_global` | Nucleosome-level QSSA validity |

## Data Classes

### QSSAValidationResult

Per-nucleosome validation result:

```python
@dataclass
class QSSAValidationResult:
    nuc_id: str
    subid: int
    tau_prot: float
    tau_slow: Dict[Tuple[int, int], float]
    epsilons: Dict[Tuple[int, int], float]
    eps_max: float
    qssa_valid: bool
    threshold: float
    qssa_valid_per_ij: Dict[Tuple[int, int], bool]
```

### SystemQSSAResult

System-wide validation result:

```python
@dataclass
class SystemQSSAResult:
    tau_prot: float
    num_nucleosomes: int
    num_valid: int
    num_invalid: int
    fraction_valid: float
    max_epsilon_overall: float
    nucleosome_results: List[QSSAValidationResult]
    system_qssa_valid: bool
```

## Examples

See `examples/example_qssa_validation.py` for complete examples:

```bash
python examples/example_qssa_validation.py
```

## Integration with Simulators

Based on QSSA validation results:

### QSSA Valid (epsilon << 1)
✓ Use **Hybrid Rejection Simulator**
- Faster computation
- Protamine dynamics integrated out
- Effective nucleosome-only model

```python
from src.core.hybrid_rejection_simulator import HybridRejectionSimulator

if result.system_qssa_valid:
    simulator = HybridRejectionSimulator(...)
```

### QSSA Invalid (epsilon ~ 1)
✗ Use **Full Gillespie Simulator**
- Complete dynamics
- Explicit protamine tracking
- No approximations

```python
from src.core.gillespie_simulator import GillespieSimulator

if not result.system_qssa_valid:
    simulator = GillespieSimulator(...)
```

## Theory

### Protamine Fast Timescale

```
tau_prot = 1 / (k_on + k_off_max)
```

Where:
- `k_on = k_bind * P_free`
- `k_off_max = k_unbind * exp(-2*beta*|J|)` (both neighbors bound)

### Nucleosome Slow Timescale

For each state (i,j):

```
tau_slow(i,j) = 1 / a_slow(i,j)
```

Where `a_slow(i,j)` is the total wrapping/unwrapping rate.

### QSSA Validity Criterion

For each state (i,j):

```
epsilon(i,j) = tau_prot / tau_slow(i,j) <= threshold
```

**System QSSA is valid** if ALL nucleosomes and ALL states satisfy the criterion.

## References

1. Segel, L. A., & Slemrod, M. (1989). The quasi-steady-state assumption: A case study in perturbation. *SIAM Review*, 31(3), 446-477.

2. Mastny, E. A., Haseltine, E. L., & Rawlings, J. B. (2007). Two classes of quasi-steady-state model reductions for stochastic kinetics. *The Journal of Chemical Physics*, 127(9), 094106.

## Author

MY, 2025-11-27
