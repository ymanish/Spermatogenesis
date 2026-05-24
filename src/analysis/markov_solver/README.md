# Markov Solver Module

A modular package for solving continuous-time Markov chains (CTMC) for nucleosome unwrapping dynamics with protamine binding effects.

## Table of Contents

- [Overview](#overview)
- [Physical Model](#physical-model)
- [Module Structure](#module-structure)
- [Quick Start](#quick-start)
- [Detailed API](#detailed-api)
- [Time Convention](#time-convention)
- [Examples](#examples)
- [Mathematical Background](#mathematical-background)

## Overview

This module solves for nucleosome detachment dynamics in the **fast protamine limit**, where protamine binding/unbinding is much faster than DNA wrapping/unwrapping. The system is described as a Markov chain in the (l,r) state space, where:

- **l**: number of left-side DNA contacts unwrapped
- **r**: number of right-side DNA contacts unwrapped
- **N**: total number of DNA-histone contacts (typically 14)

The nucleosome is considered **fully detached** when `l + r ≥ N` (all contacts broken).

### Key Features

✅ **Fast protamine approximation**: Closing rates are gated by equilibrium protamine occupancy  
✅ **Exact analytical solutions**: Via matrix exponential or ODE integration  
✅ **Multiple outputs**: Survival function S(t), MFPT, state probabilities P(t)  
✅ **Modular design**: Each component is independent and testable  
✅ **Dimensionless time**: Uses τ = k_wrap × t_physical for numerical stability  

## Physical Model

### State Space

States are represented as tuples `(l, r)`:
- **Transient states**: `l + r < N` (nucleosome partially wrapped)
- **Absorbing state**: `l + r ≥ N` (nucleosome fully detached)

Example for N=3:
```
(0,0) ← fully wrapped
(1,0), (0,1) ← 1 contact open
(2,0), (1,1), (0,2) ← 2 contacts open
ABSORBED ← 3+ contacts open (detached)
```

### Transition Rates

The generator matrix Q is built with the following transition rules:

#### Opening (Unwrapping)
DNA unwraps based on bare nucleosome free energy:
```
(l,r) → (l+1,r):  k_open_left  = k_wrap × exp(-ΔF_nuc/kT)
(l,r) → (l,r+1):  k_open_right = k_wrap × exp(-ΔF_nuc/kT)
```
where `ΔF_nuc = F_nuc(l+1,r) - F_nuc(l,r)` is the free energy change.

#### Closing (Rewrapping)
Closing is **gated by protamine occupancy** (fast protamine limit):
```
(l,r) → (l-1,r):  k_close_left  = k_wrap × P_free(l; βμ, βJ)
(l,r) → (l,r-1):  k_close_right = k_wrap × P_free(r; βμ, βJ)
```
where:
- `P_free(n)` = equilibrium probability that the boundary site is free of protamine
- `βμ = ln(k_bind × c / k_unbind)` = chemical potential (dimensionless)
- `βJ` = cooperativity parameter (dimensionless)

### Generator Matrix Q

The full generator is an (M+1) × (M+1) matrix where:
- **M** = number of transient states
- **Column j** = all transitions **from** state j
- **Row i** = transitions **to** state i
- **Diagonal** = -Σ(outgoing rates)

Block structure:
```
       | Transient | Absorbing |
-------|-----------|-----------|
Trans. |   Q_TT    |    0      |  M × (M+1)
-------|-----------|-----------|
Abs.   |   Q_AT    |    0      |  1 × (M+1)
```

## Module Structure

```
markov_solver/
├── __init__.py           # Public API exports
├── state_space.py        # (l,r) state enumeration
├── generator.py          # Q matrix construction
├── mfpt.py              # Mean first passage time
├── survival.py          # Survival function S(t)
├── nucleosome_utils.py  # Load nucleosome data
├── projection.py        # Project to observables
└── solver.py            # High-level interface
```

### Component Descriptions

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `state_space.py` | Enumerate transient and absorbing states | `build_state_space(N_MAX)` |
| `generator.py` | Build rate matrix Q with protamine effects | `build_full_Q_from_nucleosome(...)` |
| `mfpt.py` | Compute mean first passage times | `compute_mfpt_from_Q_TT(...)` |
| `survival.py` | Compute survival function S(t) | `compute_survival(...)` |
| `nucleosome_utils.py` | Load nucleosome free energy data | `load_nucleosomes_from_file(...)` |
| `projection.py` | Map (l,r) to observables like n_open | `project_to_open_sites(...)` |
| `solver.py` | Complete solver (all-in-one) | `solve_Q_TT_complete(...)` |

## Quick Start

### Basic Usage

```python
from src.analysis.markov_solver import (
    load_nucleosomes_from_file,
    build_full_Q_from_nucleosome,
    compute_survival,
    compute_mfpt_from_Q_TT
)
import numpy as np

# 1. Load nucleosome free energy data
nucs = load_nucleosomes_from_file(
    "data/nucleosome.tsv",
    k_wrap=1.0,
    max_nucs=5
)
nuc = nucs[0]

# 2. Define protamine parameters
protamine_params = {
    'k_bind': 1.0,          # (μM·s)^-1
    'k_unbind': 89.7,       # s^-1
    'p_conc': 10.0,         # μM
    'cooperativity': 0.0    # kT
}

# 3. Build generator matrix
Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
    nuc,
    k_wrap=1.0,
    protamine_params=protamine_params,
    dimensionless=True
)

# 4. Compute survival function
t_grid = np.linspace(0, 1000, 500)  # Dimensionless time τ
S = compute_survival(
    Q_TT,
    state_index,
    start_state=(0, 0),  # Fully wrapped
    t_grid=t_grid,
    method='expm'
)

# 5. Compute MFPT
mfpt, tau_vec = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state=(0, 0))
print(f"MFPT = {mfpt:.2f} τ")
```

### High-Level Interface

For convenience, use the all-in-one solver:

```python
from src.analysis.markov_solver import solve_Q_TT_complete

results = solve_Q_TT_complete(
    nuc,
    start_state=(0, 0),
    t_max=1000.0,
    n_points=500,
    method='expm',
    k_wrap=1.0,
    protamine_params=protamine_params
)

# Access results
S = results['survival']          # Survival function
mfpt = results['mfpt']           # Mean first passage time
t_grid = results['t_grid']       # Time grid
Q_TT = results['Q_TT']          # Generator matrix (transient block)
```

## Detailed API

### 1. `state_space.build_state_space(N_MAX)`

Enumerate all states in (l,r) space.

**Parameters:**
- `N_MAX` (int): Maximum number of contacts

**Returns:**
- `transient_states`: List of (l,r) tuples with l+r < N_MAX
- `absorbing_states`: List of (l,r) tuples with l+r == N_MAX
- `index_map`: Dict mapping (l,r) → index (for transient states only)

**Example:**
```python
from src.analysis.markov_solver import build_state_space

transient, absorbing, idx_map = build_state_space(N_MAX=14)
print(f"Number of transient states: {len(transient)}")
print(f"State (0,0) has index: {idx_map[(0,0)]}")
```

### 2. `generator.build_full_Q_from_nucleosome(...)`

Build the generator matrix Q with protamine effects.

**Parameters:**
- `nucleosome`: Nucleosome instance with G_mat (free energy matrix)
- `k_wrap` (float): Wrapping rate constant (s^-1)
- `protamine_params` (dict): Protamine parameters:
  - `'k_bind'`: Binding rate (μM·s)^-1
  - `'k_unbind'`: Unbinding rate s^-1
  - `'p_conc'`: Protamine concentration (μM)
  - `'cooperativity'`: Cooperativity J (kT)
- `kT` (float, optional): Thermal energy (default: nucleosome.kT)
- `binding_sites` (int, optional): Number of sites (default: nucleosome.binding_sites)
- `sparse` (bool): Use sparse matrices (default: False)
- `dimensionless` (bool): Return Q in dimensionless units (default: True)

**Returns:**
- `Q_full`: (M+1) × (M+1) full generator
- `Q_TT`: M × M transient-to-transient block
- `Q_AT`: 1 × M absorbing-from-transient block
- `states`: List of transient (l,r) states
- `state_index`: Dict mapping (l,r) → index
- `abs_index`: Index of absorbing state (= M)

**Example:**
```python
Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
    nuc,
    k_wrap=1.0,
    protamine_params={
        'k_bind': 1.0,
        'k_unbind': 89.7,
        'p_conc': 10.0,
        'cooperativity': 0.0
    },
    dimensionless=True
)
```

### 3. `survival.compute_survival(...)`

Compute survival function S(t) = P(not absorbed by time t).

**Parameters:**
- `Q_trans` (ndarray): Transient generator Q_TT (M × M)
- `index_map` (dict): Mapping (l,r) → index for transient states
- `start_state` (tuple): Initial state (l,r)
- `t_grid` (ndarray): Time points (dimensionless τ)
- `method` (str): 'expm' or 'ode'
- `return_states` (bool): If True, also return P_states(t)

**Returns:**
- `S` (ndarray): Survival probability at each time in t_grid
- `P_states` (ndarray, optional): State probabilities (len(t_grid) × M)

**Methods:**
- `'expm'`: Matrix exponential (accurate, slower for large M)
- `'ode'`: ODE integration (faster, good for large M)

**Example:**
```python
t_grid = np.linspace(0, 1000, 500)

# Basic usage
S = compute_survival(Q_TT, state_index, (0,0), t_grid, method='expm')

# With state probabilities
S, P_states = compute_survival(
    Q_TT, state_index, (0,0), t_grid,
    method='expm', return_states=True
)
```

### 4. `mfpt.compute_mfpt_from_Q_TT(...)`

Compute mean first passage time to absorption.

**Parameters:**
- `Q_TT` (ndarray): Transient generator (M × M)
- `index_map` (dict): State → index mapping
- `start_state` (tuple): Initial state (l,r)

**Returns:**
- `mfpt` (float): Mean first passage time from start_state
- `tau_vec` (ndarray): MFPT from all transient states

**Mathematical Formula:**
```
Q_TT^T @ τ = -1
```
where τ is the vector of MFPTs.

**Example:**
```python
mfpt, tau_vec = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state=(0,0))
print(f"MFPT from (0,0): {mfpt:.2f} τ")
print(f"MFPT range: [{tau_vec.min():.2f}, {tau_vec.max():.2f}] τ")
```

### 5. `nucleosome_utils.load_nucleosomes_from_file(...)`

Load nucleosome free energy data from TSV file.

**Parameters:**
- `file_path` (Path): Path to nucleosome data file
- `k_wrap` (float): Wrapping rate (default: 1.0)
- `max_nucs` (int, optional): Maximum nucleosomes to load
- `subids_range` (tuple, optional): Range of subids (min, max)

**Returns:**
- `nucleosomes` (list): List of Nucleosome instances

**Example:**
```python
nucs = load_nucleosomes_from_file(
    "data/nucleosome.tsv",
    k_wrap=1.0,
    max_nucs=10
)

# Access nucleosome data
nuc = nucs[0]
print(f"Nucleosome {nuc.id}-{nuc.subid}")
print(f"G_mat shape: {nuc.G_mat.shape}")
print(f"Binding sites: {nuc.binding_sites}")
```

### 6. `solver.solve_Q_TT_complete(...)`

High-level all-in-one solver.

**Parameters:**
- `nucleosome`: Nucleosome instance
- `start_state` (tuple): Initial state (default: (0,0))
- `t_max` (float): Maximum time (dimensionless)
- `n_points` (int): Number of time points
- `method` (str): 'expm' or 'ode'
- `sparse` (bool): Use sparse matrices
- `k_wrap` (float): Wrapping rate
- `protamine_params` (dict): Protamine parameters
- `kT` (float, optional): Thermal energy

**Returns:**
- `results` (dict): Contains:
  - `'survival'`: S(t) array
  - `'mfpt'`: Mean first passage time
  - `'t_grid'`: Time grid
  - `'Q_TT'`: Transient generator
  - `'Q_full'`: Full generator
  - `'states'`: List of states
  - `'state_index'`: State → index map
  - `'k_wrap'`: Wrapping rate used

## Time Convention

### Dimensionless Time

All time variables use **dimensionless time** τ defined as:
```
τ = k_wrap × t_physical
```

This convention:
- ✅ Makes Q_TT dimensionless when multiplied by τ
- ✅ Ensures numerical stability
- ✅ Allows direct comparison with Gillespie simulations
- ✅ Simplifies parameter scans

### Time Grids

**In Markov solver:**
```python
t_grid = np.linspace(0, tau_max, n_points)  # Dimensionless τ
S = compute_survival(Q_TT, state_index, (0,0), t_grid)
```

**Converting to physical time:**
```python
t_physical = t_grid / k_wrap  # seconds
```

**Example:**
```python
k_wrap = 1.0  # s^-1
tau_max = 1000.0  # dimensionless

t_grid = np.linspace(0, tau_max, 500)
# → τ ∈ [0, 1000]
# → t_physical ∈ [0, 1000] seconds (when k_wrap=1.0)

S = compute_survival(Q_TT, state_index, (0,0), t_grid)

# Plot in physical time
import matplotlib.pyplot as plt
plt.plot(t_grid / k_wrap, S)
plt.xlabel('Time (seconds)')
plt.ylabel('Survival S(t)')
```

### Generator Matrix Units

When `dimensionless=True` (default):
```python
Q_full, Q_TT, ... = build_full_Q_from_nucleosome(..., dimensionless=True)
```
- `Q_TT` is in units of k_wrap (effectively dimensionless)
- Rates are scaled: `k_close = k_wrap × P_free(n)`
- Product `Q_TT × τ` is dimensionless

When `dimensionless=False`:
```python
Q_full, Q_TT, ... = build_full_Q_from_nucleosome(..., dimensionless=False)
```
- `Q_TT` has physical units (s^-1)
- Use with physical time t_grid in seconds

## Examples

### Example 1: Basic Survival Function

```python
from src.analysis.markov_solver import (
    load_nucleosomes_from_file,
    build_full_Q_from_nucleosome,
    compute_survival
)
import numpy as np
import matplotlib.pyplot as plt

# Load data
nucs = load_nucleosomes_from_file("data/nucleosome.tsv", k_wrap=1.0, max_nucs=1)
nuc = nucs[0]

# Protamine parameters
protamine_params = {
    'k_bind': 1.0,
    'k_unbind': 89.7,
    'p_conc': 10.0,
    'cooperativity': 0.0
}

# Build Q matrix
Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
    nuc, k_wrap=1.0, protamine_params=protamine_params
)

# Compute survival
t_grid = np.linspace(0, 1000, 500)
S = compute_survival(Q_TT, state_index, (0,0), t_grid, method='expm')

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_grid, S, 'b-', lw=2)
plt.xlabel('Dimensionless Time (τ)', fontsize=12)
plt.ylabel('Survival S(t)', fontsize=12)
plt.title('Nucleosome Survival Function', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: MFPT Calculation

```python
from src.analysis.markov_solver import compute_mfpt_from_Q_TT

# Compute MFPT
mfpt, tau_vec = compute_mfpt_from_Q_TT(Q_TT, state_index, start_state=(0,0))

print(f"MFPT from (0,0): {mfpt:.2f} τ")
print(f"MFPT range: [{tau_vec.min():.2f}, {tau_vec.max():.2f}] τ")

# Convert to physical time
t_phys = mfpt / nuc.k_wrap
print(f"MFPT (physical): {t_phys:.4f} seconds")
```

### Example 3: State Probabilities

```python
# Compute survival with state probabilities
S, P_states = compute_survival(
    Q_TT, state_index, (0,0), t_grid,
    method='expm', return_states=True
)

# Plot probability evolution for specific states
plt.figure(figsize=(10, 6))
for (l, r), idx in list(state_index.items())[:5]:
    plt.plot(t_grid, P_states[:, idx], label=f'({l},{r})')

plt.xlabel('Time (τ)')
plt.ylabel('Probability')
plt.title('State Probability Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Verify probability conservation
total_prob = P_states.sum(axis=1)
absorbed_prob = 1 - total_prob
print(f"Final absorbed probability: {absorbed_prob[-1]:.6f}")
```

### Example 4: Parameter Scan

```python
import numpy as np
import matplotlib.pyplot as plt

# Scan protamine concentration
concentrations = [0.0, 1.0, 10.0, 100.0]
t_grid = np.linspace(0, 1000, 500)

fig, ax = plt.subplots(figsize=(10, 6))

for p_conc in concentrations:
    protamine_params = {
        'k_bind': 1.0,
        'k_unbind': 89.7,
        'p_conc': p_conc,
        'cooperativity': 0.0
    }
    
    Q_full, Q_TT, Q_AT, states, state_index, abs_index = build_full_Q_from_nucleosome(
        nuc, k_wrap=1.0, protamine_params=protamine_params
    )
    
    S = compute_survival(Q_TT, state_index, (0,0), t_grid, method='expm')
    mfpt, _ = compute_mfpt_from_Q_TT(Q_TT, state_index, (0,0))
    
    ax.plot(t_grid, S, label=f'c={p_conc} μM (MFPT={mfpt:.1f})')

ax.set_xlabel('Time (τ)', fontsize=12)
ax.set_ylabel('Survival S(t)', fontsize=12)
ax.set_title('Effect of Protamine Concentration', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Example 5: Comparing Methods

```python
import time

# Compare expm vs ODE methods
t_grid = np.linspace(0, 1000, 500)

# Method 1: Matrix exponential
start = time.time()
S_expm = compute_survival(Q_TT, state_index, (0,0), t_grid, method='expm')
time_expm = time.time() - start

# Method 2: ODE solver
start = time.time()
S_ode = compute_survival(Q_TT, state_index, (0,0), t_grid, method='ode')
time_ode = time.time() - start

# Compare
diff = np.max(np.abs(S_expm - S_ode))
print(f"expm time: {time_expm:.3f} s")
print(f"ODE time:  {time_ode:.3f} s")
print(f"Max |difference|: {diff:.2e}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(t_grid, S_expm, 'b-', label='expm', lw=2)
ax1.plot(t_grid, S_ode, 'r--', label='ODE', lw=2, alpha=0.7)
ax1.set_ylabel('S(t)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_grid, S_expm - S_ode, 'k-', lw=1.5)
ax2.axhline(0, color='gray', ls='--', alpha=0.5)
ax2.set_xlabel('Time (τ)')
ax2.set_ylabel('Difference')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Mathematical Background

### Generator Matrix

The generator Q satisfies:
```
dP(t)/dt = Q @ P(t)
```

**Column-sum convention**: Column j contains all transitions FROM state j.

**Properties:**
- Off-diagonal: Q[i,j] ≥ 0 (rate from j to i)
- Diagonal: Q[j,j] = -Σ(outgoing rates from j)
- Column sums: Σ_i Q[i,j] = 0 (probability conservation)

### Survival Function

The survival function is the probability of not being absorbed by time t:
```
S(t) = Σ_transient P_i(t) = 1 - P_absorbed(t)
```

Computed via:
```
P(t) = exp(Q_TT × t) @ P(0)
S(t) = Σ P_i(t)
```

### Mean First Passage Time

The MFPT satisfies the linear system:
```
Q_TT^T @ τ = -1
```

Solution:
```
τ = -(Q_TT^T)^(-1) @ 1
```

For a single starting state (l,r):
```
mfpt = τ[state_index[(l,r)]]
```

### Probability Conservation

At all times:
```
Σ_transient P_i(t) + P_absorbed(t) = 1
```

Verification:
```python
P_trans = P_states.sum(axis=1)  # Sum over transient states
P_absorbed = 1 - P_trans
# Check: P_trans + P_absorbed ≈ 1.0
```

## Testing and Validation

### Unit Tests

Run tests for individual components:
```bash
conda activate nucleosome
python src/analysis/test_markov_solver.py
```

### Verification Script

Check probability conservation:
```bash
python check_probability_conservation.py
```

### Time Consistency

Verify dimensionless time alignment:
```bash
python verify_time_consistency.py
```

## Performance Tips

### For Large Systems (M > 100)

1. **Use sparse matrices:**
   ```python
   Q_full, Q_TT, ... = build_full_Q_from_nucleosome(..., sparse=True)
   ```

2. **Use ODE solver:**
   ```python
   S = compute_survival(Q_TT, state_index, (0,0), t_grid, method='ode')
   ```

3. **Reduce time points:**
   ```python
   t_grid = np.linspace(0, 1000, 100)  # Fewer points
   ```

### For Parameter Scans

Reuse state space enumeration:
```python
states, absorbing, index_map = build_state_space(N_MAX=14)

for p_conc in concentrations:
    protamine_params['p_conc'] = p_conc
    Q_full, Q_TT, ... = build_full_Q_from_nucleosome(...)
    # Compute observables...
```

## Troubleshooting

### Issue: Numerical instability in expm

**Solution:** Use ODE solver or reduce time step:
```python
S = compute_survival(Q_TT, state_index, (0,0), t_grid, method='ode')
```

### Issue: Probability not conserved

**Check:**
1. Column sums of Q should be zero
2. Use `check_probability_conservation.py`

### Issue: Negative probabilities

**Cause:** Time step too large for matrix exponential

**Solution:**
```python
# Use smaller time grid
t_grid = np.linspace(0, 1000, 1000)  # More points
# Or use ODE
S = compute_survival(Q_TT, state_index, (0,0), t_grid, method='ode')
```

## References

- **Markov Chain Theory**: Norris, J. R. (1998). *Markov Chains*. Cambridge University Press.
- **Fast Protamine Limit**: QSSA validation documentation has been archived with the older QSSA workflow materials.

## See Also

- `compare_gillespie_vs_markov.py` - Comparison script

## Authors

- MY (2025)

## License

MIT License - See LICENSE file for details.
