# QSSA Validation for Nucleosome Simulations

## Overview

The **Quasi-Steady-State Approximation (QSSA)** allows you to simplify your simulations when protamines equilibrate much faster than nucleosome dynamics. This module validates whether QSSA is applicable for your simulation parameters.

## Theory

### Timescale Separation

Your system has two types of dynamics:

1. **Fast**: Protamine binding/unbinding (timescale τ_prot)
2. **Slow**: Nucleosome wrapping/unwrapping (timescale τ_slow(n))

The timescale ratio is:

```
ε(n) = τ_prot / τ_slow(n)
```

### QSSA Validity Criterion

QSSA is valid when **ε << 1** (typically **ε ≤ 0.1**)

- ✓ **QSSA valid (ε ≤ 0.1)**: Protamines equilibrate ~10× faster
  - Can use effective nucleosome-only model
  - Integrate out protamine degrees of freedom
  - Much faster simulations

- ✗ **QSSA invalid (ε > 0.1)**: Timescales are comparable
  - Must use full protamine-resolved Gillespie simulation
  - Track both nucleosome and protamine states explicitly

## Implementation

### Computing τ_prot (Protamine Fast Timescale)

```python
k_on = k_bind * p_conc
k_off_max = k_unbind * exp(-2 * β * |J|)  # Maximum cooperativity
τ_prot = 1 / (k_on + k_off_max)
```

**Key points:**
- `k_off_max` occurs when protamine has both neighbors bound
- Cooperativity J > 0 stabilizes binding → slower unbinding
- Higher p_conc → faster binding → smaller τ_prot

### Computing τ_slow(n) (Nucleosome Slow Timescale)

For each level n (total number of open contacts):

1. Identify all states (i,j) with that n
2. Compute exit rate from each state:
   - Unwrapping left: rate ∝ exp(-ΔG_left/kT)
   - Unwrapping right: rate ∝ exp(-ΔG_right/kT)
   - Wrapping left: rate = k_wrap
   - Wrapping right: rate = k_wrap
3. Average rates using Boltzmann weights
4. τ_slow(n) = 1 / average_rate

**Implementation in `Nucleosome` class:**
```python
tau_slow = nuc.compute_tau_slow_per_n()
# Returns: {0: τ_0, 1: τ_1, ..., 13: τ_13}
```

### Your G_mat Representation

Your energy matrix `G_mat[i, j]` encodes:
- **i**: Number of left-open contacts
- **j**: Matrix column, where `right_open = (L-1) - j`
- **Total open**: `n = i + (L-1-j)`
- **Upper triangle** (i ≤ j): Valid states
- **Outside triangle**: Fully open/detached state (energy = 0)

**Example (L=14):**
```
G_mat[0, 13] → 0 left, 0 right open → n=0 (fully wrapped)
G_mat[1, 12] → 1 left, 1 right open → n=2
G_mat[7, 6]  → 7 left, 7 right open → n=14 (outside triangle → detached)
```

## Usage

### Quick Validation

```python
from src.core.build_nucleosomes import nucleosome_generator, Nucleosomes
from src.core.protamine import protamines
from src.analysis.qssa_validator import validate_qssa_for_system, print_qssa_summary

# Load nucleosomes
gen = nucleosome_generator(file_path="data.tsv", k_wrap=22.0)
nucs = Nucleosomes(k_wrap=22.0, nucleosomes=list(gen))

# Create protamine instance
prot = protamines(
    k_unbind=0.23,
    k_bind=2113,
    p_conc=0.1,  # μM
    cooperativity=4.5
)

# Validate QSSA
result = validate_qssa_for_system(nucs, prot, threshold=0.1)
print_qssa_summary(result, verbose=True)
```

### Interpreting Results

#### Example Output 1: QSSA Valid ✓

```
QSSA VALIDATION SUMMARY
================================================================================

Protamine Fast Timescale:
  tau_prot = 2.341e-04 seconds

System Overview:
  Total nucleosomes: 10
  QSSA valid: 10 (100.0%)
  QSSA invalid: 0
  Maximum epsilon: 0.0823

✓ SYSTEM QSSA IS VALID
  → Can use effective nucleosome-only model
  → Protamines can be integrated out
```

**Interpretation:**
- τ_prot ≈ 0.23 ms (very fast)
- All nucleosomes have ε < 0.1
- Protamines equilibrate ~12× faster than nucleosomes
- **Action**: Use reduced model (faster simulations)

#### Example Output 2: QSSA Invalid ✗

```
QSSA VALIDATION SUMMARY
================================================================================

Protamine Fast Timescale:
  tau_prot = 4.731e-03 seconds

System Overview:
  Total nucleosomes: 10
  QSSA valid: 3 (30.0%)
  QSSA invalid: 7
  Maximum epsilon: 0.4521

✗ SYSTEM QSSA IS INVALID
  → Must use full protamine-resolved Gillespie simulation
  → Protamine dynamics too slow relative to nucleosomes
```

**Interpretation:**
- τ_prot ≈ 4.7 ms (slower)
- Some nucleosomes have ε > 0.1 (up to 0.45)
- Timescales are comparable
- **Action**: Use full Gillespie (no shortcuts)

### Per-Nucleosome Analysis

```python
result = validate_qssa_for_system(nucs, prot)

for nuc_result in result.nucleosome_results:
    print(f"Nucleosome {nuc_result.nuc_id}:")
    print(f"  Status: {'VALID' if nuc_result.qssa_valid else 'INVALID'}")
    print(f"  Max epsilon: {nuc_result.eps_max:.4f}")
    
    # Find problematic levels
    for n, eps in nuc_result.epsilons.items():
        if eps > 0.05:
            print(f"    Level n={n}: ε={eps:.4f}")
```

### Saving Results

```python
from src.analysis.qssa_validator import generate_qssa_report, save_qssa_data

# Save detailed report
generate_qssa_report(result, "output/qssa_report.txt")

# Save data for analysis
save_qssa_data(result, "output/qssa_data.tsv")
```

## Parameter Scan

Compare QSSA validity across different conditions:

```python
conditions = [
    {"p_conc": 0.1, "cooperativity": 4.5},
    {"p_conc": 1.0, "cooperativity": 4.5},
    {"p_conc": 10.0, "cooperativity": 4.5},
]

for cond in conditions:
    prot = protamines(k_unbind=0.23, k_bind=2113, **cond)
    result = validate_qssa_for_system(nucs, prot)
    
    print(f"p_conc={cond['p_conc']} μM:")
    print(f"  tau_prot = {result.tau_prot:.6e} s")
    print(f"  max(ε) = {result.max_epsilon_overall:.4f}")
    print(f"  QSSA: {'✓ VALID' if result.system_qssa_valid else '✗ INVALID'}")
```

## Expected Trends

### Effect of Protamine Concentration

| p_conc (μM) | k_on | τ_prot | ε | QSSA Valid? |
|-------------|------|---------|---|-------------|
| 0.01        | Low  | Large   | Small | ✓ Usually valid |
| 0.1         | Medium | Medium | Medium | ✓ Often valid |
| 1.0         | High | Small | Large | ✗ May be invalid |
| 10.0        | Very high | Very small | Very large | ✗ Usually invalid |

**Rule of thumb:** Higher concentration → faster binding → smaller τ_prot → larger ε → QSSA less likely

### Effect of Cooperativity

| J (kBT) | k_off_max | τ_prot | ε | QSSA Valid? |
|---------|-----------|---------|---|-------------|
| 0       | k_unbind  | Small | Large | ✗ Less likely |
| 4.5     | k_unbind × exp(-9) | Larger | Smaller | ✓ More likely |
| 10      | k_unbind × exp(-20) | Very large | Very small | ✓ Usually valid |

**Rule of thumb:** Higher cooperativity → slower unbinding → larger τ_prot → smaller ε → QSSA more likely

## Integration with Simulation Workflow

### Before Running Simulations

```python
# 1. Load a small sample of nucleosomes
gen = nucleosome_generator(file_path, k_wrap=22.0)
nucs_sample = Nucleosomes(k_wrap=22.0, nucleosomes=list(itertools.islice(gen, 20)))

# 2. Test your parameters
prot = protamines(k_unbind=0.23, k_bind=2113, p_conc=0.1, cooperativity=4.5)
result = validate_qssa_for_system(nucs_sample, prot)

# 3. Decide which model to use
if result.system_qssa_valid:
    print("✓ Use reduced model (effective nucleosome dynamics)")
    # TODO: Implement reduced model
else:
    print("✗ Use full Gillespie simulation")
    # Use existing simulation code
```

### In Cluster Jobs

Add QSSA check to your job scripts:

```bash
#!/bin/bash
#SBATCH --job-name=qssa_check

python examples/example_qssa_validation.py \
    --file data/001.tsv \
    --k-wrap 22.0 \
    --p-conc 0.1 \
    --cooperativity 4.5 \
    --output output/qssa/

# If QSSA valid, use reduced model
# If QSSA invalid, launch full Gillespie
```

## API Reference

### Main Functions

#### `validate_qssa_for_system(nucs, prot, threshold=0.1, beta=1.0)`

Validate QSSA for entire system.

**Parameters:**
- `nucs`: Nucleosomes instance
- `prot`: protamines instance
- `threshold`: QSSA validity threshold (default 0.1)
- `beta`: Inverse temperature (default 1.0)

**Returns:** `SystemQSSAResult`

#### `compute_protamine_fast_timescale(prot, beta=1.0)`

Compute τ_prot for protamine dynamics.

**Returns:** float (seconds)

#### `validate_qssa_for_nucleosome(nuc, tau_prot, threshold=0.1)`

Validate QSSA for single nucleosome.

**Returns:** `QSSAValidationResult`

### Data Structures

#### `SystemQSSAResult`

```python
@dataclass
class SystemQSSAResult:
    tau_prot: float                          # Protamine timescale (s)
    num_nucleosomes: int                     # Total nucleosomes
    num_valid: int                           # Passing QSSA
    num_invalid: int                         # Failing QSSA
    fraction_valid: float                    # 0.0 to 1.0
    max_epsilon_overall: float               # Worst case ε
    nucleosome_results: List[...]            # Per-nucleosome details
    system_qssa_valid: bool                  # True if all pass
```

#### `QSSAValidationResult`

```python
@dataclass
class QSSAValidationResult:
    nuc_id: str                             # Nucleosome ID
    subid: int                              # Sub-ID
    tau_prot: float                         # Protamine timescale
    tau_slow: Dict[int, float]              # {n: τ_slow(n)}
    epsilons: Dict[int, float]              # {n: ε(n)}
    eps_max: float                          # max(ε)
    qssa_valid: bool                        # ε_max ≤ threshold
    threshold: float                        # Threshold used
```

## Troubleshooting

### Q: All nucleosomes fail QSSA, what do I do?

**A:** You must use full Gillespie simulation. Consider:
1. Reducing protamine concentration (lower p_conc)
2. Increasing cooperativity (higher J)
3. Using shorter simulation times

### Q: Can I still run simulations if QSSA fails?

**A:** Yes! QSSA validation doesn't prevent simulations, it just tells you:
- **QSSA valid**: Can use faster reduced model (if implemented)
- **QSSA invalid**: Must use full Gillespie (your current approach)

### Q: How do I implement the reduced model?

**A:** If QSSA is valid, you can:
1. Compute effective nucleosome energies with protamine effects averaged
2. Simulate only nucleosome dynamics (no explicit protamine states)
3. Much faster, but requires implementing effective model

### Q: What if ε is close to 0.1?

**A:** The threshold is somewhat arbitrary:
- ε < 0.05: Very safe, QSSA clearly valid
- 0.05 ≤ ε ≤ 0.15: Borderline, use caution
- ε > 0.15: QSSA clearly invalid

For borderline cases, compare results from both models.


## Examples

See `examples/example_qssa_validation.py` for complete working examples.

## Contact

For questions or issues with QSSA validation, contact the development team.
