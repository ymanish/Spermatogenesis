# NucFreeEnergy

This module computes the free energy of DNA under nucleosome-like midstep constraints.
The main entry point in this directory is `binding_model.py`, especially
`binding_model_free_energy(...)`.

## What `binding_model_free_energy` computes

The function evaluates the free energy of a DNA segment with:

- sequence-dependent free DNA mechanics from `free_gs` and `free_M`
- nucleosome target geometry from `nuc_mu0_full`
- nucleosome stiffness from `nuc_K_full`
- optional opening from the left and/or right through `left_open` and `right_open`

It returns a dictionary with:

- `F`: total free energy of the constrained model
- `F_entropy`: entropic contribution
- `F_enthalpy`: mismatch / deformation penalty
- `F_Ajacob`: Jacobian term from the variable transformation
- `F_freedna`: free-energy baseline of the corresponding unconstrained DNA window

Note: in the trivial branch with too few remaining constraints, the code returns
`F_jacob` instead of `F_Ajacob`. That is just how the current implementation is written.

## Inputs in practical terms

- `free_gs`: free-DNA ground state for the sequence, one 6D step per base-step
- `free_M`: free-DNA stiffness matrix for the same sequence, shape `(6N, 6N)`
- `nuc_mu0_full`: reference nucleosome midstep triads for the fully wrapped state
- `nuc_K_full`: stiffness matrix for those nucleosome midstep constraints
- `left_open`, `right_open`: how many nucleosome midstep constraints are removed from the left or right edge
- `use_correction`: whether to run the second-pass correction that improves the enthalpy estimate

## Step-by-step logic of the free-energy calculation

### 1. Choose which nucleosome contacts are still active

`get_midstep_locations(left_open, right_open)` starts from the hard-coded list of
28 nucleosomal midstep locations and removes:

- the first `left_open` entries
- the last `right_open` entries

So increasing `left_open` or `right_open` means fewer active nucleosome constraints.

### 2. Crop the DNA problem to the remaining wrapped window

After the active midsteps are chosen, the code keeps only the DNA region between:

- the first remaining constrained midstep
- the last remaining constrained midstep

This is important: changing `left_open` or `right_open` does not only weaken the
constraint set. It also changes the DNA window that is actually evaluated.

### 3. Build the free-DNA geometry between active midsteps

`midstep_groundstate_se3(free_gs, midstep_constraint_locations)` composes the
sequence-dependent free-DNA ground-state steps between neighboring active
midsteps. This gives `sks`, the geometry that the sequence would naturally prefer
between those constraint points.

Conceptually:

- `sks` = what the sequence wants to do
- `nuc_mu0` = what the nucleosome wants the same span to do

### 4. Slice the nucleosome reference to match the opened state

`_slice_nucleo_params(nuc_mu0_full, nuc_K_full, left_open, right_open)` removes
the same left and right entries from the nucleosome reference geometry and
stiffness.

So `left_open` and `right_open` keep the DNA and nucleosome models aligned.

### 5. Change variables from local base-step coordinates to composite midstep coordinates

`midstep_composition_transformation(...)` builds a linear transformation that
replaces selected local DNA variables by composite variables spanning each
midstep interval.

Why this is done:

- the nucleosome constraints are defined on composite midstep-to-midstep motion
- the free-DNA stiffness is originally written in local base-step coordinates
- the transformation puts both descriptions into compatible coordinates

The transformed stiffness is:

`M_transformed = T^{-T} free_M T^{-1}`

implemented efficiently with sparse LU instead of explicitly forming `T^{-1}`.

### 6. Reorder variables so the constrained ones are grouped at the end

The transformed matrix is rearranged into block form:

```text
[ M_R   M_RM ]
[ M_RM' M_M  ]
```

where:

- `R` are the unconstrained/free variables
- `M` are the composite variables that couple to the nucleosome model

### 7. Integrate out the unconstrained DNA variables

The code computes the Schur complement:

`M_Mp = M_M - M_RM^T M_R^{-1} M_RM`

This is the effective stiffness seen by the constrained midstep variables after
the remaining free DNA degrees of freedom have been marginalized out.

This is the main statistical-mechanics reduction in the script.

### 8. Compare free-DNA preference against nucleosome preference

`coordinate_transformation(...)` builds:

- `B`: maps nucleosome midpoint variables into the composite coordinate system
- `Pbar`: the mismatch between the free-DNA composite geometry and the nucleosome reference geometry

Then the combined stiffness is formed:

`Kcomb = nuc_K + B^T M_Mp B`

Interpretation:

- `nuc_K` penalizes deviations from the nucleosome reference
- `B^T M_Mp B` is the effective DNA stiffness seen in the same coordinates

The minimizer with respect to nucleosome variables is solved from:

`alpha = -Kcomb^{-1} B^T M_Mp Pbar`

and the corresponding constrained composite ground state is:

`Y_C = Pbar + B alpha`

### 9. Recover the remaining free variables

The code solves for:

`gamma = -M_R^{-1} M_RM Y_C`

This gives the optimal values of the unconstrained variables conditioned on the
constrained ones.

### 10. Optional correction pass

If `use_correction=True`, the code reconstructs the full optimal deformation,
rebuilds the midstep transformation around that corrected state, and recomputes
the enthalpy term using `midstep_composition_transformation_correction(...)`.

This is meant to improve the treatment of nonlinear geometry in the composed
midstep variables.

### 11. Compute the free-energy terms

The final free energy is split into:

- entropy from `M_R`
- entropy from `Kcomb`
- Jacobian from the transformation
- enthalpy from the residual mismatch

In code:

- `F_entropy = F_A + F_R + F_K`
- `F_enthalpy` comes from the corrected mismatch quadratic form
- `F = F_entropy + F_enthalpy`

The free-DNA baseline `F_freedna` is also computed for the same cropped window so
you can compare:

- constrained model free energy: `F`
- unconstrained free-DNA free energy: `F_freedna`
- binding penalty relative to free DNA: `F - F_freedna`

## What happens when `left_open` or `right_open` change?

For the same underlying sequence, increasing `left_open` or `right_open` changes
the free energy because the model itself changes in three coupled ways.

### 1. Fewer nucleosome constraints remain

You remove constraint points from the chosen end. That usually makes the DNA less
restricted and tends to reduce the deformation penalty, because the sequence has
more freedom to follow its own preferred geometry.

### 2. The nucleosome reference model is shortened

The arrays `nuc_mu0_full` and `nuc_K_full` are sliced to the smaller wrapped
region. This means you are no longer paying for the removed left/right contacts.

### 3. The evaluated DNA window is also shortened

The function crops `free_gs` and `free_M` to the span between the first and last
remaining active midsteps. So the absolute free energy can change simply because
the number of included degrees of freedom changed.

This means:

- changing `left_open` / `right_open` is not just "same system with weaker constraints"
- it is "a smaller wrapped segment with a different set of active contacts"

## Expected trend with opening

For a fixed sequence, opening from either side usually pushes the model toward the
free-DNA limit:

- `F_enthalpy` often decreases because the remaining wrapped segment can fit the
  nucleosome reference more easily
- the entropy terms also change because the number and coupling of constrained
  variables changes
- as more contacts are removed, `F` should generally approach `F_freedna` for the
  remaining window

But the change is not guaranteed to be monotonic in `left_open` or `right_open`.
The reason is sequence dependence:

- different parts of the sequence may fit the nucleosome geometry better or worse
- opening the left side and opening the right side remove different local motifs
- the stiffness couplings are nonlocal after marginalization

So two states with the same total number of opened contacts can have different
free energies if the opening is distributed differently between left and right.

## How to interpret a comparison correctly

If you compare:

- `(left_open=0, right_open=0)` versus
- `(left_open=2, right_open=0)`

you are comparing two different wrapped states of the same sequence, not the same
full-length constraint problem.

A useful quantity is:

`deltaF_bind = F - F_freedna`

because it asks how costly the nucleosome-like constraints are relative to free
DNA for that same remaining wrapped window.

## Edge cases

- If only one active midstep location remains, the function returns the free-DNA
  term directly because no interval remains to constrain.
- If `left_open + right_open` removes all 28 midstep locations, the current code
  will fail before reaching that fallback branch. In practice, keep at least one
  active location, and for a meaningful constrained calculation keep at least two.

## Minimal usage sketch

```python
from methods.read_nuc_data import read_nucleosome_triads, GenStiffness
from methods.free_energy import calculate_midstep_triads
from binding_model import binding_model_free_energy
import numpy as np

seq = "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT"

genstiff = GenStiffness(method="hybrid")
free_M, free_gs = genstiff.gen_params(seq, use_group=True)

nuctriads = read_nucleosome_triads("methods/State/Nucleosome.state")
midstep_locs = [
    2, 6, 14, 17, 24, 29,
    34, 38, 45, 49, 55, 59,
    65, 69, 76, 80, 86, 90,
    96, 100, 107, 111, 116, 121,
    128, 131, 139, 143
]
nuc_mu0 = calculate_midstep_triads(midstep_locs, nuctriads)
nuc_K = np.load("MDParams/nuc_K_pos_resc_sym.npy")

out = binding_model_free_energy(
    free_gs,
    free_M,
    nuc_mu0,
    nuc_K,
    left_open=0,
    right_open=0,
    use_correction=True,
)

print(out["F"], out["F"] - out["F_freedna"])
```

## Files worth reading together

- `binding_model.py`: implementation of the model
- `methods/midstep_composites.py`: composite-coordinate transformations
- `methods/free_energy.py`: construction of nucleosome midstep triads
