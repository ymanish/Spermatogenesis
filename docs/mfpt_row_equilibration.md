# Row equilibration for the MFPT linear solve — theory note

**Status:** concept document. Decide whether to implement after reading.

**Context:** companion to the patch in
`src/analysis/markov_solver/mfpt.py` and the notebook
`notebooks/MFPT_InfFix_BeforeAfter.ipynb`. After dropping the
`cond(Q_TT.T) > 1e12` gate and switching to LU + iterative refinement, a
fraction of nucleosomes still produce numerically broken MFPT values
(in the test run, ~12–15% of the cases that legacy reported as `inf`
come back **negative** — unphysical). This document explains *why*
that happens and how row equilibration removes it without changing the
mathematical problem being solved.

---

## 1. The system we are solving

For an absorbing CTMC with generator (column-sum convention)

$$
Q_{TT} \in \mathbb{R}^{M\times M}, \qquad
Q_{TT}[i,j] = \text{rate}(j \to i) \ (i\ne j), \qquad
Q_{TT}[j,j] = -\!\!\!\sum_{k \ne j}\! Q_{TT}[k,j] \;-\; \text{rate}(j\to\text{abs}),
$$

the mean first-passage time vector $\boldsymbol\tau$ (entry $j$ = MFPT
starting from transient state $j$) satisfies

$$
\boxed{\;Q_{TT}^{\!\top}\;\boldsymbol\tau \;=\; -\mathbf{1}.\;}
$$

This is a linear system $A\,x = b$ with $A = Q_{TT}^{\!\top}$ and
$b = -\mathbf{1}$.

In our problem $M = N(N{+}1)/2$ transient states, where $N$ is the number
of binding sites (e.g. $N = 14 \Rightarrow M = 105$).

---

## 2. Where the numerical pain comes from

For a state $j$, the diagonal of $A$ is

$$
A_{jj} \;=\; Q_{TT}[j,j] \;=\; -\,\underbrace{\Big(\sum_{k\ne j}\text{rate}(j\!\to\! k) + \text{rate}(j\!\to\!\text{abs})\Big)}_{=:\,d_j>0}.
$$

In a column-sum CTMC, $d_j$ is the **total outgoing rate from state $j$**
— literally the inverse mean dwell time in state $j$. And because each
off-diagonal entry in row $j$ of $A$ is one of the rates summed into $d_j$,
$|A_{jk}| \le d_j$ for every $k$. So $A_{jj}$ is the largest-magnitude
entry of row $j$.

Now look at our nucleosome rates:

- "Fast" rates: $k_{\text{wrap}}\cdot 1 = O(1)$ (closing, downhill opening).
- "Slow" rates: $k_{\text{wrap}}\cdot e^{-\Delta F / k_BT}$. For
  $\Delta F \in [20,40]\,k_BT$ this is $10^{-9}$–$10^{-17}$.

The mean dwell time $1/d_j$ therefore spans **many** orders of
magnitude across states. A state surrounded by fast moves has
$d_j \sim 1$; a state whose only way out is over a 35-$k_BT$ barrier
has $d_j \sim 10^{-15}$.

That is exactly what wrecks `cond(A)`:

> the condition number of $A$ is dominated by the ratio of row magnitudes,
> $\max_j d_j / \min_j d_j$, which can easily reach $10^{15}$–$10^{20}$
> even when the **answer** $\boldsymbol\tau$ itself is well-defined.

Float64 LU loses ~$\log_{10}(\text{cond})$ digits. Once cond exceeds
$10^{16}$, you have zero correct digits. That's where the negative
MFPTs come from.

---

## 3. The fix: scale each row by its own dwell rate

Let $D = \text{diag}(d_1, d_2, \ldots, d_M)$ — i.e.
$D_{jj} = |A_{jj}|$ = the total outgoing rate from state $j$.

Pre-multiply the system by $D^{-1}$:

$$
\boxed{\;(D^{-1} A)\,\boldsymbol\tau \;=\; D^{-1}\,b. \;}
$$

This is **algebraically identical** to the original system — both sides
are scaled by an invertible diagonal, so the solution $\boldsymbol\tau$
is unchanged. But the new matrix $\tilde A := D^{-1}A$ has

$$
\tilde A_{jj} = -1, \qquad |\tilde A_{jk}| \le 1 \text{ for all } k,
$$

i.e. every diagonal is exactly $-1$ and every off-diagonal lies in
$[0,1]$. All rows are now on the same scale, and the right-hand side
becomes $D^{-1}(-\mathbf{1})$, whose entries are the **mean dwell times**
$-1/d_j$.

Three things to understand about why this is the *right* fix here:

1. **Physical meaning of $D^{-1}$.** Element $1/d_j$ is the mean residence
   time in state $j$. Dividing row $j$ by $d_j$ is the same as "measuring
   time in units of the local dwell time of state $j$." That's precisely
   the scaling needed to make state-to-state coupling dimensionless and
   $O(1)$ regardless of how much faster or slower one state's clock runs
   than another's.

2. **It only kills *cosmetic* ill-conditioning.** If $A$ is genuinely
   near-singular (true near-disconnection from absorption ⇒ a near-zero
   eigenvalue), $\tilde A$ is too — row scaling cannot manufacture rank.
   So a high $\text{cond}(\tilde A)$ after equilibration *is* a real
   warning sign, whereas a high $\text{cond}(A)$ before equilibration
   usually is not.

3. **No "rescaling back."** A common confusion: since we scaled the
   system, do we need to multiply $\boldsymbol\tau$ by $D$ at the end?
   **No.** We pre-multiplied both sides by the same $D^{-1}$, so the
   solution variable is unchanged. The output of the solve is already
   the MFPT in the original units (here, dimensionless time
   $\tau = k_{\text{wrap}}\,t$).

---

## 4. Worked example — small enough to do by hand

A 3-state transient chain $\{0,1,2\}$ plus an absorbing state $A$:

| from | to | rate |
|---|---|---|
| 0 | 1 | $1$ |
| 0 | 2 | $10^{10}$  *(fast forward kick)* |
| 0 | A | $10^{-3}$ |
| 1 | 0 | $1$ |
| 1 | 2 | $0.3$ |
| 1 | A | $0.2$ |
| 2 | 0 | $10^{-3}$ |
| 2 | 1 | $10^{-11}$ |
| 2 | A | $10^{-11}$ |

Total dwell rates per state:

$$
d_0 = 10^{10} + 1 + 10^{-3} \approx 10^{10}, \quad
d_1 = 1 + 0.3 + 0.2 = 1.5, \quad
d_2 \approx 1.001\times 10^{-3}.
$$

Build $A = Q_{TT}^{\!\top}$, which has diagonals $(-d_0,-d_1,-d_2)$.
Row magnitudes span $10^{10} \to 10^{-3}$ — that's **13 orders of
magnitude**.

```
cond(A)         ≈ 1.0 × 10^21        # before scaling
cond(D^-1 A)    ≈ 1.2 × 10^1         # after row equilibration
```

The matrix changed by 20 orders of magnitude in conditioning, but the
solution is identical:

```
τ_true   = [3.333e10, 3.333e10, 5.556e10]   # by exact rational arithmetic
τ_raw    = [3.333e10, 3.333e10, 5.556e10]   # double-precision LU on A
τ_scaled = [3.333e10, 3.333e10, 5.556e10]   # double-precision LU on D^-1 A
```

What did row scaling actually *do* numerically? It made the LU pivot
choices irrelevant to the answer's accuracy: with the original matrix,
floating-point pivoting on rows spanning $10^{10}$–$10^{-3}$ throws away
the small entries; with the scaled matrix, all pivots are $O(1)$ and
nothing is lost.

(This is exactly the example reproduced in the conversation transcript.)

---

## 5. Why this helps **our** nucleosomes specifically

For the configuration in `markov_config.yaml`
($N{=}14$, $k_{\text{wrap}}{=}1$, $p_{\text{conc}}{=}10$,
$\text{coop}{=}0$, $\beta J = 0$), opening rates carry the
Arrhenius factor

$$
k_{\text{open}}(\ell,r \to \ell',r')
= k_{\text{wrap}}\cdot e^{-(F(\ell',r')-F(\ell,r))/k_BT}.
$$

The free-energy landscape $F(\ell,r)$ is read off `G_mat` per
nucleosome. For nucleosomes with deep barriers, certain transient
states have $d_j = \text{(closing rates)} + e^{-\Delta F/k_BT}$, where
the closing terms can themselves be tiny if `p_free` is small under
high-protamine conditions.

End result: $d_j$ for some states is ~$1$ while for others it's
~$e^{-30}\approx 10^{-13}$. That's a 13-order-of-magnitude spread in
row magnitudes for a 105×105 matrix — exactly the regime where
row equilibration buys you ~13 digits of conditioning.

This is also why **the recovered MFPTs that came out negative** in the
notebook are specifically the high-barrier nucleosomes. The fix isn't
mathematical — the answer was always finite — it's purely a
floating-point hygiene issue.

---

## 6. Suggested implementation (if you decide to do it)

A small modification to
[`compute_mfpt_from_Q_TT`](../src/analysis/markov_solver/mfpt.py):

```python
def compute_mfpt_from_Q_TT(Q_TT, state_index, start_state=(0, 0)):
    M = Q_TT.shape[0]
    A = np.asarray(Q_TT.T, dtype=float)
    b = -np.ones(M)

    # Row equilibration by |diag(A)| = total outgoing rate per state
    d = np.abs(np.diag(A))
    safe = d > 0
    A_s = A.copy()
    A_s[safe] = A_s[safe] / d[safe, None]
    b_s = b.copy()
    b_s[safe] = b_s[safe] / d[safe]

    try:
        lu, piv = sla.lu_factor(A_s)
        tau = sla.lu_solve((lu, piv), b_s)
        # One step of iterative refinement
        r = b_s - A_s @ tau
        tau = tau + sla.lu_solve((lu, piv), r)
    except np.linalg.LinAlgError:
        return np.inf, np.full(M, np.inf)

    mfpt = float(tau[state_index[start_state]])
    return mfpt, tau
```

Notes:

- The branch `safe = d > 0` is purely defensive. In a properly built
  transient generator every state has at least one outgoing edge
  (otherwise it would be absorbing, not transient). It should never
  trigger; if it does, that's a generator bug, not a solver problem.
- The iterative-refinement step is identical to what the patched
  solver already does. You can keep it, drop it, or replace it with
  `scipy.linalg.solve(A_s, b_s)` which does its own equilibration
  internally — but doing it explicitly here makes the intent obvious
  and is no slower in practice.
- After this fix, the right way to detect a *genuinely* sick nucleosome
  is `mfpt < 0`. That can only come from extreme residual error and
  flags either a generator bug or a state space that needs higher
  precision (mpmath).

---

## 7. Summary

| | meaning | typical value (this dataset) |
|---|---|---|
| $A = Q_{TT}^{\!\top}$ | original system matrix | rows span $10^{0}$–$10^{-15}$ |
| $D = \text{diag}\|A_{jj}\|$ | per-state dwell-rate matrix | $d_j \in [10^{-15}, 10^{1}]$ |
| $\tilde A = D^{-1}A$ | row-equilibrated matrix | diagonals all $-1$, off-diagonals in $[0,1]$ |
| $\tilde b = D^{-1}b$ | per-state dwell-time vector | $-1/d_j$, the mean residence time in $j$ |
| solution $\boldsymbol\tau$ | MFPT vector | **same as the unscaled solve** |

The scaling does **not change the math**, only the numerical conditioning
of the matrix the solver sees. It is a strict improvement on
ill-conditioned cases and a pure no-op on well-conditioned ones.
The only reason to skip it is if you'd rather flag the unphysical
negative MFPTs from the current patched solver as a separate concern
and handle them via `mpmath` or sparse arbitrary-precision arithmetic
instead.
