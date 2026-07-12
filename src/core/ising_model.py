"""One-dimensional Ising/ligand-binding model using a 2x2 transfer matrix.

This script mirrors the transfer-matrix setup used in the protamine notebook.
It computes the finite-size partition function Z_n for an open chain of length n
with cooperativity J (coupling) and chemical potential mu (field), both in
thermal units beta=1 by default.
"""

from __future__ import annotations

import argparse
import numpy as np

# ---------- Transfer-matrix (finite-n) ----------


def transfer_matrix(beta_mu: float, beta_J: float) -> tuple[np.ndarray, np.ndarray]:
    """Return 2x2 transfer matrix T and boundary vector b."""
    ef2 = np.exp(0.5 * beta_mu)
    T = np.array([[1.0, ef2], [ef2, np.exp(beta_mu + beta_J)]], dtype=float)
    b = np.array([1.0, ef2], dtype=float)
    return T, b


def d_transfer_matrix_dmu(beta_mu: float, beta_J: float) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of T and b with respect to beta*mu."""
    ef2 = np.exp(0.5 * beta_mu)
    dT = np.array([[0.0, 0.5 * ef2], [0.5 * ef2, np.exp(beta_mu + beta_J)]], dtype=float)
    db = np.array([0.0, 0.5 * ef2], dtype=float)
    return dT, db


def Zn_and_dZn_dmu(n: int, beta_mu: float, beta_J: float) -> tuple[float, float]:
    """Partition function Z_n and derivative dZ/d(beta*mu) for open chain of size n."""
    if n < 1:
        return 1.0, 0.0
        raise ValueError("Chain length n must be positive.")

    T, b = transfer_matrix(beta_mu, beta_J)
    dT, db = d_transfer_matrix_dmu(beta_mu, beta_J)

    if n == 1:
        Z = float(b @ b)
        dZ = float((db @ b) + (b @ db))
        return Z, dZ

    # Powers of T: Tp[k] = T^k for k=0..n-1.
    Tp = [np.eye(2)]
    for _ in range(1, n):
        Tp.append(Tp[-1] @ T)

    Z = float(b.T @ (Tp[n - 1] @ b))

    # d(T^{m})/d(beta*mu) = sum_{k=0}^{m-1} T^k (dT/dmu) T^{m-1-k}
    m = n - 1
    dTpower = np.zeros((2, 2))
    for k in range(m):
        dTpower += Tp[k] @ dT @ Tp[m - 1 - k]

    dZ = float(
        (db.T @ (Tp[n - 1] @ b))  # db^T T^{n-1} b
        + (b.T @ (Tp[n - 1] @ db))  # b^T T^{n-1} db
        + (b.T @ (dTpower @ b))  # b^T d(T^{n-1})/dmu b
    )
    return Z, dZ


def partition_function(n: int, beta_mu: float, beta_J: float) -> float:
    """Convenience wrapper: return Z_n for given size and parameters."""
    Z, _ = Zn_and_dZn_dmu(n, beta_mu, beta_J)
    return Z


def mean_occupancy(n: int, beta_mu: float, beta_J: float) -> float:
    """Average occupancy <s> = (1/n) d ln Z / d(beta*mu) for the finite chain."""
    Z, dZ = Zn_and_dZn_dmu(n, beta_mu, beta_J)
    return (dZ / Z) / n


def p_free(n: int, beta_mu: float, beta_J: float) -> float:
    """
    Equilibrium probability that the *boundary* site of an open segment of length n
    is FREE (sigma = 0) in the 1D Ising / ligand-binding model.

    Uses the transfer-matrix representation:
        T, b as in `transfer_matrix`.
        v^{(n)} = T^{n-1} b   (2-component vector)
        Z_n     = b^T v^{(n)}
        P_free(n) = v^{(n)}_0 / Z_n   (component 0 = sigma=0)

    For n <= 0, returns 0.0 (no open sites -> no boundary site).
    """
    if beta_mu == -np.inf:
        # print("  beta_mu = -inf -> no ligands bound.")
        return 1.0  # No ligands bound -> boundary site always free

    if n <= 0:
        return 0.0

    T, b = transfer_matrix(beta_mu, beta_J)

    
    if n == 1:
        v = b.copy()  # T^0 b
    else:
        # build T^(n-1)
        Tp = np.eye(2)
        for _ in range(1, n):
            Tp = Tp @ T       # after loop: Tp = T^(n-1)
        v = Tp @ b            # v = T^(n-1) b

    Z_n = float(b @ v)
    # print(f"Z_n = {Z_n}, v = {v}")
    # v[0] is the component for sigma = 0 (empty)
    return float(v[0] / Z_n)


def p_free_site_dependent(n: int, beta_mu: float, beta_J_bonds) -> float:
    """
    Boundary-site free probability for a chain with site-dependent cooperativity.

    ``beta_J_bonds`` is ordered from the outermost exposed site toward the
    innermost site and must contain ``n - 1`` bond values.  With uniform bond
    values this reduces to ``p_free(n, beta_mu, beta_J)``.
    """
    if beta_mu == -np.inf:
        return 1.0
    if n <= 0:
        return 0.0

    bonds = np.asarray(beta_J_bonds, dtype=float)
    if bonds.size != max(n - 1, 0):
        raise ValueError(f"Expected {n - 1} beta_J bonds for n={n}, got {bonds.size}")

    ef2 = np.exp(0.5 * beta_mu)
    b_outer = np.array([1.0, ef2], dtype=float)
    b_inner = np.array([1.0, ef2], dtype=float)

    v = b_outer.copy()
    for beta_J in bonds:
        T = np.array(
            [[1.0, ef2], [ef2, np.exp(beta_mu + beta_J)]],
            dtype=float,
        )
        v = T @ v

    Z_n = float(b_inner @ v)
    return float(v[0] / Z_n)


# ---------- Cooperativity vs chain-length analysis ----------
#
# Question: on bare DNA, does nearest-neighbor cooperativity actually change the
# quantity that drives eviction kinetics, or is it already saturated at J=0?
# The kinetically relevant observable is p_free(n): the probability the boundary
# (innermost exposed) site is empty, since the effective closing rate is
# k_close_eff = k_close * p_free(n). Cooperativity needs neighbors to act, so its
# effect should grow with the number of exposed sites n and may be negligible on
# the short arms (outer ~3 sites per nucleosome end) that actually gate eviction.


def betamu_from_conc(conc, c0):
    """beta*mu = ln(conc / c0), the grand-canonical mapping (no depletion)."""
    conc = np.asarray(conc, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(conc / c0)


def any_bound(n: int, beta_mu: float, beta_J: float) -> float:
    """Probability the chain carries at least one ligand, g_n = 1 - 1/Z_n."""
    Z, _ = Zn_and_dZn_dmu(n, beta_mu, beta_J)
    return 1.0 - 1.0 / Z


def site_sweep(n_values, beta_mu: float, beta_J: float) -> dict:
    """p_free, mean occupancy, and any-bound over a range of chain lengths."""
    n_values = np.asarray(n_values, dtype=int)
    pf = np.array([p_free(int(n), beta_mu, beta_J) for n in n_values])
    occ = np.array([mean_occupancy(int(n), beta_mu, beta_J) for n in n_values])
    ab = np.array([any_bound(int(n), beta_mu, beta_J) for n in n_values])
    return {"n": n_values, "p_free": pf, "occupancy": occ, "any_bound": ab}


def cooperativity_summary(c0=89.7, coop=4.5, n_list=(2, 3, 6, 14, 30), concs=None):
    """Print the peak cooperativity effect on p_free per chain length.

    For each n, scans concentration and reports where |p_free(J=0) - p_free(J)|
    is largest. A small peak means cooperativity barely changes the rewrapping
    suppression at that arm length.
    """
    if concs is None:
        concs = np.logspace(-2, 3, 200)
    concs = np.asarray(concs, float)
    bmu = betamu_from_conc(concs, c0)
    print(f"\nPeak cooperativity effect on p_free (J=0 vs J={coop:g}), c0={c0:g} uM")
    print(f"{'n':>4} {'max|d p_free|':>14} {'at c (uM)':>10} "
          f"{'p_free(J0)':>11} {'p_free(J)':>10}")
    for n in n_list:
        pf0 = np.array([p_free(int(n), float(m), 0.0) for m in bmu])
        pfJ = np.array([p_free(int(n), float(m), coop) for m in bmu])
        d = pf0 - pfJ
        i = int(np.argmax(np.abs(d)))
        print(f"{n:>4} {d[i]:>14.4f} {concs[i]:>10.3g} "
              f"{pf0[i]:>11.4f} {pfJ[i]:>10.4f}")


def plot_cooperativity_vs_sites(c0=89.7, coop=4.5, n_max=30,
                                concs=(1.0, 10.0, 100.0), outer_sites=3,
                                outdir=None, show=False):
    """Four-panel figure: how cooperativity depends on chain length n."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    n_values = np.arange(1, n_max + 1)
    conc_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(concs)))
    c_mid = concs[len(concs) // 2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_pf, ax_dpf, ax_occ, ax_ab = axes.ravel()

    for c, color in zip(concs, conc_colors):
        bmu = float(betamu_from_conc(c, c0))
        s0 = site_sweep(n_values, bmu, 0.0)
        sJ = site_sweep(n_values, bmu, coop)
        ax_pf.plot(n_values, s0["p_free"], ls="--", color=color, lw=1.8)
        ax_pf.plot(n_values, sJ["p_free"], ls="-", color=color, lw=2.2,
                   label=f"c = {c:g} uM")
        ax_dpf.plot(n_values, s0["p_free"] - sJ["p_free"], "-o", color=color,
                    ms=3, label=f"c = {c:g} uM")

    bmu = float(betamu_from_conc(c_mid, c0))
    s0 = site_sweep(n_values, bmu, 0.0)
    sJ = site_sweep(n_values, bmu, coop)
    ax_occ.plot(n_values, s0["occupancy"], "--", color="k", lw=1.8, label="J = 0")
    ax_occ.plot(n_values, sJ["occupancy"], "-", color="C3", lw=2.2, label=f"J = {coop:g}")
    ax_ab.plot(n_values, s0["any_bound"], "--", color="k", lw=1.8, label="J = 0")
    ax_ab.plot(n_values, sJ["any_bound"], "-", color="C3", lw=2.2, label=f"J = {coop:g}")

    for ax in (ax_pf, ax_dpf, ax_occ, ax_ab):
        ax.axvline(outer_sites, color="gray", ls=":", lw=1.2)
        ax.set_xlabel("number of exposed sites n")

    ax_pf.set_ylabel("p_free (boundary site empty)")
    ax_pf.set_title("Boundary-free probability\n(solid J>0, dashed J=0)")
    ax_pf.legend(fontsize=9, title="color = conc")
    ax_dpf.set_ylabel("p_free(J=0) - p_free(J>0)")
    ax_dpf.set_title(f"Cooperativity effect on p_free  (J = {coop:g})")
    ax_dpf.axhline(0, color="k", lw=0.8)
    ax_dpf.legend(fontsize=9)
    ax_occ.set_ylabel("mean occupancy per site")
    ax_occ.set_title(f"Occupancy  (c = {c_mid:g} uM)")
    ax_occ.legend(fontsize=9)
    ax_ab.set_ylabel("any-bound probability g_n")
    ax_ab.set_title(f"Any-bound  (c = {c_mid:g} uM)")
    ax_ab.legend(fontsize=9)

    fig.suptitle(
        f"Bare-DNA protamine cooperativity vs chain length  (c0 = {c0:g} uM)\n"
        f"vertical dotted line = {outer_sites} outer sites per nucleosome end",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fpath = Path(outdir) / "bare_dna_cooperativity_vs_sites.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print(f"saved {fpath}")
    if show:
        plt.show()
    return fig


def plot_pfree_vs_conc(c0=89.7, coop=4.5, n_fixed=(3, 6, 14, 30),
                       concs=None, outdir=None, show=False):
    """p_free and the cooperativity effect vs concentration, at fixed arm lengths.

    This is the decision figure: for the short arms that gate eviction (n=3, 6),
    is there *any* concentration where cooperativity moves p_free?
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    if concs is None:
        concs = np.logspace(-2, 3, 60)
    concs = np.asarray(concs, float)
    bmu = betamu_from_conc(concs, c0)

    fig, (ax_pf, ax_d) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(n_fixed)))
    for n, color in zip(n_fixed, colors):
        pf0 = np.array([p_free(int(n), float(m), 0.0) for m in bmu])
        pfJ = np.array([p_free(int(n), float(m), coop) for m in bmu])
        ax_pf.plot(concs, pf0, ls="--", color=color, lw=1.8)
        ax_pf.plot(concs, pfJ, ls="-", color=color, lw=2.2, label=f"n = {n}")
        ax_d.plot(concs, pf0 - pfJ, "-", color=color, lw=2.2, label=f"n = {n}")

    for ax in (ax_pf, ax_d):
        ax.set_xscale("log")
        ax.set_xlabel("protamine concentration c (uM)")
        ax.axvline(c0, color="gray", ls=":", lw=1.0)
    ax_pf.set_ylabel("p_free (boundary site empty)")
    ax_pf.set_title("Boundary-free probability\n(solid J>0, dashed J=0)")
    ax_pf.legend(fontsize=9, title="arm length")
    ax_d.set_ylabel("p_free(J=0) - p_free(J>0)")
    ax_d.set_title(f"Cooperativity effect on p_free  (J = {coop:g})")
    ax_d.axhline(0, color="k", lw=0.8)
    ax_d.legend(fontsize=9)

    fig.suptitle(
        f"Does cooperativity change rewrapping suppression at short arms?  "
        f"(c0 = {c0:g} uM)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fpath = Path(outdir) / "bare_dna_pfree_vs_conc.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print(f"saved {fpath}")
    if show:
        plt.show()
    return fig


def run_site_analysis(c0=89.7, coop=4.5, n_max=30, outdir="bare_dna_cooperativity",
                      show=False):
    """Run the full bare-DNA cooperativity-vs-sites analysis and save figures."""
    cooperativity_summary(c0=c0, coop=coop)
    plot_cooperativity_vs_sites(c0=c0, coop=coop, n_max=n_max, outdir=outdir, show=show)
    plot_pfree_vs_conc(c0=c0, coop=coop, outdir=outdir, show=show)


def main(n, beta_mu, beta_J) -> None:


    Z, dZ = Zn_and_dZn_dmu(n, beta_mu, beta_J)
    occupancy = (dZ / Z) / n

    print(f"n = {n}, beta*mu = {beta_mu:.6g}, beta*J = {beta_J:.6g}")
    print(f"Partition function Z_n       : {Z:.6g}")
    print(f"dZ/d(beta*mu)                : {dZ:.6g}")
    print(f"Mean occupancy <s> per site  : {occupancy:.6g}")

    # optional quick check for P_free:
    p0 = p_free(n, beta_mu, beta_J)
    print(f"P_free(boundary site)        : {p0:.6g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sites", "print"], default="sites",
                        help="'sites': cooperativity-vs-chain-length analysis; "
                             "'print': per-n Z_n table (original behavior).")
    parser.add_argument("--c0", type=float, default=89.7,
                        help="k_unbind/k_bind, sets betamu = ln(c/c0). Default 89.7 uM.")
    parser.add_argument("--coop", type=float, default=4.5, help="beta*J cooperativity.")
    parser.add_argument("--nmax", type=int, default=30, help="max chain length.")
    parser.add_argument("--outdir", default="bare_dna_cooperativity",
                        help="where to save figures.")
    parser.add_argument("--no-show", action="store_true", help="save only, do not display.")
    parser.add_argument("--p-conc", type=float, default=10.0,
                        help="protamine concentration for --mode print.")
    args = parser.parse_args()

    if args.mode == "sites":
        run_site_analysis(c0=args.c0, coop=args.coop, n_max=args.nmax,
                          outdir=args.outdir, show=not args.no_show)
    else:
        betamu = np.log(args.p_conc / args.c0)
        for n in range(1, args.nmax + 1):
            main(n=n, beta_mu=betamu, beta_J=args.coop)

