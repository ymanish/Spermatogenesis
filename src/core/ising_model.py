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
    p_conc = 10.0  # protamine concentration in uM
    k_bind = 1.0   # protamine binding rate in 1/(uM s)
    k_unbind = 89.7 # protamine unbinding rate in 1/s
    cooperativity = 0.0  # unitless cooperativity parameter
    betamu = np.log(p_conc * k_bind / k_unbind)
    main(n=14, beta_mu=betamu, beta_J=cooperativity)
