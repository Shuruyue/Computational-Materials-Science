"""Week 3, Task 0: linear-algebra warm-up.

Gaussian elimination without pivoting (lecture notes §0.2), the difference matrix
T = tridiag(-1, 2, -1) with its closed-form eigenvalues and condition number (0.3, 4.3),
and the condition number of the Hilbert matrix (4.3).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import hilbert, invhilbert


def gauss_solve(A, b):
    """Solve A x = b by Gaussian elimination without pivoting, then back substitution.

    Column k is eliminated by subtracting l_i = U[i, k]/U[k, k] times row k from every
    row i below it. Returns (x, U, c): U is the upper-triangular matrix left after
    elimination and c the right-hand side it acts on, U x = c. A zero pivot fails
    (division by zero); the matrices of this week are diagonally dominant, so it never does.
    """
    U = np.array(A, dtype=float)
    c = np.array(b, dtype=float)
    n = len(c)
    for k in range(n - 1):
        l = U[k + 1:, k] / U[k, k]
        U[k + 1:, k + 1:] -= np.outer(l, U[k, k + 1:])
        U[k + 1:, k] = 0.0
        c[k + 1:] -= l * c[k]
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x, U, c


def tridiag_matrix(a, b, c):
    """Dense matrix from three diagonals in the Thomas convention:
    sub-diagonal a[1:], diagonal b, super-diagonal c[:-1]."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    return np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)


def t_matrix(m: int) -> np.ndarray:
    """T = tridiag(-1, 2, -1), m x m."""
    return tridiag_matrix(-np.ones(m), 2 * np.ones(m), -np.ones(m))


def t_eigenvalues(m: int) -> np.ndarray:
    """lambda_p = 2 - 2 cos(p pi/(m+1)) = 4 sin^2(p pi/(2(m+1))), p = 1..m (LeVeque eq. 2.23).
    The sin^2 form avoids the cancellation of 2 - 2 cos for small p."""
    p = np.arange(1, m + 1)
    return 4 * np.sin(p * np.pi / (2 * (m + 1))) ** 2


def t_condition(m: int) -> float:
    """kappa(T) = lambda_m / lambda_1 = cos^2/sin^2 of pi/(2(m+1)) = cot^2(pi/(2(m+1)))."""
    return 1 / np.tan(np.pi / (2 * (m + 1))) ** 2


def hilbert_condition(n: int) -> float:
    """2-norm condition number of the n x n Hilbert matrix, kappa = lambda_max(H) * lambda_max(H^-1).

    H^-1 comes from its exact integer entries (scipy.linalg.invhilbert(exact=True)), so no
    ill-conditioned solve is involved; both largest eigenvalues are well conditioned. This stays
    accurate for n >= 12, where np.linalg.cond(H) has lost every digit (kappa > 1/eps).
    """
    Hinv = np.array(invhilbert(n, exact=True), dtype=float)
    return np.linalg.eigvalsh(hilbert(n))[-1] * np.linalg.eigvalsh(Hinv)[-1]
