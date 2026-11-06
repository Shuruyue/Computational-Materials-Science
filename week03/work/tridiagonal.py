"""Week 3, Task 1 (a)-(c): the Thomas algorithm and the 1D Poisson problem.

Lecture notes §4.1 (where tridiagonal systems come from), §4.2 (Thomas), §4.3 (error vs round-off).
"""
from __future__ import annotations

import numpy as np


def thomas(a, b, c, d) -> np.ndarray:
    """Solve the tridiagonal system a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i, i = 1..n.

    a: sub-diagonal (a[0] unused), b: diagonal, c: super-diagonal (c[-1] unused), d: right side.
    Forward elimination (lecture notes §4.2):
        c'_i = c_i / (b_i - a_i c'_{i-1}),   d'_i = (d_i - a_i d'_{i-1}) / (b_i - a_i c'_{i-1})
    back substitution:
        x_n = d'_n,   x_i = d'_i - c'_i x_{i+1}.
    About 8n operations. No pivoting, so the matrix should be diagonally dominant.
    The loop runs on Python floats: indexing NumPy scalars one at a time is several times slower.
    """
    a, b, c, d = (np.asarray(v, float).tolist() for v in (a, b, c, d))
    n = len(d)
    cp = [0.0] * n
    dp = [0.0] * n
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.array(x)


def poisson_1d(k: int):
    """-u'' = pi^2 sin(pi x), u(0) = u(1) = 0, exact u = sin(pi x).

    m = 2^k - 1 interior points, h = 2^-k, solved as tridiag(-1, 2, -1) U = h^2 f with thomas.
    m is odd, so x = 1/2 is a grid point and max_j sin(pi x_j) = 1 exactly.
    Returns (x, U, h).
    """
    m = 2**k - 1
    h = 2.0**-k
    x = h * np.arange(1, m + 1)
    ones = np.ones(m)
    U = thomas(-ones, 2 * ones, -ones, h**2 * np.pi**2 * np.sin(np.pi * x))
    return x, U, h


def poisson_discrete_error(h):
    """Exact max error of the discrete Poisson solution (lecture notes §4.1).

    sin(pi x_j) is an eigenvector of T, so U_j = A sin(pi x_j) with A = z^2/sin^2 z, z = pi h/2,
    and max_j |U_j - u(x_j)| = A - 1 = (z - sin z)(z + sin z)/sin^2 z  (~ pi^2 h^2/12).
    z - sin z is summed as its Taylor series, so the result keeps full relative precision
    even when it is 1e-13 (the form A - 1 would cancel almost every digit).
    """
    z = np.pi * np.asarray(h, float) / 2
    z_minus_sin = np.zeros_like(z)
    for n in range(13, 0, -1):                     # z^3/3! - z^5/5! + ..., smallest terms first
        z_minus_sin = z_minus_sin + (-1) ** (n + 1) * z ** (2 * n + 1) / np.prod(np.arange(1.0, 2 * n + 2))
    return z_minus_sin * (z + np.sin(z)) / np.sin(z) ** 2
