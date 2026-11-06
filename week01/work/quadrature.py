"""Week 1, Task 0: composite quadrature and convergence-analysis tools.

Formulas follow sections 2.4-2.5 of the lecture notes.

Conventions: f is a vectorised callable, [a, b] the interval,
N the number of sub-intervals, h = (b - a) / N.
"""
from __future__ import annotations

import numpy as np


def trapezoid(f, a: float, b: float, N: int) -> float:
    """Composite trapezoidal rule T_h. Expected global error O(h^2).

    T_h = h * [f0/2 + f1 + ... + f_{N-1} + fN/2]
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    return h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])


def simpson(f, a: float, b: float, N: int) -> float:
    """Composite Simpson rule S_h; N must be even. Expected global error O(h^4).

    S_h = h/3 * [f0 + 4 f1 + 2 f2 + 4 f3 + ... + 2 f_{N-2} + 4 f_{N-1} + fN]
    Each parabola spans two sub-intervals, hence N even.
    """
    if N < 2 or N % 2:
        raise ValueError(f"Simpson needs an even N >= 2, got {N}")
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


def observed_order(errors, Ns) -> np.ndarray:
    """Pairwise (two-grid) observed order between neighbouring points:

    p_k = ln(|e_k| / |e_{k+1}|) / ln(N_{k+1} / N_k)

    Returns an array one shorter than the input; entry k belongs to N_{k+1}.
    """
    e = np.abs(np.asarray(errors, float))
    N = np.asarray(Ns, float)
    return np.log(e[:-1] / e[1:]) / np.log(N[1:] / N[:-1])


def observed_order_3grid(Q1: float, Q2: float, Q3: float, r: float = 2.0) -> float:
    """Observed order from three grids refined by ratio r (h, h/r, h/r^2).
    No exact answer needed:

    p = ln[(Q1 - Q2) / (Q2 - Q3)] / ln r
    """
    return np.log((Q1 - Q2) / (Q2 - Q3)) / np.log(r)


def richardson(Q_coarse: float, Q_fine: float, p: float, r: float = 2.0) -> float:
    """Richardson extrapolation. Assumes Q_h = Q* + C h^p, so

    Q* = Q_fine + (Q_fine - Q_coarse) / (r^p - 1)

    |Q* - Q_fine| is then the error estimate of the fine-grid result.
    """
    return Q_fine + (Q_fine - Q_coarse) / (r**p - 1)


def fit_slope(h, errors) -> float:
    """Least-squares slope of log|error| against log h.

    Positive for a convergent method; equals p when |error| = C h^p.
    """
    slope, _ = np.polyfit(np.log(np.asarray(h, float)),
                          np.log(np.abs(np.asarray(errors, float))), 1)
    return slope
