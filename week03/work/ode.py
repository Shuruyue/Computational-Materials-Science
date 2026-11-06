"""Week 3, Task 3: time stepping, absolute stability and stiffness on the scalar test problem.

y' = lam (y - cos t) - sin t, y(0) = 1, exact solution y = cos t for every lam
(LeVeque eqs. 7.1 and 8.1; lecture notes §4.6-4.8).

The integrators only use + - * / on the numbers they are given, so the same code runs on
floats and on decimal.Decimal (with dcos/dsin below). The Decimal run is the same recurrence
in (nearly) exact arithmetic: it separates the method's own error from round-off.
"""
from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np


def forward_euler(f, y0, h, N, t0=0):
    """Y_{n+1} = Y_n + h f(t_n, Y_n), t_n = t0 + n h. Returns [Y_0, ..., Y_N]."""
    Y = [y0]
    for n in range(N):
        Y.append(Y[-1] + h * f(t0 + n * h, Y[-1]))
    return Y


def rk4(f, y0, h, N, t0=0):
    """Classical RK4 (lecture notes §4.6): k1 = f(t, Y), k2 = f(t + h/2, Y + h k1/2),
    k3 = f(t + h/2, Y + h k2/2), k4 = f(t + h, Y + h k3), Y_{n+1} = Y_n + h (k1 + 2k2 + 2k3 + k4)/6."""
    Y = [y0]
    for n in range(N):
        t, y = t0 + n * h, Y[-1]
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        Y.append(y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return Y


def backward_euler_linear(lam, g, y0, h, N, t0=0):
    """Backward Euler Y_{n+1} = Y_n + h f(t_{n+1}, Y_{n+1}) for the linear f = lam y + g(t),
    solved for Y_{n+1} directly: Y_{n+1} = (Y_n + h g(t_{n+1})) / (1 - h lam)."""
    Y = [y0]
    for n in range(N):
        Y.append((Y[-1] + h * g(t0 + (n + 1) * h)) / (1 - h * lam))
    return Y


def trapezoid_linear(lam, g, y0, h, N, t0=0):
    """Trapezoid Y_{n+1} = Y_n + (h/2)[f(t_n, Y_n) + f(t_{n+1}, Y_{n+1})] for f = lam y + g(t):
    Y_{n+1} = [(1 + h lam/2) Y_n + (h/2)(g(t_n) + g(t_{n+1}))] / (1 - h lam/2)."""
    Y = [y0]
    for n in range(N):
        t = t0 + n * h
        Y.append(((1 + h * lam / 2) * Y[-1] + h / 2 * (g(t) + g(t + h))) / (1 - h * lam / 2))
    return Y


def cos_problem(lam, cos=math.cos, sin=math.sin):
    """f(t, y) = lam (y - cos t) - sin t and g(t) = -lam cos t - sin t, so that f = lam y + g.
    Pass dcos/dsin (and Decimal lam) for the high-precision rerun."""
    def f(t, y):
        return lam * (y - cos(t)) - sin(t)

    def g(t):
        return -lam * cos(t) - sin(t)
    return f, g


# ---------------------------------------------------------------------------------------
# Stability functions: applied to y' = lam y every method gives Y_{n+1} = R(z) Y_n, z = h lam.

def R_fe(z):
    return 1 + z


def R_rk4(z):
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def R_be(z):
    return 1 / (1 - z)


def R_tr(z):
    return (1 + z / 2) / (1 - z / 2)


def rk4_real_limit() -> float:
    """Left end of RK4's stability interval on the real axis. R(z) > 0 there, so the boundary is
    R(z) = 1: z (1 + z/2 + z^2/6 + z^3/24) = 0, i.e. the real root of z^3 + 4z^2 + 12z + 24 = 0."""
    roots = np.roots([1, 4, 12, 24])
    return float(roots[np.argmin(abs(roots.imag))].real)


# ---------------------------------------------------------------------------------------
# High precision: cos and sin for decimal.Decimal by Taylor series (|x| <~ 4 here).

def _series(x, total, term, k):
    x2 = x * x
    while True:
        term = -term * x2 / ((k + 1) * (k + 2))
        k += 2
        if total + term == total:
            return total
        total += term


def dcos(x: Decimal) -> Decimal:
    return _series(x, Decimal(1), Decimal(1), 0)


def dsin(x: Decimal) -> Decimal:
    return _series(x, x, x, 1)


def set_precision(digits: int = 40) -> None:
    """Working precision of every Decimal operation that follows."""
    getcontext().prec = digits
