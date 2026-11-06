"""Week 2, Task 4: bulk modulus of Cu from a Morse pair potential (Girifalco & Weizer 1959).

phi(r) = D [e^{-2 alpha (r - r0)} - 2 e^{-alpha (r - r0)}]
E(a)   = 1/2 sum_j phi(a |s_j|)       energy per atom; s_j are FCC sites in units of a
B      = 4 E''(a0) / (9 a0)            lecture notes §3.8

The neighbour list s_j is built once and kept fixed while a changes (lecture notes §3.8, last paragraph):
then E(a) is a smooth function of a and the only error in a finite difference is truncation.
"""
from __future__ import annotations

from math import factorial

import numpy as np
from scipy.optimize import brentq

D_MORSE = 0.3429             # eV
ALPHA = 1.3588               # 1/Å
R0 = 2.866                   # Å
EV_PER_A3_IN_GPA = 160.21766

FCC_BASIS = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def phi(r, n: int = 0):
    """n-th derivative of the Morse potential:
    phi^(n)(r) = D [(-2 alpha)^n e^{-2 alpha (r - r0)} - 2 (-alpha)^n e^{-alpha (r - r0)}].
    Accepts complex r."""
    y = np.exp(-ALPHA * (r - R0))
    return D_MORSE * ((-2 * ALPHA) ** n * y**2 - 2 * (-ALPHA) ** n * y)


def fcc_neighbours(s_max: float) -> np.ndarray:
    """|s_j| of every FCC site with 0 < |s_j| <= s_max (in units of a), one entry per site, sorted.

    Sites are basis + integer translations. Their coordinates are multiples of 1/2, so |s|^2 is
    exact in floating point and the cutoff test needs no tolerance.
    """
    n = int(np.ceil(s_max)) + 1
    t = np.arange(-n, n + 1)
    cells = np.stack(np.meshgrid(t, t, t, indexing='ij'), axis=-1).reshape(-1, 3)
    sites = (cells[:, None, :] + FCC_BASIS[None, :, :]).reshape(-1, 3)
    s2 = (sites**2).sum(axis=1)
    return np.sqrt(np.sort(s2[(s2 > 0) & (s2 <= s_max**2)]))


def fcc_shells(s):
    """Group a neighbour list into shells. FCC has |s|^2 = k/2 for integer k, so shells are exact.
    Returns (shell radii |s|, number of sites in each shell)."""
    k, counts = np.unique(np.rint(2 * s**2).astype(int), return_counts=True)
    return np.sqrt(k / 2), counts


def energy(a, s, n: int = 0):
    """n-th derivative of E(a) = 1/2 sum_j phi(a s_j):  E^(n)(a) = 1/2 sum_j s_j^n phi^(n)(a s_j)
    (eV/Å^n per atom). a may be a scalar or an array of any shape; s is the fixed neighbour list."""
    a = np.asarray(a)
    return 0.5 * (s**n * phi(a[..., None] * s, n)).sum(axis=-1)


def equilibrium(s, bracket=(3.4, 3.8)) -> float:
    """a0 from the analytic E'(a0) = 0 (brentq, xtol = 1e-12 Å)."""
    return brentq(lambda a: energy(a, s, 1), *bracket, xtol=1e-12)


def bulk_modulus(E2, a0):
    """B = 4 E''(a0) / (9 a0) in GPa (FCC: V = a^3/4 per atom, E'(a0) = 0)."""
    return 4 * E2 / (9 * a0) * EV_PER_A3_IN_GPA


def _g(y):
    """g(y) = 2(cosh y - 1) - y^2 = 2 sum_{k>=2} y^(2k)/(2k)!. The series is used for y < 1, where
    the closed form would cancel; for y >= 1 the closed form loses at most about one digit."""
    y = np.asarray(y, float)
    out = 2 * (np.cosh(y) - 1) - y**2
    small = y < 1
    series = np.zeros(small.sum())
    for k in range(12, 1, -1):                     # smallest terms first
        series += 2 * y[small] ** (2 * k) / factorial(2 * k)
    out[small] = series
    return out


def second_difference_error(a, h, s):
    """Exact truncation error D^2 E(a) - E''(a) of the 3-point second difference, without forming
    D^2 E at all.

    Each Morse term is an exponential e^{-c a} (c = 2 alpha s or alpha s). For it D^2 gives
    e^{-c a} * 2(cosh ch - 1)/h^2 instead of c^2 e^{-c a}, so the error of that term is
    e^{-c a} g(ch)/h^2 with g(y) = 2(cosh y - 1) - y^2 = y^4/12 + y^6/360 + ...
    """
    h = float(h)
    y = np.exp(-ALPHA * (a * s - R0))
    return 0.5 * D_MORSE * np.sum(y**2 * _g(2 * ALPHA * s * h) - 2 * y * _g(ALPHA * s * h)) / h**2


def noisy(E, delta: float, rng):
    """Wrap E(a) so that every call adds independent noise delta * U(-1, 1) to each value,
    a stand-in for DFT energies that are converged only to about delta."""
    def E_noisy(a):
        a = np.asarray(a, float)
        return E(a) + delta * rng.uniform(-1.0, 1.0, a.shape)
    return E_noisy


def energy_rescreened(a, s_all, rc: float = 10.0):
    """E(a) with the neighbour list rebuilt for every a: only sites with a |s_j| <= rc count.
    Whenever a shell crosses rc, E(a) jumps by 1/2 (sites in shell) phi(rc). s_all must reach
    beyond rc / a."""
    a = np.asarray(a, float)
    r = a[..., None] * s_all
    return 0.5 * np.where(r <= rc, phi(r), 0.0).sum(axis=-1)
