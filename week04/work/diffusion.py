"""Week 4, Task 0 and the exact solutions: diffusion into a semi-infinite solid (lecture notes §0.3, §5.3, §5.8, §5.9).

Problem: c_t = D c_xx for x > 0, c(x, 0) = C0, c(0, t) = Cs, c(inf, t) = C0.
Exact solution (c - C0)/(Cs - C0) = erfc(x / (2 sqrt(D t))).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc, erfcinv, eval_hermite

WEEK01 = Path(__file__).resolve().parent.parent.parent / 'week01' / 'work'
if str(WEEK01) not in sys.path:
    sys.path.append(str(WEEK01))
from quadrature import simpson  # noqa: E402

SQRT_PI = np.sqrt(np.pi)


def erf_simpson(z: float, N: int = 1000) -> float:
    """erf(z) = (2/sqrt(pi)) int_0^z exp(-u^2) du with the composite Simpson rule of week 1 (N sub-intervals)."""
    return 2 / SQRT_PI * simpson(lambda u: np.exp(-u * u), 0.0, z, N)


def erfc_solution(x, t, D: float = 1.0, cs: float = 1.0, c0: float = 0.0):
    """c(x, t) = C0 + (Cs - C0) erfc(x / (2 sqrt(D t))). Accepts complex x or t (for complex-step derivatives)."""
    x = np.asarray(x)
    return c0 + (cs - c0) * erfc(x / (2 * np.sqrt(D * t)))


def ct_exact(x, t, D: float = 1.0):
    """dc/dt of the erfc solution with Cs - C0 = 1: x exp(-x^2/(4Dt)) / (2 sqrt(pi D) t^(3/2)).

    It equals D c_xx, the curvature term, which is the same expression.
    """
    x = np.asarray(x, float)
    return x * np.exp(-x * x / (4 * D * t)) / (2 * SQRT_PI * np.sqrt(D) * t ** 1.5)


def erfc_x_derivative(x, t, k: int, D: float = 1.0):
    """k-th x-derivative (k >= 1) of c = erfc(x / (2 sqrt(D t))), from the Hermite polynomials:

    d^n/dx^n exp(-x^2/(4Dt)) = (-1)^n (4Dt)^(-n/2) H_n(u) exp(-u^2),  u = x/(2 sqrt(D t)),
    and c_x = -exp(-u^2)/sqrt(pi D t), so the k-th derivative uses H_{k-1}.
    Since c_t = D c_xx, time derivatives follow as d^m c/dt^m = D^m (2m-th x-derivative).
    """
    x = np.asarray(x, float)
    u = x / (2 * np.sqrt(D * t))
    n = k - 1
    return -(-1) ** n * (4 * D * t) ** (-n / 2) * eval_hermite(n, u) * np.exp(-u * u) / np.sqrt(np.pi * D * t)


def pde_residual(x: float, t: float, ht: float = 1e-4, hx: float = 1e-3, D: float = 1.0):
    """Central differences of the erfc solution: (c_t, c_xx, c_t - D c_xx).

    c_t  = [c(x, t + ht) - c(x, t - ht)] / (2 ht)
    c_xx = [c(x + hx, t) - 2 c(x, t) + c(x - hx, t)] / hx^2
    Both carry the week-2 truncation errors (ht^2/6) c_ttt and (hx^2/12) c_xxxx.
    """
    c = lambda xx, tt: erfc_solution(xx, tt, D)
    ct = (c(x, t + ht) - c(x, t - ht)) / (2 * ht)
    cxx = (c(x + hx, t) - 2 * c(x, t) + c(x - hx, t)) / hx ** 2
    return float(ct), float(cxx), float(ct - D * cxx)


def absorbed(t, D: float = 1.0, cs: float = 1.0, c0: float = 0.0):
    """Amount taken up through the surface, M(t) = int_0^inf (c - C0) dx = 2 (Cs - C0) sqrt(D t / pi)."""
    return 2 * (cs - c0) * np.sqrt(D * np.asarray(t, float) / np.pi)


def depth_exact(theta: float, t, D: float = 1.0):
    """Depth where (c - C0)/(Cs - C0) = theta: x = 2 erfcinv(theta) sqrt(D t)."""
    return 2 * erfcinv(theta) * np.sqrt(D * np.asarray(t, float))


def depth_interp(x, c, level: float) -> float:
    """First depth where the decreasing profile c(x) falls to `level`, by linear interpolation
    between the two grid points around it (nan if c never reaches the level)."""
    x = np.asarray(x, float)
    c = np.asarray(c, float)
    below = np.nonzero(c <= level)[0]
    if len(below) == 0 or below[0] == 0:
        return float('nan')
    i = below[0]
    return float(x[i - 1] + (c[i - 1] - level) / (c[i - 1] - c[i]) * (x[i] - x[i - 1]))


def two_stage_solution(x, t, tb: float, D: float, cb: float, cd: float, c0: float):
    """Exact profile of the boost-diffuse process by superposition (lecture notes §5.9).

    The surface is held at Cb for 0 < t < tb and at Cd afterwards; D is constant, so
    c = C0 + (Cb - C0) erfc(x/(2 sqrt(D t))) + (Cd - Cb) erfc(x/(2 sqrt(D (t - tb)))) for t > tb.
    """
    x = np.asarray(x, float)
    c = c0 + (cb - c0) * erfc(x / (2 * np.sqrt(D * t)))
    if t > tb:
        c = c + (cd - cb) * erfc(x / (2 * np.sqrt(D * (t - tb))))
    return c
