"""Week 3, Task 1 (d)-(e): CVD in a deep pore, reaction-diffusion c'' = phi^2 c (lecture notes §4.4).

xi = x/L, c = n/n0, c(0) = 1 at the pore mouth, c'(1) = 0 at the bottom,
Thiele modulus phi = AR sqrt(3 s) (AR aspect ratio, s reactive sticking coefficient).
Exact solution c = cosh(phi(1 - xi))/cosh(phi); step coverage SC = c(1) = 1/cosh(phi).
"""
from __future__ import annotations

import numpy as np

from tridiagonal import thomas


def cvd_solve(phi: float, N: int, bc: str = 'ghost'):
    """Finite differences on xi_j = j/N with unknowns c_1..c_N, solved with thomas.

    Interior rows: c_{j-1} - (2 + phi^2 h^2) c_j + c_{j+1} = 0 (c_0 = 1 moved to the right side).
    Pore bottom:
      'ghost'      c_{N+1} = c_{N-1} (central difference)  ->  2 c_{N-1} - (2 + phi^2 h^2) c_N = 0,  O(h^2)
      'one-sided'  (c_N - c_{N-1})/h = 0                     ->  c_{N-1} - c_N = 0,                   O(h)
    Returns (xi, c) including the mouth xi = 0, c = 1.
    """
    h = 1.0 / N
    a = np.ones(N)
    b = np.full(N, -(2 + (phi * h) ** 2))
    c = np.ones(N)
    d = np.zeros(N)
    d[0] = -1.0
    if bc == 'ghost':
        a[-1] = 2.0
    elif bc == 'one-sided':
        b[-1] = -1.0
    else:
        raise ValueError(bc)
    return np.linspace(0.0, 1.0, N + 1), np.concatenate([[1.0], thomas(a, b, c, d)])


def cvd_exact(phi: float, xi):
    """c(xi) = cosh(phi(1 - xi))/cosh(phi)."""
    return np.cosh(phi * (1 - np.asarray(xi, float))) / np.cosh(phi)


def cvd_discrete(phi: float, N: int, bc: str = 'ghost'):
    """Exact solution of the difference equations themselves (no truncation error left to find).

    c_j = cosh(theta (N + s - j)) / cosh(theta (N + s)) with sinh(theta/2) = phi h/2, which makes
    c_{j-1} + c_{j+1} = 2 cosh(theta) c_j = (2 + phi^2 h^2) c_j hold. The shift puts the symmetry
    point on the boundary condition: s = 0 for the ghost point (c_{N+1} = c_{N-1}), s = -1/2 for
    the one-sided form (c_N = c_{N-1}). Returns c_0..c_N.
    """
    theta = 2 * np.arcsinh(phi / N / 2)
    s = {'ghost': 0.0, 'one-sided': -0.5}[bc]
    j = np.arange(N + 1)
    return np.cosh(theta * (N + s - j)) / np.cosh(theta * (N + s))


def step_coverage(phi):
    """SC = c(1)/c(0) = 1/cosh(phi)."""
    return 1 / np.cosh(phi)


def effectiveness(phi):
    """Effectiveness factor eta = tanh(phi)/phi: mean deposition rate over the wall relative to
    a wall that sees the mouth concentration everywhere."""
    return np.tanh(phi) / phi


def thiele(AR, s):
    """phi = AR sqrt(3 s)."""
    return AR * np.sqrt(3 * s)


def s_limit(AR, sc_min: float = 0.9):
    """Largest sticking coefficient with SC >= sc_min: phi <= arccosh(1/sc_min), s = (phi/AR)^2/3."""
    return (np.arccosh(1 / sc_min) / AR) ** 2 / 3
