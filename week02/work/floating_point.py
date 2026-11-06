"""Week 2, Task 0: floating-point experiments.

Lecture notes §3.1 (machine epsilon, spacing of floats) and §3.2 (catastrophic cancellation).
"""
from __future__ import annotations

import numpy as np


def machine_epsilon(dtype=np.float64):
    """Halve e from 1 until 1 + e/2 rounds back to 1. Returns (e, number of halvings).

    All arithmetic stays in `dtype`, so the loop stops at the gap between 1 and the
    next float: 2^-52 for float64 (52 mantissa bits), 2^-23 for float32.
    """
    one, two = dtype(1), dtype(2)
    e, n = one, 0
    while one + e / two != one:
        e = e / two
        n += 1
    return e, n


def one_minus_cos_direct(x):
    """(1 - cos x)/x^2 as written. For small x, cos x is within a few spacings of 1 and the
    subtraction 1 - cos x keeps only those last few (wrong) bits."""
    x = np.asarray(x, float)
    return (1 - np.cos(x)) / x**2


def one_minus_cos_stable(x):
    """Same function via the identity 1 - cos x = 2 sin^2(x/2): no subtraction anywhere."""
    x = np.asarray(x, float)
    return 2 * np.sin(x / 2) ** 2 / x**2


def small_root_direct(b: float, c: float) -> float:
    """Small root of x^2 - b x + c = 0 (b > 0, b^2 >> 4c) from [b - sqrt(b^2 - 4c)]/2.
    b and sqrt(b^2 - 4c) agree to almost every digit, so the difference is mostly round-off."""
    return (b - np.sqrt(b * b - 4 * c)) / 2


def small_root_stable(b: float, c: float) -> float:
    """Same root as c / (large root), because x1 x2 = c. The large root [b + sqrt(b^2 - 4c)]/2
    adds two positive numbers, so nothing cancels."""
    return c / ((b + np.sqrt(b * b - 4 * c)) / 2)
