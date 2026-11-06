"""Week 2, Tasks 1-2: finite-difference derivatives and their error models.

Formulas follow lecture notes §3.3-3.7; D3 is LeVeque eq. (1.3).

Conventions: f is a vectorised callable, x the point, h > 0 the step
(x or h may be arrays; the formulas broadcast).
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial

import numpy as np

EPS = np.finfo(float).eps   # 2.22e-16, the course's epsilon (lecture notes §3.1)


def forward(f, x, h):
    """D+ f = [f(x+h) - f(x)]/h. Error (h/2) f'' + O(h^2)."""
    return (f(x + h) - f(x)) / h


def backward(f, x, h):
    """D- f = [f(x) - f(x-h)]/h. Error -(h/2) f'' + O(h^2)."""
    return (f(x) - f(x - h)) / h


def central(f, x, h):
    """D0 f = [f(x+h) - f(x-h)]/(2h). Error (h^2/6) f''' + O(h^4); even powers of h only."""
    return (f(x + h) - f(x - h)) / (2 * h)


def one_sided3(f, x, h):
    """LeVeque's third-order D3 f = [2f(x+h) + 3f(x) - 6f(x-h) + f(x-2h)]/(6h).
    Error (h^3/12) f'''' + O(h^4)."""
    return (2 * f(x + h) + 3 * f(x) - 6 * f(x - h) + f(x - 2 * h)) / (6 * h)


def second(f, x, h):
    """D^2 f = [f(x+h) - 2f(x) + f(x-h)]/h^2. Error (h^2/12) f'''' + O(h^4)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def richardson5(f, x, h):
    """D5 = [4 D0(h) - D0(2h)]/3: multiply by 4, subtract once, and the h^2 term of D0 is gone.
    Error -(h^4/30) f^(5) + O(h^6)."""
    return (4 * central(f, x, h) - central(f, x, 2 * h)) / 3


def five_point(f, x, h):
    """The same D5 written out: [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)]/(12h)."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def complex_step(f, x, h):
    """Im f(x + ih)/h. Error -(h^2/6) f''' + O(h^4); nothing is subtracted, so round-off is not
    amplified by 1/h. f must be built from operations that accept complex input."""
    return np.imag(f(x + 1j * h)) / h


def actual_step(x, h):
    """Numerical Recipes' temp = x + h; h = temp - x: the step the formula really uses once
    x + h has been rounded to a float. Exact when x + h is representable."""
    return (x + h) - x


# ---------------------------------------------------------------------------------------
# Error models. A linear stencil is (offsets j, weights w_j, derivative order m), meaning
#     D f(x) = sum_j w_j f(x + j h) / h^m.
# Taylor-expanding every f(x + j h) gives  D f = sum_n M_n f^(n)(x) h^(n-m) / n!  with the
# moments M_n = sum_j w_j j^n. Exact rationals keep the zero moments exactly zero.

STENCILS = {
    'D+': ((0, 1), (-1, 1), 1),
    'D-': ((-1, 0), (-1, 1), 1),
    'D0': ((-1, 1), (Fraction(-1, 2), Fraction(1, 2)), 1),
    'D3': ((-2, -1, 0, 1), (Fraction(1, 6), -1, Fraction(1, 2), Fraction(1, 3)), 1),
    'D2': ((-1, 0, 1), (1, -2, 1), 2),
    'D5': ((-2, -1, 1, 2), (Fraction(1, 12), Fraction(-2, 3), Fraction(2, 3), Fraction(-1, 12)), 1),
}


def moment(stencil, n: int) -> Fraction:
    """M_n = sum_j w_j j^n, exactly."""
    offsets, weights, _ = stencil
    return sum((Fraction(w) * j**n for j, w in zip(offsets, weights)), Fraction(0))


def error_model(stencil):
    """(p, C, K) for the error model |D f - f^(m)| <= |C| h^p |f^(m+p)| + K eps|f| / h^m.

    p : order of accuracy, the first n > m with M_n != 0, minus m.
    C : leading error coefficient M_(m+p) / (m+p)!   (1/2, 1/6, 1/12, 1/12, -1/30 for D+, D0, D3, D2, D5).
    K : round-off coefficient sum_j |w_j|, every function value off by eps|f| with the worst signs
        (2, 1, 4, 1.5 for D+, D0, D2, D5: lecture notes §3.4-3.6).
    """
    offsets, weights, m = stencil
    n = m + 1
    while moment(stencil, n) == 0:
        n += 1
    C = moment(stencil, n) / factorial(n)
    K = sum(abs(Fraction(w)) for w in weights)
    return n - m, float(C), float(K)


def truncation_error(stencil, derivs, h):
    """Exact truncation error D f - f^(m) = sum_{n>m} M_n f^(n)(x) h^(n-m) / n! for a function whose
    derivatives at x are known, derivs[n] = f^(n)(x), n = 0..len(derivs)-1.

    The series is summed term by term (smallest first), never as "D f minus the exact derivative",
    so it stays accurate however small the error is.
    """
    _, _, m = stencil
    h = np.asarray(h, float)
    err = np.zeros_like(h)
    for n in range(len(derivs) - 1, m, -1):
        c = moment(stencil, n) / factorial(n)
        if c:
            err = err + float(c) * derivs[n] * h ** (n - m)
    return err


def complex_step_truncation_error(derivs, h):
    """Exact error of Im f(x + ih)/h: the imaginary part keeps the odd Taylor terms, so
    error = sum_{n >= 3, odd} (-1)^((n-1)/2) f^(n)(x) h^(n-1) / n!   (leading term -(h^2/6) f''')."""
    h = np.asarray(h, float)
    err = np.zeros_like(h)
    for n in range(len(derivs) - 1, 2, -1):
        if n % 2:
            err = err + (-1) ** ((n - 1) // 2) * derivs[n] * h ** (n - 1) / factorial(n)
    return err


def optimal_step(A: float, p: float, B: float, q: float):
    """Minimum of e(h) = A h^p + B / h^q (lecture notes §0.3):
    h* = (qB / (pA))^(1/(p+q)),  e* = (1 + p/q) A h*^p.  Returns (h*, e*)."""
    h = (q * B / (p * A)) ** (1 / (p + q))
    return h, (1 + p / q) * A * h**p
