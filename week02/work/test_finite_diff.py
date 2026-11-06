"""Tests for the difference formulas (Tasks 1-2). Run: pytest -q

Each test checks a property the derivation predicts: exactness on low-degree polynomials,
the lecture's hand exercises, the stencil error models, and the full error of sin x at x = 1
(exact truncation series + round-off within the lecture's bound).
"""
from math import factorial

import numpy as np
import pytest

from finite_diff import (EPS, STENCILS, actual_step, backward, central, complex_step,
                         complex_step_truncation_error, error_model, five_point, forward,
                         one_sided3, optimal_step, richardson5, second, truncation_error)

FORMULAS = {'D+': forward, 'D-': backward, 'D0': central, 'D3': one_sided3, 'D2': second,
            'D5': richardson5}


# 1. Exact for polynomials up to the degree the error term allows (error involves f^(m+p)).
@pytest.mark.parametrize("name, degree", [('D+', 1), ('D-', 1), ('D0', 2), ('D3', 3),
                                          ('D2', 3), ('D5', 4)])
def test_exact_for_polynomials(name, degree):
    coeffs = np.arange(1.0, degree + 2)               # 1 + 2x + 3x^2 + ...
    p = np.polynomial.Polynomial(coeffs)
    m = STENCILS[name][2]
    x, h = 0.7, 0.25
    assert FORMULAS[name](p, x, h) == pytest.approx(p.deriv(m)(x), rel=1e-13)


def test_complex_step_exact_for_quadratic():
    # Lecture exercise 3.7: f = x^2, x = 3, h = 0.5 -> Im(8.75 + 3i)/0.5 = 6.
    assert complex_step(lambda z: z**2, 3.0, 0.5) == 6.0


# 2. Lecture hand exercises 3.3, 3.5, 3.7.
def test_lecture_exercises():
    cube = lambda x: x**3
    assert forward(cube, 1.0, 0.1) == pytest.approx(3.31, abs=1e-12)
    assert central(cube, 1.0, 0.1) == pytest.approx(3.01, abs=1e-12)
    assert second(lambda x: x**4, 1.0, 0.1) == pytest.approx(12.02, abs=1e-10)
    assert complex_step(cube, 1.0, 0.1) == pytest.approx(2.99, abs=1e-12)


# 3. Error models from the stencil moments: order p, leading coefficient C, round-off coefficient K.
@pytest.mark.parametrize("name, p, C, K", [('D+', 1, 1 / 2, 2), ('D-', 1, -1 / 2, 2),
                                          ('D0', 2, 1 / 6, 1), ('D3', 3, 1 / 12, 2),
                                          ('D2', 2, 1 / 12, 4), ('D5', 4, -1 / 30, 1.5)])
def test_error_model(name, p, C, K):
    assert error_model(STENCILS[name]) == (p, pytest.approx(C, rel=1e-15), K)


# The explicit functions implement exactly the tabulated stencils.
@pytest.mark.parametrize("name", list(STENCILS))
def test_formulas_match_stencils(name):
    offsets, weights, m = STENCILS[name]
    x, h = 0.3, 0.05
    by_stencil = sum(float(w) * np.exp(x + j * h) for j, w in zip(offsets, weights)) / h**m
    assert FORMULAS[name](np.exp, x, h) == pytest.approx(by_stencil, rel=1e-12)


# Richardson on D0 is the five-point formula.
@pytest.mark.parametrize("h", [0.1, 0.01, 0.001])
def test_richardson_is_five_point(h):
    assert abs(richardson5(np.sin, 1.0, h) - five_point(np.sin, 1.0, h)) < 1e-12


# 4. The whole error of sin x at x = 1, for every h on Task 1's grid:
#    computed - exact derivative = exact truncation series + round-off, |round-off| <= K eps|f|/h^m.
SIN_DERIVS = [(np.sin(1.0), np.cos(1.0), -np.sin(1.0), -np.cos(1.0))[n % 4] for n in range(41)]
H = 10.0 ** (-np.arange(65) / 4)


@pytest.mark.parametrize("name", list(STENCILS))
def test_sin_error_is_truncation_plus_bounded_roundoff(name):
    stencil = STENCILS[name]
    m = stencil[2]
    _, _, K = error_model(stencil)
    h = H[H >= 1e-6] if m == 2 else H[H >= 1e-15]
    exact = SIN_DERIVS[m]
    roundoff = FORMULAS[name](np.sin, 1.0, h) - exact - truncation_error(stencil, SIN_DERIVS, h)
    bound = K * EPS * np.sin(1.0) / h**m + 4 * EPS * abs(exact)
    assert np.all(abs(roundoff) <= bound)


def test_sin_truncation_series_leading_terms():
    # Lecture 3.3 table: D+ error -0.4207 h, D0 error -0.09005 h^2 (leading terms).
    h = 1e-4
    assert truncation_error(STENCILS['D+'], SIN_DERIVS, h) == pytest.approx(-np.sin(1) / 2 * h, rel=1e-3)
    assert truncation_error(STENCILS['D0'], SIN_DERIVS, h) == pytest.approx(-np.cos(1) / 6 * h**2, rel=1e-7)


# 5. Complex step: closed form cos 1 * sinh(h)/h, and exact to the last bit for tiny h.
@pytest.mark.parametrize("h", [0.3, 0.1, 1e-2, 1e-4])
def test_complex_step_closed_form(h):
    exact_err = np.cos(1.0) * sum(h ** (2 * k) / factorial(2 * k + 1) for k in range(1, 15))
    assert complex_step_truncation_error(SIN_DERIVS, h) == pytest.approx(exact_err, rel=1e-14)
    assert complex_step(np.sin, 1.0, h) - np.cos(1.0) == pytest.approx(exact_err, rel=1e-9)


@pytest.mark.parametrize("h", [1e-8, 1e-12, 1e-20, 1e-100])
def test_complex_step_no_roundoff_growth(h):
    assert abs(complex_step(np.sin, 1.0, h) - np.cos(1.0)) <= np.spacing(np.cos(1.0))


# 6. The step really used: exact for powers of 2 at x = 1, rounded otherwise (lecture 3.4).
def test_actual_step():
    for k in (1, 10, 30, 52):
        assert actual_step(1.0, 2.0**-k) == 2.0**-k
    assert abs(actual_step(1.0, 1e-10) / 1e-10 - 1) == pytest.approx(8.3e-8, rel=0.01)
    assert actual_step(1.0, 1e-16) == 0.0


# 7. Optimal step of e(h) = A h^p + B/h^q (lecture 0.3).
def test_optimal_step_lecture_example():
    h, e = optimal_step(1.0, 2, 1e-6, 1)                # e = h^2 + 1e-6/h
    assert h == pytest.approx(7.94e-3, rel=1e-3)
    assert e == pytest.approx(1.89e-4, rel=1e-2)
    h, e = optimal_step(0.5, 1, 2e-16, 1)               # exercise 0: 0.5 h + 2e-16/h
    assert (h, e) == (pytest.approx(2e-8), pytest.approx(2e-8))


@pytest.mark.parametrize("p, q", [(1, 1), (2, 1), (2, 2), (4, 1)])
def test_optimal_step_is_minimum(p, q):
    A, B = 0.3, 1e-15
    h, e = optimal_step(A, p, B, q)
    err = lambda t: A * t**p + B / t**q
    assert e == pytest.approx(err(h), rel=1e-12)
    assert err(h * 1.01) > e and err(h * 0.99) > e
