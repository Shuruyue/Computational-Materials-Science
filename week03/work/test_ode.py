"""Task 3 tests: the four integrators, stability functions and the high-precision rerun. Run: pytest -q"""
import math
from decimal import Decimal

import numpy as np
import pytest

from ode import (R_be, R_fe, R_rk4, R_tr, backward_euler_linear, cos_problem, dcos, dsin, forward_euler, rk4,
                 rk4_real_limit, set_precision, trapezoid_linear)


def test_decimal_trig():
    set_precision(40)
    for x in (0.0, 0.3, 1.0, 2.0, 3.0):
        assert float(dcos(Decimal(x))) == pytest.approx(math.cos(x), rel=0, abs=2e-16)
        assert float(dsin(Decimal(x))) == pytest.approx(math.sin(x), rel=0, abs=2e-16)
    one = dcos(Decimal(1)) ** 2 + dsin(Decimal(1)) ** 2
    assert abs(one - 1) < Decimal(10) ** -38


def test_one_step_table():
    # Lecture notes §4.6: y' = -y, h = 0.1, one step; exact e^-0.1
    f = lambda t, y: -y
    g = lambda t: 0.0
    assert forward_euler(f, 1.0, 0.1, 1)[-1] == pytest.approx(0.9, abs=1e-15)
    assert backward_euler_linear(-1.0, g, 1.0, 0.1, 1)[-1] == pytest.approx(1 / 1.1, abs=1e-15)
    assert trapezoid_linear(-1.0, g, 1.0, 0.1, 1)[-1] == pytest.approx(0.95 / 1.05, abs=1e-15)
    assert rk4(f, 1.0, 0.1, 1)[-1] - math.exp(-0.1) == pytest.approx(8.2e-8, rel=0.01)


def test_lecture_exercise_4_6():
    f = lambda t, y: -2 * y
    g = lambda t: 0.0
    assert forward_euler(f, 1.0, 0.25, 2)[-1] == pytest.approx(0.25, abs=1e-15)
    assert backward_euler_linear(-2.0, g, 1.0, 0.25, 2)[-1] == pytest.approx(4 / 9, abs=1e-15)
    assert trapezoid_linear(-2.0, g, 1.0, 0.25, 2)[-1] == pytest.approx(0.36, abs=1e-15)
    assert rk4(f, 1.0, 0.5, 1)[-1] == pytest.approx(0.375, abs=1e-15)


@pytest.mark.parametrize("R", [R_fe, R_rk4, R_be, R_tr])
def test_stability_function_is_one_step_on_y_equals_lam_y(R):
    lam, h = -3.0, 0.2
    step = {R_fe: lambda: forward_euler(lambda t, y: lam * y, 1.0, h, 1)[-1],
            R_rk4: lambda: rk4(lambda t, y: lam * y, 1.0, h, 1)[-1],
            R_be: lambda: backward_euler_linear(lam, lambda t: 0.0, 1.0, h, 1)[-1],
            R_tr: lambda: trapezoid_linear(lam, lambda t: 0.0, 1.0, h, 1)[-1]}[R]
    assert step() == pytest.approx(R(h * lam), rel=1e-14)


def test_rk4_real_limit():
    z = rk4_real_limit()
    assert z == pytest.approx(-2.785293563, abs=1e-9)
    assert R_rk4(z) == pytest.approx(1.0, abs=1e-14)
    assert R_tr(-1e5) == pytest.approx(-0.99996, abs=1e-8)       # exercise 4.8


def test_cos_problem_has_exact_solution_cos():
    f, g = cos_problem(-2100.0)
    for t in (0.0, 0.7, 2.0):
        assert f(t, math.cos(t)) == pytest.approx(-math.sin(t), abs=1e-12)
        assert -2100.0 * math.cos(t) + g(t) == pytest.approx(f(t, math.cos(t)), abs=1e-12)


def test_leveque_table_7_1_stable_rows():
    # N = 2500, 5000 agree with LeVeque to 6 digits; also with the 40-digit rerun
    f, _ = cos_problem(-2100.0)
    for N, ref in ((2500, -7.92298e-8), (5000, -3.96033e-8)):
        err = forward_euler(f, 1.0, 2 / N, N)[-1] - math.cos(2.0)
        assert err == pytest.approx(ref, rel=1e-5)
    set_precision(40)
    fd, _ = cos_problem(Decimal(-2100), dcos, dsin)
    h = Decimal(2 / 2500)
    exact = forward_euler(fd, Decimal(1), h, 2500)[-1] - dcos(2500 * h)
    assert float(exact) == pytest.approx(forward_euler(f, 1.0, 2 / 2500, 2500)[-1] - math.cos(2.0), rel=1e-8)


def test_leveque_table_8_1_and_l_stability():
    lam, h, N = -1e6, 0.1, 30
    _, g = cos_problem(lam)
    be1, be2 = (backward_euler_linear(lam, g, y0, h, N)[-1] - math.cos(3.0) for y0 in (1.0, 1.5))
    tr1, tr2 = (trapezoid_linear(lam, g, y0, h, N)[-1] - math.cos(3.0) for y0 in (1.0, 1.5))
    assert be1 == pytest.approx(4.9223e-8, rel=1e-4) and be2 == pytest.approx(4.9223e-8, rel=1e-4)
    assert tr1 == pytest.approx(1.1772e-10, rel=1e-3)
    assert tr2 == pytest.approx(0.49940, rel=1e-4)
    # linearity: the extra 0.5 in y(0) is carried by R(z)^N exactly
    assert tr2 - tr1 == pytest.approx(0.5 * R_tr(h * lam) ** N, rel=1e-12)
