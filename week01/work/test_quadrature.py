"""Task 0 acceptance tests (code verification). Run in this folder: pytest -q

Each test checks a property the derivation predicts, not just "a number came out".
"""
import numpy as np
import pytest
from scipy.special import i0, iv, zeta

from quadrature import (fit_slope, observed_order, observed_order_3grid,
                        richardson, simpson, trapezoid)


# 1. Degree of precision: trapezoid is exact for linear f, Simpson for cubics.
@pytest.mark.parametrize("N", [1, 2, 7, 64])
def test_trapezoid_exact_for_linear(N):
    # integral of 3x + 1 on [0, 2] = 8
    assert trapezoid(lambda x: 3 * x + 1, 0.0, 2.0, N) == pytest.approx(8.0, abs=1e-14)


@pytest.mark.parametrize("N", [2, 4, 32])
def test_simpson_exact_for_cubic(N):
    f = lambda x: x**3 - 2 * x**2 + 5
    exact = 4**4 / 4 - 2 * 4**3 / 3 + 5 * 4  # integral on [0, 4]
    assert simpson(f, 0.0, 4.0, N) == pytest.approx(exact, abs=1e-12)


# 2. Simpson pairs cells into parabolas, so odd N must be rejected.
@pytest.mark.parametrize("N", [1, 3, 9])
def test_simpson_rejects_odd_N(N):
    with pytest.raises(ValueError):
        simpson(np.sin, 0.0, 1.0, N)


# 3. Error constant (lecture notes §2.6): for f = x^2 on [0, 1], I - T_h = -(b-a) h^2 f''/12 = -h^2/6.
@pytest.mark.parametrize("N", [1, 4, 16, 64, 256])
def test_trapezoid_error_constant(N):
    h = 1.0 / N
    err = 1 / 3 - trapezoid(lambda x: x**2, 0.0, 1.0, N)
    assert err == pytest.approx(-h**2 / 6, rel=1e-10)


# 4. Observed orders on a smooth integrand: integral_0^pi sin x dx = 2.
@pytest.mark.parametrize("rule, p_theory", [(trapezoid, 2), (simpson, 4)])
def test_observed_order_smooth(rule, p_theory):
    Ns = np.array([8, 16, 32, 64])
    errs = [rule(np.sin, 0.0, np.pi, N) - 2.0 for N in Ns]
    assert np.allclose(observed_order(errs, Ns), p_theory, atol=0.05)
    assert fit_slope(np.pi / Ns, errs) == pytest.approx(p_theory, abs=0.05)


# 5. Three-grid order and Richardson need no exact answer: integral_0^1 e^x dx = e - 1.
def test_three_grid_and_richardson():
    exact = np.e - 1
    Q1, Q2, Q3 = (trapezoid(np.exp, 0.0, 1.0, N) for N in (10, 20, 40))
    assert observed_order_3grid(Q1, Q2, Q3) == pytest.approx(2.0, abs=0.01)
    Qstar = richardson(Q2, Q3, p=2)
    assert abs(Qstar - exact) < abs(Q3 - exact)
    assert abs(Qstar - exact) < 1e-2 * abs(Q3 - exact)


# Hand-calculation exercise 2.4 in the lecture notes: Q = 5.1200, 5.0300, 5.0075 -> p = 2, Q* = 5.0000.
def test_lecture_exercise_2_4():
    p = observed_order_3grid(5.1200, 5.0300, 5.0075)
    assert p == pytest.approx(2.0, abs=1e-9)
    assert richardson(5.0300, 5.0075, p) == pytest.approx(5.0, abs=1e-9)


# Lecture 2.7: one Richardson step on trapezoid (p = 2) is exactly Simpson.
def test_simpson_is_richardson_of_trapezoid():
    for N in (2, 8, 32):
        T_coarse = trapezoid(np.sin, 0.0, np.pi, N // 2)
        T_fine = trapezoid(np.sin, 0.0, np.pi, N)
        assert richardson(T_coarse, T_fine, p=2) == pytest.approx(
            simpson(np.sin, 0.0, np.pi, N), abs=1e-14)


# Stronger than "the slope is about right": cases where the whole error is known in closed form.

# sin on [0, pi]: sum_{k=1}^{N-1} sin(k pi/N) = cot(pi/2N), so T_N = (pi/N) cot(pi/2N) exactly.
@pytest.mark.parametrize("N", [1, 2, 4, 16, 1024])
def test_trapezoid_sin_closed_form(N):
    exact_T = 0.0 if N == 1 else np.pi / N / np.tan(np.pi / (2 * N))
    assert trapezoid(np.sin, 0.0, np.pi, N) == pytest.approx(exact_T, abs=1e-13)


# sqrt(x) on [0, 1]: Euler-Maclaurin extended to an x^(1/2) end point (Navot 1961) gives
# T_h - I = zeta(-1/2) h^1.5 + h^2/24 + O(h^4); Simpson = (4 T_h - T_2h)/3 cancels the h^2 term.
@pytest.mark.parametrize("N", [256, 1024, 4096])
def test_sqrt_error_expansion(N):
    h = 1.0 / N
    eT = trapezoid(np.sqrt, 0.0, 1.0, N) - 2 / 3
    eS = simpson(np.sqrt, 0.0, 1.0, N) - 2 / 3
    assert eT == pytest.approx(zeta(-0.5) * h**1.5 + h**2 / 24, rel=1e-7)
    assert eS == pytest.approx(zeta(-0.5) * (4 - 2**1.5) / 3 * h**1.5, rel=1e-6)


# exp(cos x) = I0(1) + 2 sum_m Im(1) cos(mx); the N-cell periodic trapezoid kills every cos(mx)
# except m = N, 2N, ..., so T_N - I = 4 pi sum_{k>=1} I_{kN}(1) exactly.
@pytest.mark.parametrize("N", [2, 4, 6, 8, 10, 12, 14])
def test_periodic_trapezoid_exact_error(N):
    err = trapezoid(lambda x: np.exp(np.cos(x)), 0.0, 2 * np.pi, N) - 2 * np.pi * i0(1.0)
    predicted = 4 * np.pi * sum(iv(k * N, 1.0) for k in range(1, 30))
    assert err == pytest.approx(predicted, abs=1e-14)
