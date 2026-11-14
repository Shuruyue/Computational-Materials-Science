"""Task 0 tests: error function, the erfc solution and its derived quantities. Run: pytest -q"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, erfc, erfcinv

from diffusion import (absorbed, ct_exact, depth_exact, depth_interp, erf_simpson, erfc_solution, erfc_x_derivative,
                       pde_residual, two_stage_solution)

Z = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


@pytest.mark.parametrize('z', Z)
def test_erf_simpson_matches_scipy(z):
    assert abs(erf_simpson(z) - erf(z)) < 1e-12


def test_error_function_values():
    # lecture notes §0.3 table and the assignment
    assert erf(1) == pytest.approx(0.842701, abs=5e-7)
    assert erfc(2) == pytest.approx(4.6777e-3, rel=1e-4)
    assert erfcinv([0.5, 0.25, 0.01]) == pytest.approx([0.476936, 0.813420, 1.821386], abs=5e-7)


@pytest.mark.parametrize('x, t', [(0.5, 0.2), (1.0, 1.0), (2.0, 3.0)])
def test_erfc_solution_satisfies_diffusion_equation(x, t):
    ct, cxx, res = pde_residual(x, t)
    assert abs(res) / abs(ct) < 1e-5
    assert ct == pytest.approx(ct_exact(x, t), rel=1e-6)


@pytest.mark.parametrize('D', [1.0, 2.5])
def test_ct_exact_against_complex_step(D):
    # complex step in t has no subtractive cancellation: d c/dt = Im c(x, t + ih) / h exactly to round-off
    x, t, h = 0.8, 0.6, 1e-30
    ct_cs = np.imag(erfc_solution(x, t + 1j * h, D)) / h
    assert ct_cs == pytest.approx(ct_exact(x, t, D), rel=1e-12)
    # and D c_xx from the complex-step derivative of c_x
    cx = lambda xx: np.imag(erfc_solution(xx + 1j * h, t, D)) / h
    hx = 1e-4
    cxx = (cx(x + hx) - cx(x - hx)) / (2 * hx)
    assert D * cxx == pytest.approx(ct_exact(x, t, D), rel=1e-7)


def test_hermite_derivatives():
    x, t, h = 0.7, 0.4, 1e-30
    cx_cs = np.imag(erfc_solution(x + 1j * h, t)) / h
    assert erfc_x_derivative(x, t, 1) == pytest.approx(cx_cs, rel=1e-13)
    assert erfc_x_derivative(x, t, 2) == pytest.approx(ct_exact(x, t), rel=1e-13)    # c_xx = c_t (D = 1)
    # each derivative is the slope of the previous one
    for k in (2, 3, 4, 5, 6):
        hx = 1e-5
        fd = (erfc_x_derivative(x + hx, t, k - 1) - erfc_x_derivative(x - hx, t, k - 1)) / (2 * hx)
        assert erfc_x_derivative(x, t, k) == pytest.approx(fd, rel=1e-6)


def test_residual_is_the_week2_truncation_error():
    # c_t - c_xx of the central differences = (ht^2/6) c_ttt - (hx^2/12) c_xxxx + O(h^4), with c_ttt = c_xxxxxx
    for x, t in ((0.5, 0.2), (1.0, 1.0)):
        ct, cxx, res = pde_residual(x, t)
        pred = 1e-8 / 6 * erfc_x_derivative(x, t, 6) - 1e-6 / 12 * erfc_x_derivative(x, t, 4)
        assert res == pytest.approx(pred, rel=0.02)


def test_absorbed_amount_is_the_integral():
    for D, t in ((1.0, 1.0), (2.0, 0.7)):
        val = quad(lambda x: erfc(x / (2 * np.sqrt(D * t))), 0, np.inf, epsabs=1e-14)[0]
        assert val == pytest.approx(absorbed(t, D), rel=1e-10)
    assert absorbed(1.0) == pytest.approx(2 / np.sqrt(np.pi), rel=1e-15)


def test_lecture_examples_5_3():
    assert erfc_solution(1, 1) == pytest.approx(0.4795, abs=5e-5)
    assert erfc_solution(2, 1) == pytest.approx(0.1573, abs=5e-5)
    assert erfc_solution(1, 0.25) == erfc_solution(2, 1)              # same x / sqrt(t)
    # hand-calculation exercise 5.3: 925 C, t = 6 h, 0.20 -> 1.00 wt%
    D, t = 1.745e-11, 6 * 3600
    assert erfc_solution(0.5e-3, t, D, 1.0, 0.2) == pytest.approx(0.652, abs=5e-4)
    assert depth_exact(0.5, t, D) == pytest.approx(0.586e-3, rel=2e-3)
    assert depth_exact(0.5, 2 * t, D) / depth_exact(0.5, t, D) == pytest.approx(np.sqrt(2), rel=1e-14)
    assert absorbed(4.0) == pytest.approx(2.257, abs=5e-4)
    assert depth_exact(0.5, 1.0) == pytest.approx(0.9539, abs=5e-5)


def test_depth_interp():
    x = np.linspace(0, 2, 21)
    assert depth_interp(x, 1 - 0.5 * x, 0.4) == pytest.approx(1.2, abs=1e-14)   # exact on a straight line
    assert np.isnan(depth_interp(x, np.ones_like(x), 0.4))
    # a profile with a subsurface maximum: the first crossing below the level is found
    c = np.array([0.8, 1.0, 0.9, 0.5, 0.3, 0.2])
    assert depth_interp(np.arange(6.0), c, 0.4) == pytest.approx(3.5)


def test_two_stage_superposition():
    D, tb, cb, cd, c0 = 1.7e-11, 3 * 3600.0, 1.1, 0.8, 0.2
    x = np.linspace(0, 3e-3, 31)
    # before and at the switch it is the single-stage solution at Cb
    assert np.array_equal(two_stage_solution(x, tb, tb, D, cb, cd, c0), erfc_solution(x, tb, D, cb, c0))
    # afterwards the surface sits at Cd, and the profile is continuous through the switch
    assert two_stage_solution(0.0, 2 * tb, tb, D, cb, cd, c0) == pytest.approx(cd, abs=1e-15)
    after = two_stage_solution(x[1:], tb * (1 + 1e-9), tb, D, cb, cd, c0)
    assert np.allclose(after, erfc_solution(x[1:], tb, D, cb, c0), atol=1e-6)
