"""Task 1 tests: Thomas, 1D Poisson and the CVD pore. Run: pytest -q"""
import numpy as np
import pytest
from scipy.linalg import solve_banded

from cvd import (cvd_discrete, cvd_exact, cvd_solve, effectiveness, s_limit, step_coverage, thiele)
from linear_algebra import tridiag_matrix
from tridiagonal import poisson_1d, poisson_discrete_error, thomas


# (a) the two hand examples (lecture notes §0.2 and exercise 4.2)
def test_thomas_hand_examples():
    assert np.allclose(thomas([0, 1, 1], [2, 3, 4], [1, 1, 0], [4, 10, 14]), [1, 2, 3], rtol=0, atol=1e-15)
    one = np.ones(4)
    assert np.allclose(thomas(-one, 2 * one, -one, [1, 0, 0, 1]), 1, rtol=0, atol=1e-15)


def test_thomas_random_against_lapack():
    rng = np.random.default_rng(0)
    n = 1000
    a = rng.uniform(-1, 1, n)
    c = rng.uniform(-1, 1, n)
    b = 4 + rng.uniform(0, 1, n)
    d = rng.uniform(-1, 1, n)
    x = thomas(a, b, c, d)
    dense = np.linalg.solve(tridiag_matrix(a, b, c), d)
    ab = np.zeros((3, n))
    ab[0, 1:], ab[1], ab[2, :-1] = c[:-1], b, a[1:]
    banded = solve_banded((1, 1), ab, d)
    assert np.abs(x - dense).max() / np.abs(dense).max() < 1e-12
    assert np.abs(x - banded).max() / np.abs(banded).max() < 1e-12


def test_poisson_quadratic_is_exact():
    # Exercise 4.1: -u'' = 1, m = 4 -> U = u(x_j) = (0.08, 0.12, 0.12, 0.08); D^2 is exact for quadratics
    one = np.ones(4)
    assert np.allclose(thomas(-one, 2 * one, -one, 0.04 * one), [0.08, 0.12, 0.12, 0.08], rtol=1e-14)


# (c) the discrete solution is A sin(pi x): the program follows the exact error formula
@pytest.mark.parametrize("k", [2, 4, 6, 8, 10])
def test_poisson_exact_discrete_error(k):
    x, U, h = poisson_1d(k)
    err = np.abs(U - np.sin(np.pi * x)).max()
    assert err == pytest.approx(poisson_discrete_error(h), rel=1e-5)


def test_poisson_error_formula():
    # h = 1/4: A = 1.05303 (lecture 4.1); small h: A - 1 -> pi^2 h^2/12 without cancellation
    assert 1 + poisson_discrete_error(0.25) == pytest.approx(1.05303, abs=1e-5)
    z = np.pi * 0.25 / 2
    assert poisson_discrete_error(0.25) == pytest.approx(z**2 / np.sin(z) ** 2 - 1, rel=1e-13)
    h = 1e-6
    assert poisson_discrete_error(h) == pytest.approx(np.pi**2 * h**2 / 12, rel=1e-11)


# (d) CVD: the program reproduces the exact solution of its own difference equations
@pytest.mark.parametrize("bc", ["ghost", "one-sided"])
@pytest.mark.parametrize("phi, N", [(0.3, 10), (2.0, 10), (2.0, 160), (10.0, 200)])
def test_cvd_matches_discrete_closed_form(bc, phi, N):
    # what is left is round-off, which grows like N^2 eps (the condition number of the system)
    _, c = cvd_solve(phi, N, bc)
    assert np.abs(c - cvd_discrete(phi, N, bc)).max() < N**2 * np.finfo(float).eps


def test_cvd_lecture_numbers():
    # Lecture notes §4.4: phi = 2, N = 10 ghost -> 0.266654; N = 20 -> 0.266016; exact 1/cosh 2 = 0.265802
    assert cvd_solve(2.0, 10)[1][-1] == pytest.approx(0.266654, abs=5e-7)
    assert cvd_solve(2.0, 20)[1][-1] == pytest.approx(0.266016, abs=5e-7)
    assert step_coverage(2.0) == pytest.approx(0.265802, abs=5e-7)


def test_cvd_orders():
    # ghost: error ~ SC tanh(phi) phi^3 h^2/24; one-sided: error ~ SC tanh(phi) phi h/2
    phi = 2.0
    k = step_coverage(phi) * np.tanh(phi)
    for N in (160, 320):
        xi, cg = cvd_solve(phi, N, 'ghost')
        _, co = cvd_solve(phi, N, 'one-sided')
        assert np.abs(cg - cvd_exact(phi, xi)).max() * N**2 == pytest.approx(k * phi**3 / 24, rel=1e-3)
        assert np.abs(co - cvd_exact(phi, xi)).max() * N == pytest.approx(k * phi / 2, rel=2e-2)


def test_step_coverage_design():
    # Exercise 4.4: AR = 20, s = 1e-3 -> phi = 1.095, SC = 0.602, eta = 0.729; SC >= 0.9 needs s <= 1.82e-4
    phi = thiele(20, 1e-3)
    assert phi == pytest.approx(1.0954, abs=1e-4)
    assert step_coverage(phi) == pytest.approx(0.602, abs=1e-3)
    assert effectiveness(phi) == pytest.approx(0.729, abs=1e-3)
    assert s_limit(20) == pytest.approx(1.82e-4, rel=2e-3)
    assert step_coverage(thiele(20, s_limit(20))) == pytest.approx(0.9, rel=1e-12)
