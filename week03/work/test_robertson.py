"""Task 4 tests: Robertson right-hand side, Jacobian, eigenvalues, RK4 and backward Euler. Run: pytest -q"""
import numpy as np
import pytest

from robertson import (LIT_T40, RK4_LIMIT, Y0, backward_euler_newton, fast_slow_eigenvalues, jac, log_grid,
                       rhs, rk4_fixed, solve)
from ode import rk4_real_limit

Y_T40 = np.array([0.7158270687, 9.185534765e-6, 0.2841637457])


def test_mass_conservation_structure():
    # rates and every column of J sum to 0, up to the round-off of adding terms of that size
    eps = np.finfo(float).eps
    for y in (Y0, Y_T40, np.array([0.3, 2e-5, 0.7])):
        f = rhs(0, y)
        J = jac(0, y)
        assert abs(f.sum()) <= 4 * eps * np.abs(f).max()
        assert np.abs(J.sum(axis=0)).max() <= 4 * eps * np.abs(J).max()


def test_jacobian_against_complex_step():
    y = Y_T40
    J = jac(0, y)
    for j in range(3):
        dy = np.zeros(3, complex)
        dy[j] = 1e-30j
        col = np.imag(rhs(0, y + dy)) / 1e-30
        assert np.allclose(J[:, j], col, rtol=1e-13, atol=0)


def test_fast_slow_eigenvalues():
    lam_fast, lam_slow = fast_slow_eigenvalues(Y_T40)
    ev = np.sort(np.linalg.eigvals(jac(0, Y_T40)).real)
    assert lam_fast == pytest.approx(ev[0], rel=1e-12)
    assert lam_slow == pytest.approx(ev[1], rel=1e-9)
    assert abs(ev[2]) < 1e-12
    assert lam_fast == pytest.approx(-3393, abs=1)                 # lecture 4.10, t = 40
    assert fast_slow_eigenvalues(Y0) == (pytest.approx(-0.04), pytest.approx(0.0))


def test_rk4_limit_constant():
    assert RK4_LIMIT == pytest.approx(-rk4_real_limit(), rel=1e-15)


def test_log_grid():
    t = log_grid(10)
    assert t[0] == 0 and t[1] == pytest.approx(4e-7, rel=1e-15)
    assert len(t) - 1 == 171 and t[-1] == 4e10 and 40.0 in t
    assert np.all(np.diff(t) > 0)


def test_backward_euler_newton_short():
    t, Y, iters = backward_euler_newton(10, t_end=40.0)
    assert t[-1] == 40.0
    assert np.abs(Y.sum(axis=1) - 1).max() < 1e-14
    assert 3 <= iters.mean() <= 4
    assert np.max(np.abs(Y[-1] / Y_T40 - 1)) == pytest.approx(3.7e-2, rel=0.02)


def test_backward_euler_newton_warns_when_not_converged():
    # one Newton iteration cannot reach tol = 1e-10 on the first, stiffest steps
    with pytest.warns(RuntimeWarning, match="Newton did not converge"):
        backward_euler_newton(10, t_end=40.0, maxit=1)


def test_rk4_fixed_stable_and_unstable():
    good = rk4_fixed(60000)
    assert not good['blew_up'] and np.max(np.abs(good['y'] / Y_T40 - 1)) < 1e-9
    bad = rk4_fixed(40000)
    assert bad['blew_up'] and 25 < bad['t'] < 40 and bad['t_neg'] < bad['t']
    assert abs(bad['mass_err_neg']) < 1e-13


def test_reference_solution_at_t40():
    s = solve('Radau', 40.0, 1e-10, [1e-14, 1e-18, 1e-14], t_eval=[40.0])
    assert np.allclose(s.y[:, -1], LIT_T40, rtol=1e-8, atol=0)
