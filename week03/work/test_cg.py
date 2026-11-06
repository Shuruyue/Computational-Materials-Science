"""Task 2 tests: matrix-free laplacian, CG, steepest descent. Run: pytest -q"""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from cg import (cg, error_source, f_rhs, g, g2, g4, grid, kappa, lap, poisson_dst, sd, u_exact)


def sparse_laplacian(m):
    h = 1.0 / (m + 1)
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    return ((sp.kron(sp.eye(m), T) + sp.kron(T, sp.eye(m))) / h**2).tocsc(), h


def test_lap_is_the_5_point_matrix():
    m = 7
    A, h = sparse_laplacian(m)
    U = np.random.default_rng(1).uniform(-1, 1, (m, m))
    assert np.allclose(lap(U, h).ravel(), A @ U.ravel(), rtol=1e-14, atol=1e-12)


def test_cg_and_dst_match_direct_solve():
    m = 7
    A, h = sparse_laplacian(m)
    _, X, Y = grid(m)
    F = f_rhs(X, Y)
    direct = spsolve(A, F.ravel()).reshape(m, m)
    assert np.abs(cg(F, h, 1e-12)[0] - direct).max() < 1e-10
    assert np.abs(poisson_dst(F, h) - direct).max() < 1e-13


def test_cg_finite_termination():
    # m = 3: 9 unknowns, so CG ends in at most 9 steps (in exact arithmetic)
    h, X, Y = grid(3)
    _, k, hist = cg(f_rhs(X, Y), h, 1e-13)
    assert k <= 9 and hist[-1] < 1e-13


def test_derivatives_of_g():
    # g = P0(x) e^x with P0 = x - x^2; (P e^x)' = (P + P') e^x, so P_{n+1} = P_n + P_n' exactly
    P = [np.polynomial.Polynomial([0, 1, -1])]
    for _ in range(4):
        P.append(P[-1] + P[-1].deriv())
    x = np.linspace(0.0, 1.0, 11)
    assert np.allclose(g(x), P[0](x) * np.exp(x), rtol=1e-15, atol=1e-16)
    assert np.allclose(g2(x), P[2](x) * np.exp(x), rtol=1e-15, atol=1e-16)
    assert np.allclose(g4(x), P[4](x) * np.exp(x), rtol=1e-15, atol=1e-16)


def test_manufactured_solution_and_error_term():
    # -lap_h(u) - f = -(h^2/12)(u_xxxx + u_yyyy) + O(h^4) = -h^2 * error_source + O(h^4) at interior points
    m = 63
    h, X, Y = grid(m)
    u = u_exact(X, Y)
    inner = (slice(1, -1), slice(1, -1))                 # lap() treats outside as 0, u vanishes there too
    resid = lap(u, h) - f_rhs(X, Y)
    assert np.allclose(resid[inner] / h**2, -error_source(X, Y)[inner], rtol=0, atol=5e-3)


def test_kappa_formula():
    A, h = sparse_laplacian(7)
    ev = np.linalg.eigvalsh(A.toarray())
    assert ev[-1] / ev[0] == pytest.approx(kappa(h), rel=1e-12)


def test_cg_iterations_and_sd_prediction():
    h, X, Y = grid(15)
    F = f_rhs(X, Y)
    _, k_cg, _ = cg(F, h, 1e-8)
    bound = 0.5 * np.sqrt(kappa(h)) * np.log(2 / 1e-8)
    assert k_cg < bound
    _, k_sd, _ = sd(F, h, 1e-8)
    assert k_sd == pytest.approx(kappa(h) / 2 * np.log(1e8), rel=0.05)
