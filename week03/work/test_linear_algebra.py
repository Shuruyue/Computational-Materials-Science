"""Task 0 tests: elimination, eigenvalues and condition numbers. Run: pytest -q"""
from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import hilbert, invhilbert

from linear_algebra import (gauss_solve, hilbert_condition, t_condition, t_eigenvalues, t_matrix,
                            tridiag_matrix)

EPS = np.finfo(float).eps
A3 = np.array([[2.0, 1, 0], [1, 3, 1], [0, 1, 4]])


def test_matvec_and_elimination_by_hand():
    # Lecture notes §0.1-0.2: A (1, 2, 3) = (4, 10, 14); U has diagonal (2, 2.5, 3.6), c = (4, 8, 10.8)
    assert np.array_equal(A3 @ np.array([1.0, 2, 3]), [4.0, 10, 14])
    x, U, c = gauss_solve(A3, [4.0, 10, 14])
    assert np.allclose(x, [1, 2, 3], rtol=0, atol=1e-15)
    assert np.allclose(np.diag(U), [2, 2.5, 3.6], rtol=1e-15)
    assert np.allclose(c, [4, 8, 10.8], rtol=1e-15)
    assert np.array_equal(np.tril(U, -1), np.zeros((3, 3)))


def test_lecture_exercise_0():
    assert np.array_equal(np.array([[1, 2], [3, 4]]) @ np.array([5, 6]), [17, 39])
    x, _, _ = gauss_solve([[4.0, 1], [1, 3]], [9.0, 5])
    assert np.allclose(x, [2, 1], atol=1e-15)
    assert np.allclose(np.linalg.eigvalsh([[3.0, 1], [1, 3]]), [2, 4])


@pytest.mark.parametrize("n", [5, 50, 200])
def test_gauss_matches_lapack(n):
    rng = np.random.default_rng(n)
    A = rng.uniform(-1, 1, (n, n)) + n * np.eye(n)          # diagonally dominant: no pivoting needed
    b = rng.uniform(-1, 1, n)
    assert np.allclose(gauss_solve(A, b)[0], np.linalg.solve(A, b), rtol=1e-12, atol=1e-14)


def test_tridiag_matrix_layout():
    a, b, c = [9.0, 1, 2], [4.0, 5, 6], [7.0, 8, 9]       # a[0] and c[-1] unused
    assert np.array_equal(tridiag_matrix(a, b, c), [[4, 7, 0], [1, 5, 8], [0, 2, 6]])


@pytest.mark.parametrize("m", [3, 10, 37])
def test_t_eigenvalues(m):
    assert np.abs(np.linalg.eigvalsh(t_matrix(m)) - t_eigenvalues(m)).max() < 1e-13
    if m == 3:                                             # 2 - sqrt2, 2, 2 + sqrt2
        assert np.allclose(t_eigenvalues(3), [2 - np.sqrt(2), 2, 2 + np.sqrt(2)], rtol=1e-15)


@pytest.mark.parametrize("m", [10, 100])
def test_t_condition(m):
    assert np.linalg.cond(t_matrix(m)) == pytest.approx(t_condition(m), rel=1e-9)
    assert t_condition(m) == pytest.approx(4 * (m + 1) ** 2 / np.pi**2, rel=1.5e-2)


def test_exact_hilbert_inverse():
    # invhilbert(exact=True) really is the inverse: H H^-1 = I in rational arithmetic
    n = 6
    H = [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]
    Hinv = invhilbert(n, exact=True)
    for i in range(n):
        for j in range(n):
            assert sum(H[i][k] * int(Hinv[k][j]) for k in range(n)) == (1 if i == j else 0)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_hilbert_condition_matches_numpy_while_numpy_still_can(n):
    assert hilbert_condition(n) == pytest.approx(np.linalg.cond(hilbert(n)), rel=1e-3)


def test_hilbert_condition_growth():
    # kappa_2(H_n) ~ C (1 + sqrt2)^(4n) / sqrt(n): the ratio n -> n+2 tends to (1+sqrt2)^8 sqrt(n/(n+2))
    for n in (10, 12):
        ratio = hilbert_condition(n + 2) / hilbert_condition(n)
        assert ratio == pytest.approx((1 + np.sqrt(2)) ** 8 * np.sqrt(n / (n + 2)), rel=0.05)


@pytest.mark.parametrize("n", [4, 6, 8, 10, 12, 14])
def test_hilbert_small_residual_error_below_kappa_eps(n):
    H = hilbert(n)
    b = H @ np.ones(n)
    x = np.linalg.solve(H, b)
    assert np.linalg.norm(b - H @ x) / np.linalg.norm(b) < 1e-15
    assert np.abs(x - 1).max() <= hilbert_condition(n) * EPS
