"""Tests for the Morse-potential Cu model (Task 4). Run: pytest -q"""
import numpy as np
import pytest

from finite_diff import complex_step, second
from morse import (ALPHA, D_MORSE, R0, _g, bulk_modulus, energy, energy_rescreened,
                   equilibrium, fcc_neighbours, fcc_shells, noisy, phi,
                   second_difference_error)

S = fcc_neighbours(5.5)


def test_phi_minimum():
    assert phi(R0) == pytest.approx(-D_MORSE, rel=1e-15)
    assert phi(R0, 1) == pytest.approx(0.0, abs=1e-15)
    assert phi(R0, 2) == pytest.approx(2 * D_MORSE * ALPHA**2, rel=1e-14)


# Every analytic derivative against the complex step of the one below it (exact to round-off).
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_phi_derivatives(n):
    r = np.array([2.3, 2.866, 3.5, 6.0])
    assert np.allclose(phi(r, n + 1), complex_step(lambda z: phi(z, n), r, 1e-20), rtol=1e-13)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_energy_derivatives(n):
    a = np.array([3.5, 3.6064, 3.7])
    assert np.allclose(energy(a, S, n + 1), complex_step(lambda z: energy(z, S, n), a, 1e-20),
                       rtol=1e-12)


def test_neighbour_list():
    assert len(S) == 2778
    assert S.max() <= 5.5
    radii, counts = fcc_shells(S)
    assert list(counts[:8]) == [12, 6, 24, 12, 24, 8, 48, 6]      # FCC coordination shells
    assert np.allclose(radii[:3] ** 2, [0.5, 1.0, 1.5])


def test_equilibrium_and_bulk_modulus():
    a0 = equilibrium(S)
    assert abs(energy(a0, S, 1)) < 1e-12
    assert energy(a0, S, 2) > 0
    # Lecture exercise 3.8: E'' = 7.07 eV/Å^2, a0 = 3.606 Å -> B = 139.6 GPa.
    assert bulk_modulus(7.07, 3.606) == pytest.approx(139.6, abs=0.1)


def test_g_series_meets_closed_form():
    for y in (0.999999, 1.0):
        closed = 2 * (np.cosh(y) - 1) - y**2
        assert _g(np.array([y]))[0] == pytest.approx(closed, rel=1e-13)
    assert _g(np.array([1e-3]))[0] == pytest.approx(1e-12 / 12, rel=1e-6)


# Closed-form truncation error of D^2 E agrees with the finite difference itself.
@pytest.mark.parametrize("h", [0.5, 0.1, 0.03])
def test_second_difference_error_closed_form(h):
    a0 = equilibrium(S)
    E = lambda a: energy(a, S)
    fd_err = second(E, a0, h) - energy(a0, S, 2)
    assert second_difference_error(a0, h, S) == pytest.approx(fd_err, rel=1e-8)


def test_second_difference_error_leading_term():
    a0 = equilibrium(S)
    h = 1e-3
    assert second_difference_error(a0, h, S) / h**2 == pytest.approx(energy(a0, S, 4) / 12, rel=1e-5)


def test_noise_is_reproducible_and_bounded():
    E = lambda a: energy(a, S)
    a = np.full(1000, 3.6)
    e1 = noisy(E, 1e-5, np.random.default_rng(7))(a)
    e2 = noisy(E, 1e-5, np.random.default_rng(7))(a)
    assert np.array_equal(e1, e2)
    assert np.all(abs(e1 - E(3.6)) <= 1e-5 + 1e-15)


def test_rescreened_jump():
    s_all = fcc_neighbours(4.0)
    # Away from a shell crossing the rebuilt list is a fixed list.
    a = 3.6
    assert energy_rescreened(a, s_all) == pytest.approx(energy(a, s_all[s_all * a <= 10.0]), rel=1e-13)
    # Shell |s|^2 = 7.5 (48 sites) leaves the 10 Å sphere at a = 10/sqrt(7.5).
    a_j = 10 / np.sqrt(7.5)
    jump = energy_rescreened(a_j + 1e-9, s_all) - energy_rescreened(a_j - 1e-9, s_all)
    assert jump == pytest.approx(-0.5 * 48 * phi(10.0), rel=1e-5)
