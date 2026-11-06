"""Tests for the Debye energy U(T) (Task 3). Run: pytest -q"""
import numpy as np
import pytest

from debye_energy import (R, THETA_CU, debye_cv, debye_cv_series, debye_dcv_series,
                          debye_energy, debye_energy_series, energy_integrand)
from finite_diff import richardson5


# Trap 1: x -> 0 is 0/0 in the textbook form; the safe form follows x^2 (1 - x/2 + x^2/12).
def test_integrand_near_zero():
    x = np.array([0.0, 1e-12, 1e-9, 1e-6, 1e-3])
    assert np.allclose(energy_integrand(x), x**2 * (1 - x / 2 + x**2 / 12), rtol=1e-12, atol=0)


# Trap 2: e^x overflows for x > ~709; the safe form stays finite and tends to 0.
def test_integrand_no_overflow():
    with np.errstate(all="raise", under="ignore"):
        y = energy_integrand(np.array([710.0, 1000.0, 1e4]))
    assert np.all(np.isfinite(y)) and np.all(y >= 0) and y[0] < 1e-290


def test_integrand_matches_textbook_form():
    x = np.linspace(0.01, 50.0, 500)
    assert np.allclose(energy_integrand(x), x**3 / np.expm1(x), rtol=1e-12)


# Simpson U(T) against the quadrature-free series for J(u), u = theta/T >= 1.
@pytest.mark.parametrize("T", [50.0, 100.0, 200.0, 343.0])
def test_energy_matches_series(T):
    assert debye_energy(T) == pytest.approx(debye_energy_series(T), rel=1e-11)


# Week 1's C_V integral (integrated by parts) is the derivative of U: C_V = 4U/T - 9R u/(e^u - 1).
@pytest.mark.parametrize("T", [30.0, 100.0, 343.0])
def test_week1_cv_is_dU_dT(T):
    assert debye_cv(T, N=4096) == pytest.approx(debye_cv_series(T), rel=1e-12)
    # D5 truncation -(h^4/30) U^(5) is ~1e-10 relative at h = 0.1 K (scales as h^4: checked)
    assert richardson5(debye_energy, T, 0.1) == pytest.approx(debye_cv_series(T), rel=1e-9)


def test_dcv_series():
    T = 100.0
    assert debye_dcv_series(T) == pytest.approx(richardson5(debye_cv_series, T, 0.1), rel=1e-10)


# Low T: U -> (3 pi^4 / 5) R T^4 / theta^3 (integral to infinity is pi^4/15).
def test_low_temperature_limit():
    T = 5.0
    assert debye_energy(T) == pytest.approx(3 * np.pi**4 / 5 * R * T**4 / THETA_CU**3, rel=1e-8)


# High T: U = 9RT [1/3 - u/8 + u^2/60 - u^4/5040 + u^6/272160 - ...] -> 3RT - 9R theta/8
# (x/(e^x - 1) = sum B_n x^n / n!, integrated term by term).
def test_high_temperature_series():
    T = 1500.0
    u = THETA_CU / T
    series = 9 * R * T * (1 / 3 - u / 8 + u**2 / 60 - u**4 / 5040 + u**6 / 272160)
    assert debye_energy(T) == pytest.approx(series, rel=1e-11)
