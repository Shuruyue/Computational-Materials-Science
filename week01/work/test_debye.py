"""Tests for the Debye integrand traps (Task 3) and the C_V limits (Task 4). Run: pytest -q"""
import numpy as np
import pytest

from debye import (R, debye_cv, debye_high_t, debye_integrand,
                   debye_integrand_naive, debye_t3_law)


# Trap 1: x -> 0 is 0/0 in the textbook form; the safe form returns the limit x^2.
def test_integrand_at_zero():
    with np.errstate(all="ignore"):
        assert np.isnan(debye_integrand_naive(np.array([0.0]))[0])
    x = np.array([0.0, 1e-12, 1e-9, 1e-6, 1e-3])
    assert np.allclose(debye_integrand(x), x**2 * (1 - x**2 / 12), rtol=1e-12, atol=0)


# Trap 2: e^x overflows for x > ~709; the safe form stays finite and tends to 0.
def test_integrand_no_overflow():
    x = np.array([710.0, 1000.0, 1e4])
    with np.errstate(all="ignore"):
        assert np.all(np.isnan(debye_integrand_naive(x)))
    with np.errstate(all="raise", under="ignore"):
        y = debye_integrand(x)
    assert np.all(np.isfinite(y)) and np.all(y >= 0) and y[0] < 1e-290


# Away from both traps the two forms are the same function.
def test_integrand_matches_textbook_form():
    x = np.linspace(0.01, 50.0, 500)
    assert np.allclose(debye_integrand(x), debye_integrand_naive(x), rtol=1e-12)


# High-T limit: C_V -> 3R [1 - (theta/T)^2 / 20] (Dulong-Petit plus first correction).
def test_high_temperature_limit():
    T = 1500.0
    assert debye_cv(T) == pytest.approx(debye_high_t(T), rel=1e-5)
    assert debye_cv(T) / (3 * R) == pytest.approx(1.0, abs=3e-3)
    # The 5e-6 gap left above is the next series term u^4/560, not numerical error.
    for T in (700.0, 1000.0, 1500.0):
        assert debye_cv(T) == pytest.approx(debye_high_t(T, n_terms=3), rel=1e-8)


# Low-T limit: C_V -> (12 pi^4 / 5) R (T/theta)^3.
def test_low_temperature_t3_law():
    assert debye_cv(5.0) / debye_t3_law(5.0) == pytest.approx(1.0, abs=1e-6)
