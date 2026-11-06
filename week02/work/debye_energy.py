"""Week 2, Task 3: Debye lattice energy U(T) of Cu and C_V = dU/dT.

U(T) = 9 R T (T/theta_D)^3 * integral_0^{theta_D/T} x^3 / (e^x - 1) dx
(thermal part only; the zero-point energy is a constant and drops out of dU/dT).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Week 1's Simpson rule and Debye C_V are reused, not copied (the assignment allows reusing them directly).
WEEK01 = Path(__file__).resolve().parents[2] / 'week01' / 'work'
if str(WEEK01) not in sys.path:
    sys.path.append(str(WEEK01))
from debye import R, THETA_CU, debye_cv  # noqa: E402,F401  (debye_cv is re-exported for Task 3)
from quadrature import simpson  # noqa: E402

# Below this x the integrand is replaced by its series x^2 (1 - x/2); the next term
# x^2 * x^2/12 is then smaller than double-precision round-off.
_X_SMALL = 1e-8


def energy_integrand(x):
    """x^3 / (e^x - 1) in the overflow-free form x^3 e^-x / (1 - e^-x).

    x -> 0 (0/0): series x^2 (1 - x/2) for x < 1e-8; above that 1 - e^-x = -expm1(-x)
        keeps full precision.
    large x (overflow): only e^-x is formed, which underflows harmlessly to 0.
    """
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x < _X_SMALL
    out[small] = x[small] ** 2 * (1 - x[small] / 2)
    xs = x[~small]
    out[~small] = xs**3 * np.exp(-xs) / -np.expm1(-xs)
    return out


def debye_energy(T: float, theta: float = THETA_CU, N: int = 2048) -> float:
    """Thermal Debye energy U(T) in J/mol, integral done with Simpson on N cells."""
    xD = theta / T
    return 9 * R * T * (T / theta) ** 3 * simpson(energy_integrand, 0.0, xD, N)


# ---------------------------------------------------------------------------------------
# Independent check without quadrature. Expanding 1/(e^x - 1) = sum_k e^{-kx} and integrating
# term by term,
#     J(u) = integral_0^u x^3/(e^x - 1) dx = pi^4/15 - sum_k e^{-ku} (u^3/k + 3u^2/k^2 + 6u/k^3 + 6/k^4).
# The sum converges like e^{-ku}; for u >~ 1 (T <~ theta_D) pi^4/15 - J does not cancel badly.
# Everything below accepts complex T, so the complex step of Task 2 can differentiate it.

def _J(u, kmax: int = 60):
    k = np.arange(1, kmax + 1)
    u = np.asarray(u)[..., None]
    tail = np.exp(-k * u) * (u**3 / k + 3 * u**2 / k**2 + 6 * u / k**3 + 6 / k**4)
    return np.pi**4 / 15 - tail.sum(axis=-1)


def debye_energy_series(T, theta: float = THETA_CU):
    """U(T) = 9 R T^4 / theta^3 * J(theta/T) from the series above."""
    return 9 * R * T**4 / theta**3 * _J(theta / T)


def debye_cv_series(T, theta: float = THETA_CU):
    """C_V = dU/dT = 4U/T - 9R u/(e^u - 1), u = theta/T: the first term from T^4, the second from
    the moving upper limit (Leibniz rule). Equal to week 1's integral after integrating it by parts."""
    u = theta / T
    return 4 * debye_energy_series(T, theta) / T - 9 * R * u / (np.exp(u) - 1)


def debye_dcv_series(T, theta: float = THETA_CU):
    """dC_V/dT = U'' = 3 C_V/T - (9R/theta) u^3 e^u / (e^u - 1)^2.
    From C_V = 9R (T/theta)^3 K(u) with dK/du = u^4 e^u / (e^u - 1)^2."""
    u = theta / T
    return 3 * debye_cv_series(T, theta) / T - 9 * R / theta * u**3 * np.exp(u) / (np.exp(u) - 1) ** 2
