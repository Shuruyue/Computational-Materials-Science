"""Week 1, Tasks 3-4: Debye lattice heat capacity.

C_V(T) = 9 R (T/theta_D)^3 * integral_0^{theta_D/T} x^4 e^x / (e^x - 1)^2 dx
"""
from __future__ import annotations

import numpy as np

from quadrature import simpson

R = 8.314462618          # J mol^-1 K^-1, gas constant
THETA_CU = 343.0         # K, Debye temperature of Cu
GAMMA_CU = 0.695e-3      # J mol^-1 K^-2, electronic heat-capacity coefficient of Cu

# Below this x the integrand is replaced by its limit x^2; the next term of the
# series x^2 (1 - x^2/12 + ...) is then smaller than double-precision round-off.
_X_SMALL = 1e-8


def debye_integrand_naive(x):
    """Textbook form x^4 e^x / (e^x - 1)^2. Kept only to show the two traps:
    0/0 = nan at x = 0, and inf/inf = nan once e^x overflows (x > ~709)."""
    x = np.asarray(x, float)
    return x**4 * np.exp(x) / (np.exp(x) - 1) ** 2


def debye_integrand(x):
    """Overflow-free form x^4 e^-x / (1 - e^-x)^2.

    Trap 1 (x -> 0, 0/0): use the limit x^2 for x < 1e-8; above that,
        1 - e^-x = -expm1(-x) keeps full precision with no cancellation.
    Trap 2 (large x, overflow): only e^-x is formed, which underflows
        harmlessly to 0 instead of overflowing to inf.
    """
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x < _X_SMALL
    out[small] = x[small] ** 2
    xs = x[~small]
    out[~small] = xs**4 * np.exp(-xs) / np.expm1(-xs) ** 2
    return out


def debye_cv(T: float, theta: float = THETA_CU, N: int = 2048, rule=simpson) -> float:
    """Debye C_V in J mol^-1 K^-1 at temperature T (K), integral done with `rule` on N cells."""
    xD = theta / T
    return 9 * R * (T / theta) ** 3 * rule(debye_integrand, 0.0, xD, N)


def debye_t3_law(T, theta: float = THETA_CU):
    """Low-temperature limit (12 pi^4 / 5) R (T/theta)^3."""
    return 12 * np.pi**4 / 5 * R * (np.asarray(T, float) / theta) ** 3


# C_V/3R = 1 - u^2/20 + u^4/560 - u^6/18144 + ..., u = theta/T. From x^4 e^x/(e^x-1)^2
# = x^2 (1 - x^2/12 + x^4/240 - x^6/6048 + ...) integrated term by term.
_HIGH_T_COEFFS = (-1 / 20, 1 / 560, -1 / 18144)


def debye_high_t(T, theta: float = THETA_CU, n_terms: int = 1):
    """High-temperature expansion of C_V. n_terms = 1 gives 3R [1 - (theta/T)^2 / 20];
    n_terms = 2, 3 add the u^4/560 and -u^6/18144 corrections."""
    u2 = (theta / np.asarray(T, float)) ** 2
    return 3 * R * (1 + sum(c * u2 ** (k + 1) for k, c in enumerate(_HIGH_T_COEFFS[:n_terms])))
