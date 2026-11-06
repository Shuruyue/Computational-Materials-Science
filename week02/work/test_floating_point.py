"""Task 0 acceptance tests: machine epsilon, float spacing, catastrophic cancellation. Run: pytest -q"""
from decimal import Decimal
from math import factorial

import numpy as np
import pytest

from floating_point import (machine_epsilon, one_minus_cos_direct, one_minus_cos_stable,
                            small_root_direct, small_root_stable)


# Halving loop: e equals np.finfo eps exactly, after 52 halvings (23 in single precision).
@pytest.mark.parametrize("dtype, n_bits", [(np.float64, 52), (np.float32, 23)])
def test_machine_epsilon(dtype, n_bits):
    e, n = machine_epsilon(dtype)
    assert e == np.finfo(dtype).eps
    assert n == n_bits
    assert e == dtype(2.0) ** -n_bits


# Spacing grows with |x|: spacing(x) = 2^(floor(log2 x) - 52), between eps|x|/2 and eps|x|.
@pytest.mark.parametrize("x", [1.0, 1e3, 1e6, 1e16])
def test_spacing(x):
    eps = np.finfo(float).eps
    assert np.spacing(x) == 2.0 ** (np.floor(np.log2(x)) - 52)
    assert eps * x / 2 <= np.spacing(x) <= eps * x


def test_1e16():
    # Spacing near 1e16 is 2: +1 lands half-way and rounds back (ties to even), +2 is representable.
    assert 1e16 + 1 == 1e16
    assert 1e16 + 2 != 1e16


def test_decimal_one_tenth():
    assert 0.1 + 0.2 != 0.3
    assert repr(0.1 + 0.2) == '0.30000000000000004'
    assert str(Decimal(0.1)) == '0.1000000000000000055511151231257827021181583404541015625'
    total = 0.0
    for _ in range(10):
        total += 0.1
    assert repr(total) == '0.9999999999999999'


# (1 - cos x)/x^2: at x = 1e-8, cos x = 1 - 5e-17 rounds to exactly 1, so the direct form gives 0.
def test_one_minus_cos_at_1e_8():
    assert one_minus_cos_direct(1e-8) == 0.0
    assert one_minus_cos_stable(1e-8) == 0.5


def test_one_minus_cos_stable_is_exact():
    # Series (1 - cos x)/x^2 = sum_k (-1)^(k+1) x^(2k-2) / (2k)!  = 1/2 - x^2/24 + x^4/720 - ...
    x = 10.0 ** -np.arange(1, 10)
    series = sum((-1) ** (k + 1) * x ** (2 * k - 2) / factorial(2 * k) for k in range(10, 0, -1))
    assert np.allclose(one_minus_cos_stable(x), series, rtol=4 * np.finfo(float).eps, atol=0)


def test_one_minus_cos_direct_error_grows():
    x = 10.0 ** -np.array([2, 4, 6, 8])
    rel = abs(one_minus_cos_direct(x) / one_minus_cos_stable(x) - 1)
    assert np.all(np.diff(rel) > 0)
    assert rel[-1] == 1.0


# x^2 - 1e8 x + 1 = 0. b^2 - 4 rounds to 9999999999999996, its root to the float just below 1e8,
# so b - sqrt(...) is exactly one spacing of 1e8 (2^-26) and the "small root" is 2^-27.
def test_small_root():
    direct = small_root_direct(1e8, 1.0)
    stable = small_root_stable(1e8, 1.0)
    assert direct == 2.0**-27
    assert stable == pytest.approx(1e-8, rel=2.3e-16)   # exact root 1e-8 (1 + 1e-16 + ...)
    assert abs(direct - stable) / stable == pytest.approx(0.255, abs=1e-3)
