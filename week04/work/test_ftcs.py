"""Tasks 1-3 tests: the FTCS update, its boundaries, stability and convergence. Run: pytest -q"""
import numpy as np
import pytest

from diffusion import erfc_solution
from ftcs import (amplification, exact_decay, fixed_ends_growth, ftcs, ftcs_periodic_step, ftcs_step, hand_steps,
                  method_of_lines, r_limit, time_steps)


def max_err(M, L, r, T, **kw):
    x, c, _ = ftcs(M, L, r, T, **kw)
    return np.abs(c - erfc_solution(x, T)).max()


def test_time_steps_uses_round_and_asserts_integer():
    dt, N = time_steps(0.4, 0.1, 1.0)
    assert int(1.0 / dt) == 249 and N == 250          # int() would lose one step (05, common errors)
    with pytest.raises(AssertionError):
        time_steps(0.45, 0.1, 1.0)                     # 1/0.0045 = 222.2...


def test_hand_table_lecture_5_4():
    expected = {0.25: [[0.25, 0, 0], [0.375, 0.0625, 0], [0.453125, 0.125, 0.015625]],
                0.5: [[0.5, 0, 0], [0.5, 0.25, 0], [0.625, 0.25, 0.125]],
                1.0: [[1, 0, 0], [0, 1, 0], [2, -1, 1]]}
    for r, rows in expected.items():
        got = hand_steps(r)
        for row, want in zip(got, rows):
            assert row[0] == 1 and row[-1] == 0
            assert np.array_equal(row[1:4], want)
            assert np.all(row[4:6] == 0)


def test_lecture_exercise_5_4():
    rows = hand_steps(0.4, steps=2)
    assert rows[1][1:3] == pytest.approx([0.48, 0.16], abs=1e-15)
    # zero-flux end: c_{M-1} = 0.2, c_M = 0.1, r = 0.4 -> 0.1 + 2 * 0.4 * (0.2 - 0.1) = 0.18
    c = np.array([1.0, 0.5, 0.2, 0.1])
    assert ftcs_step(c, 0.4, 'zero-flux')[-1] == pytest.approx(0.18, abs=1e-15)


def test_whole_row_update_equals_pointwise_loop_with_copy():
    rng = np.random.default_rng(0)
    c = rng.random(30)
    r = 0.37
    old = c.copy()
    loop = c.copy()
    for i in range(1, len(c) - 1):
        loop[i] = old[i] + r * (old[i + 1] - 2 * old[i] + old[i - 1])
    assert np.array_equal(ftcs_step(c, r)[1:-1], loop[1:-1])
    # overwriting in place uses the new c_{i-1}: a different method, no error raised
    bad = c.copy()
    for i in range(1, len(c) - 1):
        bad[i] = bad[i] + r * (bad[i + 1] - 2 * bad[i] + bad[i - 1])
    assert np.abs(bad - loop).max() > 1e-2


@pytest.mark.parametrize('r', [0.1, 0.25, 0.4, 0.5])
def test_discrete_maximum_principle(r):
    rng = np.random.default_rng(1)
    c = rng.random(41)
    lo, hi = c.min(), c.max()
    for _ in range(200):
        c = ftcs_step(c, r, 'zero-flux')
        assert lo - 1e-15 <= c.min() and c.max() <= hi + 1e-15


def test_r_above_half_breaks_the_bound():
    # neighbours 0, 1, 0 with r = 0.6: the new centre is -0.2 (lecture notes §5.6)
    assert ftcs_step(np.array([0.0, 1.0, 0.0]), 0.6)[1] == pytest.approx(-0.2, abs=1e-15)


def test_linear_profile_is_steady():
    x = np.linspace(0, 1, 11)
    c = 1 - x
    for _ in range(1000):
        c = ftcs_step(c, 0.45)
    assert np.abs(c - (1 - x)).max() < 1e-14


@pytest.mark.parametrize('r', [0.1, 0.4, 0.6])
def test_periodic_wave_is_multiplied_by_g(r):
    j = np.arange(64)
    for theta in np.pi * np.array([1 / 8, 1 / 4, 1 / 2, 3 / 4, 1]):
        c = np.cos(j * theta)
        new = ftcs_periodic_step(c, r)
        assert np.abs(new - amplification(r, theta) * c).max() < 1e-12


def test_amplification_table_lecture_5_7():
    th = np.pi * np.array([0.125, 0.25, 0.5, 0.75, 1])
    assert amplification(0.4, th) == pytest.approx([0.9391, 0.7657, 0.2, -0.3657, -0.6], abs=5e-5)
    assert amplification(0.6, th) == pytest.approx([0.9087, 0.6485, -0.2, -1.0485, -1.4], abs=5e-5)
    assert exact_decay(0.4, th) == pytest.approx([0.9402, 0.7813, 0.3727, 0.1085, 0.0193], abs=5e-5)
    # the theta^4 terms of g and exp(-r theta^2) agree only for r = 1/6
    th = 1e-2
    assert abs(amplification(1 / 6, th) - exact_decay(1 / 6, th)) < 1e-12
    assert abs(amplification(0.4, th) - exact_decay(0.4, th)) > 1e-10


def test_r_limit_with_fixed_ends():
    got = [r_limit(M) for M in (5, 10, 20, 50, 100, 1000)]
    assert got == pytest.approx([0.5528, 0.5125, 0.5031, 0.5005, 0.50012, 0.500001], abs=5e-5)
    assert fixed_ends_growth(r_limit(10), 10) == pytest.approx(1.0, abs=1e-15)


def test_task1_default_run():
    x, c, snaps = ftcs(100, 10, 0.4, 1.0, snapshots=(0.1, 0.5, 1.0))
    e = c - erfc_solution(x, 1.0)
    i = np.argmax(np.abs(e))
    assert x[i] == pytest.approx(1.1) and e[i] == pytest.approx(5.11e-4, abs=5e-7)
    assert c.min() == 0 and c.max() == 1
    assert np.array_equal(snaps[1.0], c) and set(snaps) == {0.1, 0.5, 1.0}


def test_scale_invariance_is_exact():
    errs = {max_err(round(L / dx), L, 0.4, T) for dx, T, L in ((0.1, 1, 10), (0.2, 4, 20), (0.05, 0.25, 5))}
    assert len(errs) == 1                              # identical, not just close


def test_r_half_wastes_every_other_point():
    xa, ca, _ = ftcs(100, 10, 0.5, 1.0)
    xb, cb, _ = ftcs(50, 10, 0.25, 1.0)
    assert np.abs(ca[::2] - cb).max() < 1e-15


def test_dirichlet_steady_state_and_divergence_on_small_grid():
    # domain [0, 1], c(0) = 1, c(1) = 0: M = 10 is stable at r = 0.51 because r_limit(10) = 0.5125
    for M, r, stable in ((10, 0.51, True), (10, 0.515, False)):
        x, c, _ = ftcs(M, 1.0, r, 3000 * r / M ** 2)
        assert (np.abs(c - (1 - x)).max() < 1e-12) == stable


def test_second_order_at_r_04():
    e1, e2 = max_err(100, 10, 0.4, 1.0), max_err(200, 10, 0.4, 1.0)
    assert 1.99 < np.log2(e1 / e2) < 2.01


def test_fourth_order_needs_r_one_sixth_and_the_mean_corner():
    e = [max_err(round(10 / dx), 10, 1 / 6, 1.0, corner=0.5) for dx in (0.2, 0.1)]
    assert 3.99 < np.log2(e[0] / e[1]) < 4.01
    e = [max_err(round(10 / dx), 10, 1 / 6, 1.0) for dx in (0.2, 0.1)]
    assert 1.99 < np.log2(e[0] / e[1]) < 2.01


def test_corner_cs_is_half_a_step_early():
    r = 1 / 6
    dt = r * 0.01
    x, c, _ = ftcs(100, 10, r, 1.0)
    assert np.abs(c - erfc_solution(x, 1.0)).max() > 2e-4
    assert np.abs(c - erfc_solution(x, 1.0 + dt / 2)).max() < 1e-7


def test_method_of_lines_is_the_small_r_limit():
    xm, cm = method_of_lines(100, 10.0, 1.0)
    assert cm[0] == 1 and cm[-1] == 0
    gaps = []
    for N in (4000, 16000):
        x, c, _ = ftcs(100, 10.0, 100 / N, 1.0, corner=0.5)
        gaps.append(np.abs(c - cm).max())
    assert gaps[1] < gaps[0] / 3.5                     # the time error shrinks in proportion to r
    # long times: the linear steady state
    xs, cs_ = method_of_lines(20, 1.0, 50.0)
    assert np.abs(cs_ - (1 - xs)).max() < 1e-12


def test_callback_stops_the_run():
    seen = []
    x, c, _ = ftcs(20, 2.0, 0.4, 1.0, callback=lambda n, t, c: seen.append(n) or n == 7)
    assert seen == list(range(1, 8))


def test_bad_right_boundary():
    with pytest.raises(ValueError):
        ftcs_step(np.zeros(5), 0.4, 'neumann')
