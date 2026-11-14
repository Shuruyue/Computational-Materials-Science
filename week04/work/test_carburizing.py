"""Task 4 tests: Wells-Mehl diffusivity, case-depth times and the FTCS runs in real units. Run: pytest -q"""
import numpy as np
import pytest

from carburizing import (C0, HOUR, case_time, grid_steps, single_stage, two_stage, two_stage_exact_time,
                         wells_mehl)
from diffusion import depth_interp, erfc_solution, two_stage_solution

D925 = wells_mehl(925)


def test_wells_mehl_lecture_values():
    assert D925 == pytest.approx(1.745e-11, rel=5e-4)
    assert wells_mehl(875) == pytest.approx(9.72e-12, rel=1e-3)
    assert wells_mehl(975) == pytest.approx(2.99e-11, rel=1e-3)
    assert wells_mehl(900) == pytest.approx(1.31e-11, rel=1e-3)   # hand-calculation exercise 5.9


def test_case_time_table_lecture_5_9():
    hours = [case_time(d, D925, 1.0) / HOUR for d in (0.5e-3, 1.0e-3, 1.5e-3)]
    assert hours == pytest.approx([1.50, 6.02, 13.54], abs=5e-3)
    assert case_time(1e-3, wells_mehl(875), 1.0) / HOUR == pytest.approx(10.80, abs=5e-3)
    assert case_time(1e-3, wells_mehl(975), 1.0) / HOUR == pytest.approx(3.51, abs=5e-3)
    # exercise 5.9: the answer 2.93 h uses D(950 C) = 2.296e-11 (the rounded 2.30e-11 would give 2.92 h)
    assert case_time(0.8e-3, wells_mehl(950), 1.0) / HOUR == pytest.approx(2.93, abs=5e-3)
    assert case_time(1.5e-3, D925, 1.0) / case_time(1e-3, D925, 1.0) == pytest.approx(2.25, rel=1e-14)


def test_grid_steps_keeps_r_just_below_r0():
    t = case_time(1e-3, D925, 1.0)
    dt, N, r = grid_steps(D925, t, 10e-6)
    assert N * dt == pytest.approx(t, rel=1e-15) and 0.3999 < r <= 0.4


def test_single_stage_matches_erfc():
    t = case_time(1e-3, D925, 1.0)
    s = single_stage(D925, 1.0, t, corner='mean')
    assert s['N'] == 9447 and s['dt'] == pytest.approx(2.29, abs=5e-3)
    assert np.abs(s['c'] - erfc_solution(s['x'], t, D925, 1.0, C0)).max() < 1e-5
    assert depth_interp(s['x'], s['c'], 0.40) == pytest.approx(1e-3, abs=1e-7)


def test_two_stage_against_superposition():
    tb = 3 * HOUR
    ts = two_stage(D925, 1.10, 0.80, tb)
    assert ts['Nb'] * ts['dt'] == pytest.approx(tb, rel=1e-15)
    exact = two_stage_exact_time(D925, 1.10, 0.80, tb)
    assert exact / HOUR == pytest.approx(6.373, abs=5e-4)
    assert 0 <= ts['t'] - exact <= ts['dt']               # the first step at or past the exact time
    ex = two_stage_solution(ts['x'], ts['t'], tb, D925, 1.10, 0.80, C0)
    assert np.abs(ts['c'] - ex).max() < 2e-5
    assert ts['c'][0] == 0.80
