"""Week 4, Task 4: gas carburizing of steel with FTCS in real units (lecture notes §5.9).

Data: base carbon C0 = 0.20 wt%, effective case depth = depth of 0.40 wt% C,
diffusion coefficient of carbon in austenite (Wells and Mehl 1940): D = 0.12 exp(-32000 / (1.987 T)) cm^2/s.
Lengths in m, times in s, concentrations in wt% C.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcinv

from diffusion import depth_interp, two_stage_solution
from ftcs import ftcs

C0 = 0.20           # wt% C of the steel
CASE = 0.40         # wt% C that defines the effective case depth
D0_CM2 = 0.12       # cm^2/s
Q_CAL = 32000.0     # cal/mol
R_CAL = 1.987       # cal/(mol K)
HOUR = 3600.0


def wells_mehl(T_celsius):
    """D in m^2/s at T in degrees Celsius (the Wells-Mehl value is in cm^2/s: times 1e-4)."""
    T = np.asarray(T_celsius, float) + 273.15
    return D0_CM2 * np.exp(-Q_CAL / (R_CAL * T)) * 1e-4


def case_time(depth: float, D: float, cs: float, c0: float = C0, level: float = CASE):
    """Time for the level to reach `depth` with a constant surface Cs (erfc solution):
    theta = (level - C0)/(Cs - C0), t = depth^2 / (4 erfcinv(theta)^2 D)."""
    eta = erfcinv((level - c0) / (cs - c0))
    return depth ** 2 / (4 * eta ** 2 * D)


def grid_steps(D: float, t: float, dx: float, r0: float = 0.4):
    """Steps for time t with r close to r0: N = ceil(t / dt0), dt0 = r0 dx^2/D; then dt = t/N (r slightly below r0).
    Returns dt, N, r."""
    N = math.ceil(t / (r0 * dx * dx / D))
    dt = t / N
    return dt, N, D * dt / (dx * dx)


def single_stage(D: float, cs: float, t: float, L: float = 6e-3, dx: float = 10e-6, r0: float = 0.4,
                 corner: str = 'cs', snapshots=()):
    """FTCS with a constant surface Cs up to time t. corner 'cs' (c_0^0 = Cs) or 'mean' ((Cs + C0)/2)."""
    dt, N, r = grid_steps(D, t, dx, r0)
    M = round(L / dx)
    cval = {'cs': cs, 'mean': (cs + C0) / 2}[corner]
    x, c, snaps = ftcs(M, L, r, t, cs, C0, 'dirichlet', cval, D, snapshots=snapshots)
    return dict(x=x, c=c, snaps=snaps, dt=dt, N=N, r=r)


def two_stage(D: float, cb: float, cd: float, tb: float, target: float = 1e-3, L: float = 6e-3,
              dx: float = 10e-6, r0: float = 0.4, t_max: float = 48 * HOUR):
    """Boost at Cb for tb, then diffuse at Cd, until the case depth first reaches `target`.

    dt divides tb exactly (dt = tb / ceil(tb/dt0)). The corner c_0^0 is (Cb + C0)/2 and the surface value
    at the switching step t = tb is (Cb + Cd)/2, both for the reason of lecture notes §5.8.
    Returns the step and time at which the target is reached, the profile then and at the end of the boost.
    """
    dt, Nb, r = grid_steps(D, tb, dx, r0)
    M = round(L / dx)
    N_max = math.ceil(t_max / dt)

    def surface(n):
        return cb if n < Nb else (cb + cd) / 2 if n == Nb else cd

    hit = {}

    def reached(n, t, c):
        if n > Nb and depth_interp(x_grid, c, CASE) >= target:
            hit.update(n=n, t=t)
            return True
        return False

    x_grid = np.linspace(0.0, L, M + 1)
    x, c, snaps = ftcs(M, L, r, N_max * dt, cb, C0, 'dirichlet', (cb + C0) / 2, D,
                       snapshots=(Nb * dt,), surface=surface, callback=reached)
    if not hit:
        raise RuntimeError('case depth not reached before t_max')
    return dict(x=x, c=c, boost=snaps[Nb * dt], dt=dt, r=r, Nb=Nb, n=hit['n'], t=hit['t'])


def two_stage_exact_time(D: float, cb: float, cd: float, tb: float, target: float = 1e-3):
    """Total time at which the superposition solution reaches CASE at `target` (brentq on c(target, t) - CASE)."""
    f = lambda t: two_stage_solution(target, t, tb, D, cb, cd, C0) - CASE
    return brentq(f, tb * (1 + 1e-12), tb + 100 * HOUR, xtol=1e-9, rtol=1e-14)
