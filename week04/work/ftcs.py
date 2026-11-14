"""Week 4, Tasks 1-3: the explicit FTCS scheme for c_t = D c_xx (lecture notes §5.4-5.8).

    c_i^{n+1} = c_i^n + r (c_{i+1}^n - 2 c_i^n + c_{i-1}^n),   r = D dt / dx^2

Grid x_i = i dx (i = 0..M), t_n = n dt. The left end is the surface, c_0^n = Cs for n >= 1; the corner
c_0^0 is a separate choice (lecture notes §5.8). The right end is either Dirichlet (c_M = C0) or zero flux
through the ghost point c_{M+1} = c_{M-1}: c_M^{n+1} = c_M^n + 2 r (c_{M-1}^n - c_M^n).

Every step is computed from the whole row of old values (NumPy slices: the right-hand side is evaluated in
full before it is stored), so no new value is used inside the same step. No PDE solver is called.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

RIGHT_BCS = ('dirichlet', 'zero-flux')


def time_steps(r: float, dx: float, T: float, D: float = 1.0):
    """dt = r dx^2 / D and N = T / dt, which must be an integer: N = round(T/dt), asserting |N dt - T| < 1e-9 T.

    round, not int: 0.4 * 0.1**2 is 0.004000000000000001 in floating point, so 1/dt = 249.99999999999997.
    """
    dt = r * dx * dx / D
    N = round(T / dt)
    assert abs(N * dt - T) < 1e-9 * T, f'T/dt = {T / dt!r} is not an integer'
    return dt, N


def ftcs_step(c, r: float, right: str = 'dirichlet', c_right: float | None = None):
    """One FTCS step from the old row c; returns a new array.

    Interior and right end are updated; the left end keeps its old value, to be overwritten by the caller
    with the surface value of the new time level. Dirichlet keeps c_M (or sets it to c_right).
    """
    new = np.empty_like(c)
    new[1:-1] = c[1:-1] + r * (c[2:] - 2 * c[1:-1] + c[:-2])
    if right == 'dirichlet':
        new[-1] = c[-1] if c_right is None else c_right
    elif right == 'zero-flux':
        new[-1] = c[-1] + 2 * r * (c[-2] - c[-1])
    else:
        raise ValueError(f'right must be one of {RIGHT_BCS}, got {right!r}')
    new[0] = c[0]
    return new


def ftcs(M: int, L: float, r: float, T: float, cs: float = 1.0, c0: float = 0.0, right: str = 'dirichlet',
         corner: float | None = None, D: float = 1.0, snapshots=(), surface=None, callback=None):
    """FTCS for diffusion into [0, L] from a surface held at Cs, initial value C0.

    M intervals (dx = L/M), N = T/dt steps (see time_steps). corner is c_0^0 (default Cs).
    snapshots: times at which to keep the profile (each must be a whole number of steps).
    surface: optional function n -> c_0^n for n >= 1 (default: the constant Cs).
    callback(n, t_n, c): called after every step; returning True stops the run at that step.
    Returns x, c at the last step taken, and a dict {time: profile}.
    """
    dx = L / M
    dt, N = time_steps(r, dx, T, D)
    snap_at = {}
    for ts in snapshots:
        n = round(ts / dt)
        assert abs(n * dt - ts) <= 1e-9 * max(ts, dt), f'snapshot t = {ts} is not a whole number of steps'
        snap_at[n] = ts
    x = np.linspace(0.0, L, M + 1)
    c = np.full(M + 1, float(c0))
    c[0] = cs if corner is None else corner
    snaps = {snap_at[0]: c.copy()} if 0 in snap_at else {}
    for n in range(1, N + 1):
        c = ftcs_step(c, r, right, c0)
        c[0] = cs if surface is None else surface(n)
        if n in snap_at:
            snaps[snap_at[n]] = c.copy()
        if callback is not None and callback(n, n * dt, c):
            break
    return x, c, snaps


def hand_steps(r: float, steps: int = 3, n_points: int = 7):
    """Lecture notes §5.4 by hand: c_0 = 1 and c_{n-1} = 0 held fixed, the rest start at 0.
    Returns the full row after each step."""
    c = np.zeros(n_points)
    c[0] = 1.0
    rows = []
    for _ in range(steps):
        c = ftcs_step(c, r, 'dirichlet', 0.0)
        c[0] = 1.0
        rows.append(c.copy())
    return rows


def ftcs_periodic_step(c, r: float):
    """One FTCS step on a periodic grid (neighbours wrap around)."""
    return c + r * (np.roll(c, -1) - 2 * c + np.roll(c, 1))


def amplification(r, theta):
    """von Neumann amplification factor of FTCS, g(theta) = 1 - 4 r sin^2(theta/2) (lecture notes §5.7)."""
    return 1 - 4 * r * np.sin(np.asarray(theta, float) / 2) ** 2


def exact_decay(r, theta):
    """Decay of the wave exp(i j theta) over one step of the exact solution: exp(-D k^2 dt) = exp(-r theta^2)."""
    return np.exp(-r * np.asarray(theta, float) ** 2)


def fixed_ends_growth(r: float, M: int) -> float:
    """|g| of the most dangerous mode with both ends fixed (p = M - 1): |1 - 4 r sin^2((M-1) pi / (2M))|."""
    return abs(1 - 4 * r * np.sin((M - 1) * np.pi / (2 * M)) ** 2)


def r_limit(M: int) -> float:
    """Exact stability limit with both ends fixed and M - 1 interior points: 1 / (2 sin^2((M-1) pi / (2M)))."""
    return 1 / (2 * np.sin((M - 1) * np.pi / (2 * M)) ** 2)


def method_of_lines(M: int, L: float, T: float, D: float = 1.0, cs: float = 1.0, c0: float = 0.0):
    """The r -> 0 limit of FTCS: the semi-discrete system dc/dt = (D/dx^2)(A c + b) solved exactly in time.

    Surface c_0 = Cs, right end c_M = C0 (Dirichlet), interior starts at C0. A = tridiag(1, -2, 1) is symmetric,
    so with A = V diag(w) V^T: c(T) = c_ss + V exp(w D T/dx^2) V^T (c(0) - c_ss), where c_ss is the linear steady
    state. Only the spatial error is left (lecture notes §5.5). Returns x and c(T) including both ends.
    """
    dx = L / M
    n = M - 1
    w, V = eigh_tridiagonal(-2.0 * np.ones(n), np.ones(n - 1))
    x = np.linspace(0.0, L, M + 1)
    c_ss = cs + (c0 - cs) * x[1:-1] / L
    c = c_ss + V @ (np.exp(w * D * T / dx ** 2) * (V.T @ (c0 - c_ss)))
    return x, np.concatenate([[cs], c, [c0]])
