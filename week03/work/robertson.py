"""Week 3, Task 4: the Robertson reaction, a classic stiff problem (lecture notes §4.10).

    A -> B          k1 = 0.04
    B + B -> C + B  k2 = 3e7
    B + C -> A + C  k3 = 1e4

y1' = -k1 y1 + k3 y2 y3,  y2' = k1 y1 - k3 y2 y3 - k2 y2^2,  y3' = k2 y2^2,  y(0) = (1, 0, 0).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

K1, K2, K3 = 0.04, 3e7, 1e4
Y0 = np.array([1.0, 0.0, 0.0])
RK4_LIMIT = 2.785293563405282            # real root of z^3 + 4z^2 + 12z + 24 (ode.rk4_real_limit)

# Reference values quoted in the assignment
LIT_T40 = np.array([0.7158270687, 9.185534764e-6, 0.2841637457])
TESTSET_T1E11 = np.array([2.083340149701255e-8, 8.333360770334713e-14, 0.9999999791665050])


def rhs(t, y):
    """Mass-action rates; every reaction term appears once with + and once with -."""
    y1, y2, y3 = y
    r1, r2, r3 = K1 * y1, K2 * y2 * y2, K3 * y2 * y3
    return np.array([-r1 + r3, r1 - r3 - r2, r2])


def jac(t, y):
    """Analytic Jacobian (lecture notes §4.10); every column sums to 0 (mass conservation)."""
    y1, y2, y3 = y
    return np.array([[-K1, K3 * y3, K3 * y2],
                     [K1, -K3 * y3 - 2 * K2 * y2, -K3 * y2],
                     [0.0, 2 * K2 * y2, 0.0]])


def fast_slow_eigenvalues(y):
    """The two nonzero eigenvalues of J, from its characteristic polynomial.

    Columns of J sum to 0, so one eigenvalue is exactly 0 and the other two solve
    lambda^2 - T lambda + M2 = 0 with T = trace J = -(k1 + k3 y3 + 2 k2 y2) and
    M2 = sum of the principal 2x2 minors = 2 k2 y2 (k1 + k3 y2). lambda_fast = (T - sqrt(T^2 - 4 M2))/2
    adds two negative numbers; lambda_slow = M2 / lambda_fast avoids the cancellation of the other root.
    Works on arrays: y has shape (3, ...).
    """
    y = np.asarray(y, float)
    y2, y3 = y[1], y[2]
    T = -(K1 + K3 * y3 + 2 * K2 * y2)
    M2 = 2 * K2 * y2 * (K1 + K3 * y2)
    lam_fast = (T - np.sqrt(T * T - 4 * M2)) / 2
    return lam_fast, M2 / lam_fast


def solve(method, t_end, rtol, atol, t_eval=None, dense_output=False, use_jac=True, **kw):
    """solve_ivp on Robertson from y(0) = (1, 0, 0); the analytic Jacobian is passed to the
    implicit methods."""
    if use_jac and method in ('Radau', 'BDF', 'LSODA'):
        kw['jac'] = jac
    return solve_ivp(rhs, (0.0, t_end), Y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval,
                     dense_output=dense_output, **kw)


def rk4_fixed(N: int, T: float = 40.0, blowup: float = 10.0):
    """Classical RK4 with constant h = T/N, on Python floats for speed.

    Stops at the first step where some |y_i| > blowup or a value is not finite. Returns a dict:
    y (last state), t (its time), blew_up, t_neg (first time y2 < 0, or None), mass_err_neg
    (y1 + y2 + y3 - 1 at that moment).
    """
    h = T / N

    def f(y1, y2, y3):
        r1, r2, r3 = K1 * y1, K2 * y2 * y2, K3 * y2 * y3
        return -r1 + r3, r1 - r3 - r2, r2

    y1, y2, y3 = 1.0, 0.0, 0.0
    t_neg = mass_err_neg = None
    blew_up = False
    n = 0
    for n in range(1, N + 1):
        a1, a2, a3 = f(y1, y2, y3)
        b1, b2, b3 = f(y1 + h / 2 * a1, y2 + h / 2 * a2, y3 + h / 2 * a3)
        c1, c2, c3 = f(y1 + h / 2 * b1, y2 + h / 2 * b2, y3 + h / 2 * b3)
        d1, d2, d3 = f(y1 + h * c1, y2 + h * c2, y3 + h * c3)
        y1 += h / 6 * (a1 + 2 * b1 + 2 * c1 + d1)
        y2 += h / 6 * (a2 + 2 * b2 + 2 * c2 + d2)
        y3 += h / 6 * (a3 + 2 * b3 + 2 * c3 + d3)
        if t_neg is None and y2 < 0:
            t_neg, mass_err_neg = n * h, y1 + y2 + y3 - 1
        if not all(np.isfinite((y1, y2, y3))) or max(abs(y1), abs(y2), abs(y3)) > blowup:
            blew_up = True
            break
    return dict(y=np.array([y1, y2, y3]), t=n * h, blew_up=blew_up, t_neg=t_neg, mass_err_neg=mass_err_neg)


def log_grid(M: int, t_end: float = 4e10) -> np.ndarray:
    """t_0 = 0, t_n = 4e-7 * 10^((n-1)/M), n = 1, 2, ... up to t_end (a grid point: 4 * 10^integer).
    Written as 4 * 10^((n-1)/M - 7) so that t = 40 and t = 4e10 come out exactly
    (4e-7 itself is not exact in binary)."""
    n_steps = int(round(M * np.log10(t_end / 4e-7))) + 1
    return np.array([0.0] + [4.0 * 10.0 ** ((n - 1) / M - 7) for n in range(1, n_steps + 1)])


def backward_euler_newton(M: int, t_end: float = 4e10, tol: float = 1e-10, maxit: int = 50):
    """Backward Euler on log_grid(M). Each step solves G(z) = z - Y_n - h f(z) = 0 by Newton:
    (I - h J(z)) Delta = -G(z), z <- z + Delta, from z = Y_n, until max_i |Delta_i|/|z_i| < tol
    (a component with Delta_i = 0 counts as converged).
    Returns (t, Y with shape (len(t), 3), Newton iterations of every step)."""
    t = log_grid(M, t_end)
    Y = np.empty((len(t), 3))
    Y[0] = Y0
    iters = np.empty(len(t) - 1, dtype=int)
    I = np.eye(3)
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        z = Y[n].copy()
        for k in range(1, maxit + 1):
            G = z - Y[n] - h * rhs(t[n + 1], z)
            delta = np.linalg.solve(I - h * jac(t[n + 1], z), -G)
            z = z + delta
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(delta == 0, 0.0, abs(delta) / abs(z))
            if rel.max() < tol:
                break
        Y[n + 1] = z
        iters[n] = k
    return t, Y, iters
