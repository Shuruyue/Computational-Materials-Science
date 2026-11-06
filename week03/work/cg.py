"""Week 3, Task 2: conjugate gradients for the 2D Poisson problem (lecture notes §4.5).

-lap(u) = f on the unit square, u = 0 on the boundary, m x m interior points, h = 1/(m+1).
Manufactured solution u = g(x) g(y), g(x) = x(1 - x) e^x.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dstn


def lap(U, h):
    """Matrix-free 5-point -laplacian of an m x m array, values outside the array taken as 0:
    (4 U_ij - U_{i-1,j} - U_{i+1,j} - U_{i,j-1} - U_{i,j+1}) / h^2."""
    out = 4.0 * U
    out[1:, :] -= U[:-1, :]
    out[:-1, :] -= U[1:, :]
    out[:, 1:] -= U[:, :-1]
    out[:, :-1] -= U[:, 1:]
    return out / h**2


def cg(b, h, tol, maxiter=100_000):
    """Conjugate gradients for lap(x, h) = b from x0 = 0 (LeVeque p. 87):
        w = A p,  alpha = r.r / p.w,  x += alpha p,  r -= alpha w,  beta = r_new.r_new / r.r,  p = r + beta p.
    Stops when ||r||_2 / ||b||_2 < tol. Returns (x, iterations, relative residual after each
    iteration, starting with 1 for x0 = 0)."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = np.vdot(r, r)
    bnorm = np.sqrt(np.vdot(b, b))
    hist = [1.0]
    k = 0
    while hist[-1] >= tol and k < maxiter:
        w = lap(p, h)
        alpha = rr / np.vdot(p, w)
        x += alpha * p
        r -= alpha * w
        rr_new = np.vdot(r, r)
        p = r + (rr_new / rr) * p
        rr = rr_new
        k += 1
        hist.append(np.sqrt(rr) / bnorm)
    return x, k, np.array(hist)


def sd(b, h, tol, maxiter=1_000_000):
    """Steepest descent for lap(x, h) = b from x0 = 0: step along the residual r with
    alpha = r.r / r.Ar (exact line minimum). Same stopping rule and return values as cg."""
    x = np.zeros_like(b)
    r = b.copy()
    rr = np.vdot(r, r)
    bnorm = np.sqrt(np.vdot(b, b))
    hist = [1.0]
    k = 0
    while hist[-1] >= tol and k < maxiter:
        w = lap(r, h)
        alpha = rr / np.vdot(r, w)
        x += alpha * r
        r -= alpha * w
        rr = np.vdot(r, r)
        k += 1
        hist.append(np.sqrt(rr) / bnorm)
    return x, k, np.array(hist)


def poisson_dst(F, h):
    """Exact solution of the discrete problem lap(U, h) = F (to round-off), for comparison only.
    The 5-point Dirichlet laplacian is diagonalised by the type-I sine transform: eigenvectors
    sin(p pi x_i) sin(q pi y_j), eigenvalues (4/h^2)[sin^2(p pi h/2) + sin^2(q pi h/2)]."""
    m = F.shape[0]
    s2 = np.sin(np.arange(1, m + 1) * np.pi * h / 2) ** 2
    lam = 4 / h**2 * (s2[:, None] + s2[None, :])
    return dstn(dstn(F, type=1, norm='ortho') / lam, type=1, norm='ortho')


def grid(m: int):
    """Interior grid of the unit square: h = 1/(m+1), X[i, j] = x_i, Y[i, j] = y_j."""
    h = 1.0 / (m + 1)
    x = h * np.arange(1, m + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return h, X, Y


def g(x):
    """g(x) = x(1 - x) e^x."""
    return x * (1 - x) * np.exp(x)


def g2(x):
    """g''(x) = -x(x + 3) e^x."""
    return -x * (x + 3) * np.exp(x)


def g4(x):
    """g''''(x) = -(x^2 + 7x + 8) e^x."""
    return -(x * x + 7 * x + 8) * np.exp(x)


def u_exact(X, Y):
    return g(X) * g(Y)


def f_rhs(X, Y):
    """f = -lap(u) = x(x + 3) e^x g(y) + g(x) y(y + 3) e^y."""
    return -g2(X) * g(Y) - g(X) * g2(Y)


def error_source(X, Y):
    """Source of the leading discretisation error. The 5-point operator is
    -lap(u) - (h^2/12)(u_xxxx + u_yyyy) + O(h^4), so U - u = h^2 e + O(h^4) with
    -lap(e) = (u_xxxx + u_yyyy)/12, e = 0 on the boundary."""
    return (g4(X) * g(Y) + g(X) * g4(Y)) / 12


def kappa(h):
    """Condition number of the 2D 5-point matrix: lambda_max/lambda_min = cot^2(pi h/2)."""
    return 1 / np.tan(np.pi * h / 2) ** 2
