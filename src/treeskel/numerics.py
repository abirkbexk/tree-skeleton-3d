"""Verified numerical building blocks.

Newton interpolation (Week 3), finite differences (Week 7), explicit ODE solvers (Week 7),
quadrature and arc length (Week 6) and ridge solvers (Week 2). Each is unit-tested against a known answer in ``tests/``."""
import numpy as np
from scipy import integrate, interpolate


def newton_coefficients(x, y):
    """Divided-difference coefficients c_k = f[x_0..x_k] of the Newton interpolating polynomial."""
    x = np.asarray(x, float); c = np.array(y, float)
    n = len(x)
    for j in range(1, n):
        c[j:] = (c[j:] - c[j - 1:n - 1]) / (x[j:] - x[:n - j])
    return c


def newton_eval(x_nodes, c, xq):
    """Evaluate Newton polynomial by nested (Horner-like) multiplication."""
    res = np.full(np.shape(xq), c[-1], dtype=float)
    for k in range(len(c) - 2, -1, -1):
        res = res * (xq - x_nodes[k]) + c[k]
    return res


def fd_first(f, h, kind="central"):
    """First derivative on a uniform grid. 'central': O(h^2) interior (2nd-order one-sided ends); 'forward': O(h)."""
    f = np.asarray(f, float); d = np.empty_like(f)
    if kind == "forward":
        d[:-1] = (f[1:] - f[:-1]) / h; d[-1] = (f[-1] - f[-2]) / h
    elif kind == "central":
        d[1:-1] = (f[2:] - f[:-2]) / (2 * h)
        d[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h); d[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)
    else:
        raise ValueError(kind)
    return d


def fd_second(f, h):
    """Second derivative, central O(h^2) (end points copy their neighbour)."""
    f = np.asarray(f, float); d = np.empty_like(f)
    d[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / h ** 2; d[0], d[-1] = d[1], d[-2]
    return d


def curvature(x, h=1.0):
    """Signed curvature of the curve (x(y), y)."""
    x1, x2 = fd_first(x, h), fd_second(x, h)
    return x2 / (1 + x1 ** 2) ** 1.5


def observed_order(hs, errs):
    """Empirical convergence order: slope of log(error) versus log(step size h) from a least-squares fit."""
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]


def solve_ode(f, x0, t0, t1, n, method="rk4"):
    """Fixed-step explicit solver for x' = f(t, x). method: euler | heun | rk4. Returns (t, x) arrays of length n+1."""
    h = (t1 - t0) / n; t, x = t0, x0; ts, xs = [t], [x]
    for _ in range(n):
        if method == "euler":
            x = x + h * f(t, x)
        elif method == "heun":
            k1 = f(t, x); k2 = f(t + h, x + h * k1); x = x + h * (k1 + k2) / 2
        elif method == "rk4":
            k1 = f(t, x); k2 = f(t + h / 2, x + h * k1 / 2); k3 = f(t + h / 2, x + h * k2 / 2); k4 = f(t + h, x + h * k3); x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(method)
        t += h; ts.append(t); xs.append(x)
    return np.array(ts), np.array(xs)


def ridge_normal_eq(A, B, lam):
    """Solve (AᵀA + λI) W = AᵀB by LU (np.linalg.solve) – squares the condition number."""
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=A.dtype), A.T @ B)


def ridge_svd(A, B, lam):
    """Ridge solution via thin SVD: W = V diag(σ/(σ²+λ)) Uᵀ B – numerically the safer route."""
    U_, s_, Vt_ = np.linalg.svd(A, full_matrices=False)
    return Vt_.T @ ((s_ / (s_ ** 2 + lam))[:, None] * (U_.T @ B))


def arc_length(t, P, method="simpson", n=None, smooth=True):
    """Length of the curve through points P(t) (m,k): L = ∫ ||dP/dt|| dt.

    The curve is a cubic *smoothing* spline per coordinate (smooth=True, GCV-chosen penalty; differentiation amplifies noise, so interpolating
    noisy points would inflate the length) or an interpolating CubicSpline (smooth=False).
    method: trapezoid | simpson | romberg | gauss | reference (adaptive QUADPACK)."""
    t = np.asarray(t, float)
    mk = (lambda y: interpolate.make_smoothing_spline(t, y)) if smooth else (lambda y: interpolate.CubicSpline(t, y))
    ders = [mk(P[:, k]).derivative() for k in range(P.shape[1])]
    f = lambda tt: np.linalg.norm(np.stack([d(tt) for d in ders], axis=-1), axis=-1)
    a, b = float(t[0]), float(t[-1]); n = n or len(t)
    if method == "reference":
        return integrate.quad(lambda x: float(f(x)), a, b, limit=500)[0]
    if method == "gauss":
        nd, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * (b - a) * np.sum(w * f(0.5 * (b - a) * nd + 0.5 * (b + a)))
    if method == "romberg":
        k = max(int(np.floor(np.log2(max(n - 1, 2)))), 1); tt = np.linspace(a, b, 2 ** k + 1); return integrate.romb(f(tt), dx=tt[1] - tt[0])
    if method == "simpson":
        n = n if n % 2 == 1 else n + 1; tt = np.linspace(a, b, n); return integrate.simpson(f(tt), x=tt)
    if method == "trapezoid":
        tt = np.linspace(a, b, n); return integrate.trapezoid(f(tt), tt)
    raise ValueError(method)
