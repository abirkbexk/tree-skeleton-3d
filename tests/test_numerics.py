import numpy as np
import pytest
from scipy import integrate
from sklearn.linear_model import Ridge

from treeskel import experiments as ex
from treeskel.numerics import (arc_length, curvature, fd_first, fd_second, newton_coefficients, newton_eval, observed_order, ridge_normal_eq,
                               ridge_svd, solve_ode)


def test_newton_interpolation_exact_for_cubic():
    xn = np.array([0.0, 1.0, 2.5, 4.0]); p = lambda x: 2 * x ** 3 - x ** 2 + 3
    c = newton_coefficients(xn, p(xn)); xq = np.linspace(-1, 5, 13)
    assert np.allclose(newton_eval(xn, c, xq), p(xq)) and np.allclose(newton_eval(xn, c, xn), p(xn))


def test_finite_difference_orders():
    order = ex.verify_finite_differences()
    assert abs(order["forward"] - 1) < 0.2 and abs(order["central"] - 2) < 0.2
    x = np.linspace(0, 1, 101); assert np.allclose(fd_second(x ** 2, x[1] - x[0])[1:-1], 2.0)


def test_curvature_of_circle_arc():
    th = np.linspace(0.2, 1.0, 400); y = th; x = np.sqrt(4 - y ** 2)      # circle radius 2 -> |kappa| = 0.5
    assert np.allclose(np.abs(curvature(x, h=y[1] - y[0]))[5:-5], 0.5, atol=5e-3)


def test_quadrature_known_integral():
    t = np.linspace(0, np.pi, 33); f = np.sin
    assert abs(integrate.trapezoid(f(t), t) - 2) < 3e-3 and abs(integrate.simpson(f(t), x=t) - 2) < 5e-6 and abs(integrate.romb(f(t), dx=t[1] - t[0]) - 2) < 1e-9
    nd, w = np.polynomial.legendre.leggauss(8); assert abs(np.pi / 2 * np.sum(w * np.sin(np.pi / 2 * (nd + 1))) - 2) < 1e-12


@pytest.mark.parametrize("method", ["trapezoid", "simpson", "romberg", "gauss", "reference"])
def test_arc_length_of_straight_3d_line(method):
    t = np.arange(0, 20.0); P = np.column_stack([0.01 * t, 0.02 * t, 2.0 + 0.0 * t]); L = np.hypot(0.01, 0.02) * 19
    assert abs(arc_length(t, P, method, smooth=False) - L) < 1e-9


def test_smoothing_spline_arc_length_is_not_inflated_by_noise():
    rng = np.random.default_rng(1); t = np.arange(0, 100.0); P = np.column_stack([0.01 * t, 0.0 * t, 2.0 + 0.0 * t]); L = 0.99
    Pn = P + np.column_stack([rng.normal(0, 0.005, len(t)), np.zeros(len(t)), rng.normal(0, 0.005, len(t))])
    assert arc_length(t, Pn, "simpson", smooth=False) > 1.2 * L        # interpolating noisy points inflates the length ...
    assert abs(arc_length(t, Pn, "simpson", smooth=True) - L) < 0.05 * L   # ... the smoothing spline does not


def test_ode_solver_orders():
    for m, o in {"euler": 1, "heun": 2, "rk4": 4}.items():
        assert abs(ex.verify_ode_solvers((20, 40, 80))[m] - o) < 0.35, m


def test_solve_ode_exponential_decay_rk4():
    t, x = solve_ode(lambda t, x: -x, 1.0, 0.0, 1.0, 100, "rk4")
    assert abs(x[-1] - np.exp(-1)) < 1e-9 and len(t) == 101


def test_ridge_normal_equations_and_svd_match_sklearn():
    rng = np.random.default_rng(0); A = rng.normal(size=(60, 8)); B = rng.normal(size=(60, 3)); lam = 0.7
    W = Ridge(alpha=lam, fit_intercept=False).fit(A, B).coef_.T
    assert np.allclose(ridge_normal_eq(A, B, lam), W) and np.allclose(ridge_svd(A, B, lam), W)


def test_observed_order_recovers_known_slope():
    h = np.array([0.1, 0.05, 0.025]); assert abs(observed_order(h, 3 * h ** 2) - 2) < 1e-9
