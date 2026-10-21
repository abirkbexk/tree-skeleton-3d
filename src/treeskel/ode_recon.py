"""Alternative skeleton formulation: regress branch slopes then integrate the ODE dx/dt = s(t) (Week 7: Euler, Heun, RK4)."""
import numpy as np
from scipy import interpolate

from .numerics import solve_ode


def to_up(rows):
    """rows (N,2,S) -> branch profiles ordered from the trunk base upward (flip rows)."""
    return rows[:, :, ::-1]


def reconstruct(base_px, slope_up, method):
    """Integrate slope profile (2,S) from base positions (2,) with `method`; returns rows (2,S) in image-row order."""
    S = slope_up.shape[1]
    t = np.arange(S, dtype=float); out = np.empty((2, S))
    for b in range(2):
        spl = interpolate.CubicSpline(t, slope_up[b])
        _, x = solve_ode(lambda tt, xx, spl=spl: float(spl(tt)), base_px[b], 0.0, S - 1.0, S - 1, method)
        out[b] = x[::-1]
    return out
