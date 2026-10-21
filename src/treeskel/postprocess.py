"""Interpolation-based smoothing of predicted branch profiles (Week 3: polynomial vs smoothing spline, over-fitting)."""
import numpy as np
from scipy import interpolate


def smooth_profile(x, method, param):
    """Smooth a length-S branch profile. method: 'none' | 'poly' (degree=param) | 'spline' (smoothing factor per row = param)."""
    rows = np.arange(len(x))
    if method == "none":
        return np.asarray(x, float)
    if method == "poly":
        return np.polynomial.Polynomial.fit(rows, x, int(param))(rows)
    if method == "spline":
        return interpolate.UnivariateSpline(rows, x, k=3, s=param * len(x))(rows)
    raise ValueError(method)


def postprocess(pred, method, param):
    """Smooth both branches of every predicted skeleton (N,2,S) and enforce left <= right."""
    out = np.empty_like(pred, dtype=float)
    for i in range(len(pred)):
        for b in range(2):
            out[i, b] = smooth_profile(pred[i, b], method, param)
        lo, hi = np.minimum(out[i, 0], out[i, 1]), np.maximum(out[i, 0], out[i, 1])
        out[i, 0], out[i, 1] = lo, hi
    return out
