"""RGB-D geometry: pinhole back-projection of skeleton pixels to 3D camera coordinates (metres) with robust depth sampling."""
import numpy as np
from scipy import ndimage

from .data import fill_nearest


def intrinsics(cfg, scale, size=None):
    """Pinhole intrinsics after centre-crop + resize: focal lengths scale with `scale`; principal point at the image centre."""
    size = size or cfg["IMG_SIZE"]
    return cfg["FX"] * scale, cfg["FY"] * scale, size / 2.0, size / 2.0


def sample_depth_along(depth, u, v, window=None, percentile=30):
    """Foreground depth (m) at sub-pixel points (u, v). None if depth is (almost) empty.

    Missing pixels are filled (nearest valid), then a low-percentile filter over a small window is applied so that a skeleton point that lies a few pixels off a thin
    branch reads the *tree* (near) rather than the distant background row; values are then bilinearly interpolated (2D interpolation, Week 3)."""
    if np.isfinite(depth).sum() < 50:
        return None
    window = window or 2 * int(round(depth.shape[0] / 40)) + 1
    fg = ndimage.percentile_filter(fill_nearest(depth), percentile, size=window, mode="nearest")
    return ndimage.map_coordinates(fg, [np.asarray(v, float), np.asarray(u, float)], order=1, mode="nearest")


def smooth_depth_profile(z, deg=3, iters=6):
    """Robust smooth depth along a branch. A rolling 30th-percentile filter (31 rows) suppresses background readings, which are always farther than the tree; a Huber-weighted (IRLS) cubic fit then rejects *persistent* outliers
    (e.g. rows where a slightly mis-placed skeleton sits on background or on a leaf). Assumes the tree is roughly a planar fruit wall, so depth varies smoothly along a branch."""
    z = ndimage.percentile_filter(z, 30, size=min(31, 2 * (len(z) // 2) - 1) if len(z) > 8 else 3, mode="nearest")   # background is always farther than the tree
    if len(z) <= deg + 3:
        return z
    t = np.linspace(-1, 1, len(z)); w = np.ones_like(z, dtype=float)
    for _ in range(iters):
        c = np.polynomial.polynomial.polyfit(t, z, deg, w=np.sqrt(w))
        r = z - np.polynomial.polynomial.polyval(t, c)
        s_ = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * s_))
    return np.polynomial.polynomial.polyval(t, c)


def lift_to_3d(u, v, depth, scale, cfg, smooth=True):
    """Back-project skeleton pixels (u,v) with depth map to 3D camera coordinates (X right, Y down, Z forward; metres). None if no depth."""
    z = sample_depth_along(depth, u, v)
    if z is None:
        return None
    if smooth:
        z = smooth_depth_profile(z)
    fx, fy, cx, cy = intrinsics(cfg, scale, size=depth.shape[0])
    return np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])
