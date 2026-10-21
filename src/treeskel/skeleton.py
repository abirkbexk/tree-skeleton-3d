"""Skeleton representation: per-row left/right branch positions extracted from annotation masks (Week 3: splines)."""
import numpy as np
from scipy import interpolate


def skeleton_to_rows(mask):
    """Extract left/right branch centre per image row from a boolean skeleton mask.

    Args: mask (H,W) bool.  Returns: rows (2,H) float with NaN where the row has no skeleton, valid (H,) bool.
    One run of pixels -> both branches share its centre; several runs -> outermost runs are left/right."""
    H = mask.shape[0]
    rows = np.full((2, H), np.nan)
    for r in range(H):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            continue
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        rows[0, r], rows[1, r] = runs[0].mean(), runs[-1].mean()
    return rows, np.isfinite(rows[0])


def fill_rows(v, valid):
    """Fill gaps in a 1-D profile: PCHIP inside the valid extent, constant hold outside. (len>=1 valid value required)"""
    out = np.array(v, dtype=float)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.full(len(out), np.nan)
    if idx.size == 1:
        out[:] = out[idx[0]]; return out
    inner = np.arange(idx[0], idx[-1] + 1)
    out[inner] = interpolate.PchipInterpolator(idx, out[idx])(inner)
    out[:idx[0]] = out[idx[0]]; out[idx[-1] + 1:] = out[idx[-1]]
    return out


def junction_row(left, right, tol):
    """Lowest image row (largest index) where the branches are further apart than `tol` px; NaN if they never split."""
    idx = np.flatnonzero(np.abs(right - left) > tol)
    return float(idx.max()) if idx.size else np.nan
