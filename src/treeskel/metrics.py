"""Skeleton evaluation metrics (pixels at working resolution, evaluated on annotated rows only)."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .skeleton import junction_row


def evaluate_predictions(pred, rows_gt, valid, tol=3.0):
    """Per-image skeleton errors. pred, rows_gt: (N,2,S) pixels; valid: (N,S) bool. Returns DataFrame."""
    ys = np.arange(rows_gt.shape[-1]); out = []
    for p_full, g_full, v in zip(pred, rows_gt, valid):
        p, g = p_full[:, v], g_full[:, v]
        rmse = np.sqrt(np.mean((p - g) ** 2))
        r = np.corrcoef(p.ravel(), g.ravel())[0, 1] if np.std(p) > 0 and np.std(g) > 0 else np.nan
        P = np.vstack([np.column_stack([p[b], ys[v]]) for b in range(2)]); G = np.vstack([np.column_stack([g[b], ys[v]]) for b in range(2)])
        chamfer = 0.5 * (cKDTree(G).query(P)[0].mean() + cKDTree(P).query(G)[0].mean())
        jp = junction_row(p_full[0][v], p_full[1][v], tol); jg = junction_row(g_full[0][v], g_full[1][v], tol)
        ys_v = ys[v]
        jp = ys_v[int(jp)] if np.isfinite(jp) else np.nan; jg = ys_v[int(jg)] if np.isfinite(jg) else np.nan
        out.append(dict(rmse_px=rmse, r=r, chamfer_px=chamfer, junction_px=abs(jp - jg) if np.isfinite(jp) and np.isfinite(jg) else np.nan))
    return pd.DataFrame(out)


def summarise(df):
    """Mean / median / std of every metric column of an :func:`evaluate_predictions` table."""
    return df.agg(["mean", "median", "std"]).T.round(3)
