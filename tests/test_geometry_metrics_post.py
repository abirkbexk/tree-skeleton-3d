import numpy as np

from treeskel.geometry import intrinsics, lift_to_3d, sample_depth_along
from treeskel.metrics import evaluate_predictions
from treeskel.postprocess import postprocess


def test_pinhole_back_projection(cfg):
    S = 64; depth = np.full((S, S), 2.0, np.float32); fx, fy, cx, cy = intrinsics(cfg, 1.0, size=S)
    P = lift_to_3d(np.array([cx, cx + 10.0, cx - 10.0] * 20), np.array([cy, cy + 5.0, cy] * 20), depth, 1.0, cfg, smooth=False)
    assert np.allclose(P[0], [0, 0, 2.0]) and np.isclose(P[1, 0], 10 * 2.0 / fx) and np.isclose(P[1, 1], 5 * 2.0 / fy)


def test_lift_without_depth_returns_none(cfg):
    assert lift_to_3d(np.arange(30.0), np.arange(30.0), np.full((64, 64), np.nan, np.float32), 1.0, cfg) is None
    assert sample_depth_along(np.full((64, 64), np.nan, np.float32), [1.0], [1.0]) is None


def test_foreground_depth_wins_over_background(cfg):
    d = np.full((64, 64), 3.0, np.float32); d[:, 30:34] = 2.0          # thin near branch in a far background
    z = sample_depth_along(d, np.full(20, 34.0), np.arange(20.0))    # query 1 px beside the branch (>=30 % of the 5x5 window is foreground)
    assert np.allclose(z, 2.0)
    far = sample_depth_along(d, np.full(20, 50.0), np.arange(20.0))  # far from the branch: background depth is returned unchanged
    assert np.allclose(far, 3.0)


def test_metrics_perfect_and_shifted(ds):
    g, v = ds["rows"][:3], ds["valid"][:3]
    m0 = evaluate_predictions(g, g, v); m1 = evaluate_predictions(g + 2.0, g, v)
    assert np.allclose(m0.rmse_px, 0) and np.allclose(m0.chamfer_px, 0) and np.allclose(m0.r, 1) and np.allclose(m1.rmse_px, 2.0)


def test_postprocess_enforces_branch_order_and_exact_for_smooth_input():
    S = 64; x = np.linspace(10, 50, S); pred = np.stack([x + 5, x - 5])[None]     # branches deliberately swapped
    out = postprocess(pred, "poly", 3)
    assert np.all(out[0, 0] <= out[0, 1] + 1e-9) and np.allclose(out[0, 0], x - 5, atol=1e-6)
