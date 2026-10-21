"""Synthetic Y-shaped apple trees, used ONLY to test the pipeline end to end when the real dataset is unavailable."""
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


def synthetic_tree(rng, season, H=480, W=640):
    """Generate one synthetic Y-shaped tree observation (for testing the pipeline only).

    Returns dict(rgb uint8 HxWx3, depth_mm uint16 HxW, mask uint8 HxW)."""
    ys = np.arange(H, dtype=np.float64)
    cx = W / 2 + rng.uniform(-0.06, 0.06) * W
    junc = H * rng.uniform(0.55, 0.78)
    top = int(H * rng.uniform(0.05, 0.12))
    lean = rng.uniform(-0.10, 0.10)
    t = np.clip((junc - ys) / junc, 0, None)
    trunk = cx + lean * (ys - junc)
    left = trunk - H * rng.uniform(0.16, 0.30) * t ** rng.uniform(1.0, 1.9)
    right = trunk + H * rng.uniform(0.16, 0.30) * t ** rng.uniform(1.0, 1.9)
    mask = np.zeros((H, W), np.uint8)
    for xs in (left, right):
        for y in range(top, H - 1):
            th = 3 + int(7 * y / H)
            cv2.line(mask, (int(round(xs[y])), y), (int(round(xs[y + 1])), y + 1), 255, th)
    g = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
    rgb = np.stack([0.62 + 0.10 * g, 0.74 + 0.05 * g, 0.85 - 0.25 * g], -1)
    rgb += rng.normal(0, 0.03, rgb.shape)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    brown = np.array([110, 80, 50]) + rng.normal(0, 6, (H, W, 3))
    rgb[mask > 0] = np.clip(brown[mask > 0], 0, 255).astype(np.uint8)
    leaf = np.zeros((H, W), np.uint8)
    n_leaf = {"winter": 0, "autumn": int(rng.integers(15, 40)), "summer": int(rng.integers(60, 120))}[season]
    pts = np.column_stack(np.nonzero(mask))  # (row, col)
    for _ in range(n_leaf):
        r, c = pts[rng.integers(len(pts))] + rng.integers(-25, 26, 2)
        rad = int(rng.integers(6, 17))
        col = (int(rng.integers(40, 90)), int(rng.integers(110, 170)), int(rng.integers(30, 80)))
        cv2.circle(rgb, (int(c), int(r)), rad, col, -1); cv2.circle(leaf, (int(c), int(r)), rad, 255, -1)
    if season == "summer":
        for _ in range(int(rng.integers(0, 6))):
            r, c = pts[rng.integers(len(pts))] + rng.integers(-30, 31, 2)
            cv2.circle(rgb, (int(c), int(r)), 10, (190, 30, 40), -1); cv2.circle(leaf, (int(c), int(r)), 10, 255, -1)
    depth = 3.0 + 0.4 * g + rng.normal(0, 0.01, (H, W))
    zone = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
    depth[zone] = 2.0 + 0.0005 * np.broadcast_to(ys[:, None], (H, W))[zone]
    depth[leaf > 0] = 1.85
    depth_mm = (depth * 1000).astype(np.uint16)
    depth_mm[rng.random((H, W)) < 0.03] = 0
    return dict(rgb=rgb, depth_mm=depth_mm, mask=mask)


def write_synthetic_dataset(out_dir, n, seed=0):
    """Write `n` synthetic (rgb, depth, skeleton) PNG triplets to `out_dir` and return the folder."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seasons = rng.choice(["winter", "autumn", "summer"], size=n, p=[0.45, 0.25, 0.30])
    for i, season in enumerate(tqdm(seasons, desc="synthetic trees")):
        d = synthetic_tree(rng, season)
        stem = f"{season}_{i:04d}"
        cv2.imwrite(str(out_dir / f"{stem}_rgb.png"), cv2.cvtColor(d["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{stem}_depth.png"), d["depth_mm"])
        cv2.imwrite(str(out_dir / f"{stem}_skeleton.png"), d["mask"])
    return out_dir
