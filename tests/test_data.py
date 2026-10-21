import cv2
import numpy as np
import pytest

from treeskel.data import classify_png, depth_normalised, discover_samples, fill_depth_griddata, fill_nearest, load_sample


def test_depth_fill_nearest_and_noop():
    d = np.full((8, 8), 2.0, np.float32); d[3:5, 3:5] = np.nan
    assert np.allclose(fill_nearest(d), 2.0) and np.allclose(fill_nearest(np.full((4, 4), 1.5, np.float32)), 1.5)


def test_griddata_fill_reproduces_plane():
    yy, xx = np.mgrid[0:20, 0:20]; d = (1.0 + 0.01 * xx + 0.02 * yy).astype(np.float32); hole = d.copy(); hole[8:11, 8:11] = np.nan
    assert np.allclose(fill_depth_griddata(hole, "linear", stride=1), d, atol=1e-4)


def test_classify_png_roles(tmp_path):
    for name, role in [("a_rgb.png", "rgb"), ("a_depth.png", "depth"), ("a_skeleton.png", "skeleton")]:
        assert classify_png(tmp_path / name, tmp_path)[0] == role
    assert classify_png(tmp_path / "summer" / "x_rgb.png", tmp_path)[3] == "summer"


def test_discover_pairs_every_synthetic_tree(records, cfg):
    assert len(records) == cfg["N_SYNTH"] and all(r["depth"] is not None for r in records)


def test_discover_raises_when_nothing_pairs(tmp_path, cfg):
    cv2.imwrite(str(tmp_path / "lonely_rgb.png"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError):
        discover_samples(tmp_path, cfg)


def test_empty_skeleton_mask_is_rejected(tmp_path, cfg):
    cv2.imwrite(str(tmp_path / "t_rgb.png"), np.zeros((40, 40, 3), np.uint8)); cv2.imwrite(str(tmp_path / "t_skeleton.png"), np.zeros((40, 40), np.uint8))
    rec = dict(rgb=tmp_path / "t_rgb.png", skel=tmp_path / "t_skeleton.png", depth=None)
    with pytest.raises(ValueError):
        load_sample(rec, 32, cfg)


def test_unreadable_image_raises(cfg, tmp_path):
    with pytest.raises(IOError):
        load_sample(dict(rgb=tmp_path / "missing.png", skel=tmp_path / "missing2.png", depth=None), 32, cfg)


def test_depth_png_scaled_to_metres_and_holes_nan(cfg, tmp_path):
    cv2.imwrite(str(tmp_path / "t_rgb.png"), np.zeros((50, 50, 3), np.uint8)); dep = np.full((50, 50), 2000, np.uint16); dep[0, 0] = 0
    cv2.imwrite(str(tmp_path / "t_depth.png"), dep)
    s = load_sample(dict(rgb=tmp_path / "t_rgb.png", skel=None, depth=tmp_path / "t_depth.png"), 25, cfg, need_skeleton=False)
    assert np.nanmedian(s["depth"]) == pytest.approx(2.0) and s["mask"] is None


def test_dataset_shapes_targets_and_mirror_identity(ds, cfg):
    S = cfg["IMG_SIZE"]; n = len(ds["rgb"])
    assert ds["rgb"].shape == (n, S, S, 3) and ds["targets"].shape == (n, 2 * S) and ds["rows"].shape == (n, 2, S)
    assert np.isfinite(ds["targets"]).all() and 0 <= ds["targets"].min() and ds["targets"].max() <= 1
    y = ds["targets"][:2].copy(); f = lambda y_: np.concatenate([1 - y_[:, S:], 1 - y_[:, :S]], 1)
    assert np.allclose(f(f(y)), y)                          # mirror augmentation is an involution


def test_depth_normalised_none_passthrough():
    assert depth_normalised(None) is None
