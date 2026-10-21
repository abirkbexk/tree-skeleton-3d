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


# ---- real Monash/TreeID naming and formats -------------------------------------------------------------------------------------------
def test_classify_real_filenames(tmp_path):
    """R01N01_Summer_{RGB,D,GT}-h-m-s.png -> roles by filename, season from the FILENAME (not the folder name)."""
    from treeskel.data import classify_png
    folder = tmp_path / "Dataset (With Summer GT)"
    got = {k: classify_png(folder / f"R01N01_{s}_{k}-12-40-50.png", tmp_path) for k, s in
           [("RGB", "Summer"), ("D", "Summer"), ("GT", "Summer")]}
    assert got["RGB"][0] == "rgb" and got["D"][0] == "depth" and got["GT"][0] == "skeleton"
    assert len({v[1] for v in got.values()}) == 1                       # same pairing key
    assert classify_png(folder / "R01N01_Winter_RGB-11-23-49.png", tmp_path)[3] == "winter"


def test_square_gt_is_centre_crop_and_8bit_depth_inverted(tmp_path):
    """A 256x256 GT for a 480x640 image is the centre-square crop; 8-bit depth: 0 = missing, bright = near."""
    import cv2, numpy as np
    from treeskel.config import get_config
    from treeskel.data import load_sample
    H, W = 48, 64
    rgb = np.full((H, W, 3), 120, np.uint8)
    d = np.zeros((H, W, 3), np.uint8); d[:, :, :] = 200; d[:, 30:34] = 0           # a hole inside the centre-square crop (cols 8..55)
    gt = np.zeros((32, 32), np.uint8); gt[:, 16] = 255                 # vertical line in the middle of the crop
    for n, a in [("a_rgb", rgb), ("a_d", d), ("a_gt", gt)]:
        cv2.imwrite(str(tmp_path / f"{n}.png"), a)
    rec = dict(rgb=tmp_path / "a_rgb.png", depth=tmp_path / "a_d.png", skel=tmp_path / "a_gt.png")
    cfg = get_config(DEPTH_8BIT_INVERTED=True)
    with pytest.warns(UserWarning):
        s = load_sample(rec, 32, cfg)
    cols = np.where(s["mask"].any(axis=0))[0]
    assert cols.min() >= 14 and cols.max() <= 18                        # line stays centred (not stretched to full width)
    assert np.isnan(s["depth"]).any()                                     # zero reading -> missing (NaN)
    assert np.nanmean(s["depth"]) < 0.5 * cfg["DEPTH_MAX_M"]            # bright 200/255 = near
