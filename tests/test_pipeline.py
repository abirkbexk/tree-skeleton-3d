import cv2
import numpy as np
import pytest

from treeskel import SkeletonModel, extract_skeleton_3d, get_config, train_pipeline
from treeskel.features import make_splits


def test_get_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        get_config(NOT_A_SETTING=1)
    assert get_config(QUICK_TEST=True, IMG_SIZE=96)["IMG_SIZE"] == 96


def test_grouped_split_has_no_leakage():
    groups = np.repeat(np.arange(20), 3); tr, va, te = make_splits(groups, 0)
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[tr]) & set(groups[va])) and not (set(groups[va]) & set(groups[te]))


@pytest.fixture(scope="module")
def trained(cfg, tmp_path_factory):
    c = dict(cfg, DATA_DIR=str(tmp_path_factory.mktemp("data")), N_SYNTH=60)
    return train_pipeline(c, out_dir=tmp_path_factory.mktemp("out"), verbose=False), c


def test_models_beat_mean_baseline_on_validation(trained):
    res = trained[0]["results"]
    assert res.index[0] != "MeanSkeleton" and res.loc[res.index[0], "val_rmse_px"] < 0.8 * res.loc["MeanSkeleton", "val_rmse_px"]


def test_save_load_roundtrip_predictions_identical(trained, tmp_path):
    out, c = trained; m = out["model"]; ds = out["ds"]; p = tmp_path / "m.joblib"; m.save(p)
    m2 = SkeletonModel.load(p); a = m.predict(ds["rgb"][:4], ds["dnorm"][:4]); b = m2.predict(ds["rgb"][:4], ds["dnorm"][:4])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_extract_skeleton_3d_end_to_end(trained, records, tmp_path):
    out, c = trained; r = records[0]
    res = extract_skeleton_3d(r["rgb"], r["depth"], model=out["model"], out_dir=tmp_path)
    assert res["used_depth"] and res["points_3d"]["left"].shape[1] == 3 and all(0.2 < L < 10 for L in res["length_m"].values())
    assert (tmp_path / f"{r['rgb'].stem}_skeleton3d.csv").exists() and (tmp_path / f"{r['rgb'].stem}_skeleton3d.ply").exists()


def test_extract_without_depth_warns_and_returns_2d_only(trained, records):
    with pytest.warns(UserWarning):
        res = extract_skeleton_3d(records[0]["rgb"], None, model=trained[0]["model"])
    assert res["points_3d"] is None and res["length_m"] is None and res["rows_px"].shape[0] == 2


def test_extract_invalid_inputs(trained, tmp_path):
    m = trained[0]["model"]
    with pytest.raises(FileNotFoundError):
        extract_skeleton_3d(tmp_path / "nope.png", model=m)
    bad = tmp_path / "x.jpg"; bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract_skeleton_3d(bad, model=m)
    with pytest.raises(ValueError):
        extract_skeleton_3d(tmp_path / "x.png", model=None)
