"""End-to-end training and inference.

* :func:`train_pipeline` - acquire data, build dataset, train/select models on validation data, choose post-processing, fit extent model.
* :class:`SkeletonModel` - the deployable bundle (regressor + extent model + post-processing); ``save`` / ``load``.
* :func:`extract_skeleton_3d` - top-level API: RGB(-D) PNG(s) -> 2D/3D skeleton, branch lengths, optional CSV/PLY export.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import acquire_data, depth_normalised, discover_samples, load_sample
from .dataset import assemble_dataset
from .features import featurise, make_splits
from .geometry import lift_to_3d
from .metrics import evaluate_predictions
from .models import evaluate_model, pca_sizes, predict_rows_sklearn, train_classical
from .numerics import arc_length
from .postprocess import postprocess
from .skeleton import junction_row

POST_CANDIDATES = [("none", 0)] + [("poly", d) for d in (2, 3, 4, 5, 7, 9, 12, 15)] + [("spline", s) for s in (0.05, 0.2, 1, 3, 10, 30, 100)]


class SkeletonModel:
    """Deployable bundle: selected regressor + skeleton-extent model + post-processing, so inference needs only images."""

    def __init__(self, name, model, extent_model, post, cfg):
        self.name, self.model, self.extent_model, self.post, self.cfg = name, model, extent_model, tuple(post), cfg

    def predict(self, rgb, dnorm):
        """Args: rgb (N,S,S,3) uint8, dnorm (N,S,S). Returns (rows (N,2,S) px after post-processing, extent (N,2) top/bottom rows)."""
        S = self.cfg["IMG_SIZE"]
        X = featurise(rgb, dnorm, self.cfg["FEAT_SIZE"])
        if self.name == "CNN":
            from .cnn import predict_cnn
            rows = predict_cnn(self.model, rgb, dnorm, S)
        else:
            rows = predict_rows_sklearn(self.model, X, S)
        ext = np.clip(self.extent_model.predict(X) * (S - 1), 0, S - 1)
        return postprocess(rows, *self.post), ext

    def save(self, path):
        """Persist to one joblib file (CNN weights are stored as a CPU state_dict)."""
        model = self.model
        if self.name == "CNN":
            model = {k: v.cpu() for k, v in self.model.state_dict().items()}
        joblib.dump(dict(name=self.name, model=model, extent_model=self.extent_model, post=self.post, cfg=self.cfg), path)

    @classmethod
    def load(cls, path):
        """Load a file written by :meth:`save`. Only load files you created yourself (joblib uses pickle)."""
        d = joblib.load(path)
        model = d["model"]
        if d["name"] == "CNN":
            from .cnn import build_cnn
            net = build_cnn(2 * d["cfg"]["IMG_SIZE"]); net.load_state_dict(model); net.eval(); model = net
        return cls(d["name"], model, d["extent_model"], d["post"], d["cfg"])


def select_postprocessing(pred_val, ds, val_idx, cfg, candidates=POST_CANDIDATES):
    """Choose smoothing method/parameter by VALIDATION rmse. Returns (table DataFrame, best (method, param))."""
    rows = []
    for m, p in candidates:
        e = evaluate_predictions(postprocess(pred_val, m, p), ds["rows"][val_idx], ds["valid"][val_idx], cfg["JUNCTION_TOL_PX"]).rmse_px.mean()
        rows.append(dict(method=m, param=p, val_rmse_px=e))
    tab = pd.DataFrame(rows); best = tab.loc[tab.val_rmse_px.idxmin()]
    return tab, (best.method, float(best.param))


def fit_extent_model(X, ds, train_idx, cfg, alpha):
    """Ridge on PCA features predicting the first/last annotated row (needed to know where the tree ends when no ground truth exists)."""
    n_pf, _ = pca_sizes(cfg, len(train_idx), X.shape[1])
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_pf, random_state=cfg["SEED"])), ("ridge", Ridge(alpha=alpha))]
                    ).fit(X[train_idx], ds["extent"][train_idx])


def train_pipeline(cfg, out_dir=None, verbose=True):
    """Full training run. Model selection uses the validation set; the test set is only scored for the final report.

    Returns:
        dict(model=SkeletonModel, ds, X, splits=(TR, VA, TE), results (DataFrame sorted by val RMSE), post_table, info, pred_val, pred_test).
    """
    S = cfg["IMG_SIZE"]
    root = acquire_data(cfg); records = discover_samples(root, cfg)
    ds = assemble_dataset(records, cfg); X = featurise(ds["rgb"], ds["dnorm"], cfg["FEAT_SIZE"])
    TR, VA, TE = make_splits(ds["group"], cfg["SEED"])
    if verbose: print(f"train {len(TR)} | val {len(VA)} | test {len(TE)}")
    fitted, info = train_classical(X, ds["targets"], ds["group"], TR, cfg)
    pred_val, pred_test, table = {}, {}, []
    for name, model in fitted.items():
        pred_val[name], mva, _ = evaluate_model(model, X, VA, ds, cfg)
        pred_test[name], mte, ms = evaluate_model(model, X, TE, ds, cfg)
        table.append(dict(model=name, val_rmse_px=mva.rmse_px.mean(), test_rmse_px=mte.rmse_px.mean(), test_r=mte.r.mean(), test_chamfer_px=mte.chamfer_px.mean(),
                          test_junction_px=mte.junction_px.mean(), fit_s=info[name]["fit_s"], predict_ms_per_img=ms))
    if cfg["RUN_CNN"]:
        from .cnn import predict_cnn, train_cnn
        net, hist, best_ep, fit_s = train_cnn(ds, TR, VA, cfg, verbose=verbose)
        fitted["CNN"] = net; info["CNN"] = dict(best_params=dict(epochs=best_ep + 1), cv_rmse_px=np.nan, fit_s=fit_s, history=hist)
        pred_val["CNN"] = predict_cnn(net, ds["rgb"][VA], ds["dnorm"][VA], S)
        t0 = time.perf_counter(); pred_test["CNN"] = predict_cnn(net, ds["rgb"][TE], ds["dnorm"][TE], S); ms = 1000 * (time.perf_counter() - t0) / len(TE)
        mva = evaluate_predictions(pred_val["CNN"], ds["rows"][VA], ds["valid"][VA], cfg["JUNCTION_TOL_PX"])
        mte = evaluate_predictions(pred_test["CNN"], ds["rows"][TE], ds["valid"][TE], cfg["JUNCTION_TOL_PX"])
        table.append(dict(model="CNN", val_rmse_px=mva.rmse_px.mean(), test_rmse_px=mte.rmse_px.mean(), test_r=mte.r.mean(), test_chamfer_px=mte.chamfer_px.mean(),
                          test_junction_px=mte.junction_px.mean(), fit_s=fit_s, predict_ms_per_img=ms))
    results = pd.DataFrame(table).set_index("model").round(3).sort_values("val_rmse_px")
    best = results.index[0]
    if verbose: print(results.to_string()); print("best model by validation RMSE:", best)
    post_table, post = select_postprocessing(pred_val[best], ds, VA, cfg)
    alpha = info["Ridge"]["best_params"]["reg__regressor__alpha"]
    extent_model = fit_extent_model(X, ds, TR, cfg, alpha)
    model = SkeletonModel(best, fitted[best], extent_model, post, cfg)
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "model_comparison.csv"); post_table.to_csv(out / "postprocessing_selection.csv"); model.save(out / "skeleton_model.joblib")
        if verbose: print("saved model + tables to", out)
    return dict(model=model, ds=ds, X=X, splits=(TR, VA, TE), results=results, post_table=post_table, info=info, fitted=fitted,
                pred_val=pred_val, pred_test=pred_test, post=post, alpha=alpha)


def extract_skeleton_3d(rgb_path, depth_path=None, model=None, cfg=None, out_dir=None):
    """Estimate the 3D skeleton (trunk + two primary branches) of one apple tree from RGB(-D) PNG(s).

    Args:
        rgb_path: path to the RGB PNG. depth_path: optional aligned depth PNG (uint16, millimetres by default).
        model: :class:`SkeletonModel` (or a path to a saved one). cfg: config dict (defaults to the model's own).
        out_dir: if given, writes ``<stem>_skeleton3d.csv`` and ``.ply`` there (only when depth is usable).

    Returns:
        dict(rows_px (2,m), v (m,) image rows, points_3d {'left','right'} (m,3) metres or None, length_m {'left','right'} or None,
        junction_row, extent (top, bottom), used_depth bool). 3D frame: X right, Y down, Z forward (metres).

    Raises:
        FileNotFoundError: missing RGB or depth file.  ValueError: input is not a PNG.
    """
    if model is None:
        raise ValueError("a trained SkeletonModel (or path to one) is required")
    if not isinstance(model, SkeletonModel):
        model = SkeletonModel.load(model)
    cfg = cfg or model.cfg; S = cfg["IMG_SIZE"]
    rgb_path = Path(rgb_path)
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if rgb_path.suffix.lower() != ".png":
        raise ValueError("only PNG input is supported")
    if depth_path is not None and not Path(depth_path).exists():
        raise FileNotFoundError(depth_path)
    s = load_sample(dict(rgb=rgb_path, depth=depth_path, skel=None), S, cfg, need_skeleton=False)
    used_depth = s["depth"] is not None and np.isfinite(s["depth"]).mean() > 0.2
    if not used_depth:
        warnings.warn("no usable depth: returning 2D skeleton only; accuracy is reduced (model was trained with a depth channel)")
    dn = depth_normalised(s["depth"]) if used_depth else np.zeros((S, S), np.float32)
    rows, ext = model.predict(s["rgb"][None], dn[None].astype(np.float16)); rows, ext = rows[0], ext[0]
    top, bot = int(round(ext[0])), int(round(ext[1]))
    v = np.unique(np.arange(min(top, bot - 10), max(bot, top + 10) + 1).clip(0, S - 1))
    jr = junction_row(rows[0][v], rows[1][v], cfg["JUNCTION_TOL_PX"])
    res = dict(rows_px=rows[:, v], v=v, points_3d=None, length_m=None, junction_row=float(v[int(jr)]) if np.isfinite(jr) else np.nan,
               extent=(int(v[0]), int(v[-1])), used_depth=bool(used_depth))
    if used_depth:
        pts = {b: lift_to_3d(rows[i][v], v.astype(float), s["depth"], s["scale"], cfg) for i, b in enumerate(("left", "right"))}
        if all(p is not None for p in pts.values()):
            res["points_3d"] = pts; res["length_m"] = {b: arc_length(v.astype(float), p, "simpson") for b, p in pts.items()}
    if out_dir is not None and res["points_3d"] is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.concat([pd.DataFrame(p, columns=["X_m", "Y_m", "Z_m"]).assign(branch=b, row=v) for b, p in res["points_3d"].items()])
        df.to_csv(out_dir / f"{rgb_path.stem}_skeleton3d.csv", index=False)
        with open(out_dir / f"{rgb_path.stem}_skeleton3d.ply", "w") as fh:
            fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(df)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
            fh.writelines(f"{x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in df[["X_m", "Y_m", "Z_m"]].values)
    return res
