"""Classical machine-learning regressors for skeleton prediction (Week 5: regression, model evaluation, training-data selection).

All models share one pipeline: standardise -> PCA (SVD) on features -> regressor -> inverse PCA on targets.
PCA is fitted inside cross-validation, so no information leaks from validation folds.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .metrics import evaluate_predictions


def make_pipeline(reg, n_pca_feat, n_pca_target, standardise_target=False, seed=42):
    """standardise -> PCA(features) -> TransformedTargetRegressor(reg, PCA(targets)).

    Args:
        reg: an sklearn regressor. n_pca_feat / n_pca_target: PCA sizes for features / targets.
        standardise_target: also standardise the PCA target coefficients (needed for SVR / MLP).
        seed: PCA random state.
    """
    tt = [("pca", PCA(n_components=n_pca_target, random_state=seed))]
    if standardise_target:
        tt.append(("sc", StandardScaler()))
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=n_pca_feat, random_state=seed)),
                     ("reg", TransformedTargetRegressor(regressor=reg, transformer=Pipeline(tt), check_inverse=False))])


def pca_sizes(cfg, n_train, n_features):
    """PCA sizes that fit inside a CV fold: (n_pca_feat, n_pca_target)."""
    n_pf = min(cfg["N_PCA_FEAT"], int(0.6 * n_train), n_features)
    return n_pf, min(cfg["N_PCA_TARGET"], n_pf)


def build_specs(cfg, n_train, n_features):
    """Model zoo: {name: (pipeline, hyper-parameter grid)}. Grids shrink when ``cfg['QUICK_TEST']``."""
    n_pf, n_pt = pca_sizes(cfg, n_train, n_features); seed = cfg["SEED"]; quick = cfg["QUICK_TEST"]
    mk = lambda reg, std=False: make_pipeline(reg, n_pf, n_pt, std, seed)
    # MLP early stopping holds out an internal validation split, which needs enough samples (n_train >= 100)
    return {
        "Ridge": (mk(Ridge()), {"reg__regressor__alpha": [1, 10, 100, 1e3, 1e4]}),
        "DecisionTree": (mk(DecisionTreeRegressor(random_state=seed)),
                         {"reg__regressor__max_depth": [4, 8, 12] if quick else [3, 5, 8, 12, None], "reg__regressor__min_samples_leaf": [2, 5]}),
        "kNN": (mk(KNeighborsRegressor(weights="distance")), {"reg__regressor__n_neighbors": [3, 5, 9] if quick else [3, 5, 9, 15]}),
        "SVR": (mk(MultiOutputRegressor(SVR(kernel="rbf", epsilon=0.05)), True), {"reg__regressor__estimator__C": [1, 10] if quick else [1, 10, 100]}),
        "MLP": (mk(MLPRegressor(hidden_layer_sizes=(128,), early_stopping=n_train >= 100, max_iter=300, random_state=seed), True),
                {"reg__regressor__alpha": [1e-3, 1e-1], "reg__regressor__hidden_layer_sizes": [(128,), (256, 128)]}),
    }


def predict_rows_sklearn(model, X, size):
    """Fitted sklearn pipeline -> predicted branch positions in pixels, shape (N, 2, size)."""
    return model.predict(X).reshape(len(X), 2, size) * (size - 1)


def train_classical(X, Y, groups, train_idx, cfg):
    """Fit the mean baseline and every model in :func:`build_specs` with grouped CV on the training indices only.

    Returns:
        (fitted: {name: estimator}, info: {name: dict(best_params, cv_rmse_px, fit_s)}).
    """
    S = cfg["IMG_SIZE"]; fitted, info = {}, {}
    base = DummyRegressor(strategy="mean"); t0 = time.perf_counter(); base.fit(X[train_idx], Y[train_idx])
    fitted["MeanSkeleton"] = base; info["MeanSkeleton"] = dict(best_params={}, cv_rmse_px=np.nan, fit_s=time.perf_counter() - t0)
    cv = GroupKFold(n_splits=3 if cfg["QUICK_TEST"] else 5)
    for name, (pipe, grid) in build_specs(cfg, len(train_idx), X.shape[1]).items():
        t0 = time.perf_counter()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True)
        gs.fit(X[train_idx], Y[train_idx], groups=groups[train_idx])
        fitted[name] = gs.best_estimator_
        info[name] = dict(best_params=gs.best_params_, cv_rmse_px=-gs.best_score_ * (S - 1), fit_s=time.perf_counter() - t0)
        print(f"{name:13s} best={gs.best_params_}  CV RMSE={info[name]['cv_rmse_px']:.2f}px  ({info[name]['fit_s']:.1f}s incl. search)")
    return fitted, info


def evaluate_model(model, X, idx, ds, cfg):
    """Predict ``idx`` and score against ground truth. Returns (pred (n,2,S) px, per-image metrics DataFrame, ms per image)."""
    t0 = time.perf_counter(); pred = predict_rows_sklearn(model, X[idx], cfg["IMG_SIZE"]); dt = time.perf_counter() - t0
    return pred, evaluate_predictions(pred, ds["rows"][idx], ds["valid"][idx], cfg["JUNCTION_TOL_PX"]), 1000 * dt / len(idx)


def fit_ridge_subset(X, Y, train_idx, alpha, cfg):
    """Ridge pipeline fitted on a subset (used for learning curves and training-data-selection experiments)."""
    n_pf, n_pt = pca_sizes(cfg, len(train_idx), X.shape[1])
    n_pf, n_pt = min(n_pf, len(train_idx) - 1), min(n_pt, len(train_idx) - 1)
    return make_pipeline(Ridge(alpha=alpha), n_pf, n_pt, False, cfg["SEED"]).fit(X[train_idx], Y[train_idx])


def results_table(rows: list[dict]) -> pd.DataFrame:
    """List of per-model result dicts -> table sorted by VALIDATION rmse (selection criterion)."""
    return pd.DataFrame(rows).set_index("model").round(3).sort_values("val_rmse_px")
