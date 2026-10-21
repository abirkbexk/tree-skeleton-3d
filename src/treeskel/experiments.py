"""Analysis experiments used in the report (SVD, training-data selection, verification of numerical methods, performance).

Each function returns tables / arrays and never plots, so results are reproducible and testable; ``scripts/run_experiments.py`` and the
notebook handle presentation.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import featurise
from .metrics import evaluate_predictions
from .models import fit_ridge_subset, make_pipeline, pca_sizes, predict_rows_sklearn
from .numerics import (fd_first, newton_coefficients, newton_eval, observed_order, ridge_normal_eq, ridge_svd, solve_ode)
from .ode_recon import reconstruct, to_up


# ---------------------------------------------------------------- Week 2: SVD of the targets -----------------------------------------------------
def svd_target_analysis(targets, size, max_k=40):
    """Rank-k reconstruction error of centred targets (measured vs Eckart-Young prediction from discarded singular values).

    Returns DataFrame indexed by k with ``energy``, ``measured_rmse_px`` and ``eckart_young_rmse_px``."""
    Yc = targets - targets.mean(0); U, sv, Vt = np.linalg.svd(Yc, full_matrices=False)
    norm = size - 1; ks = np.arange(1, min(max_k, len(sv)) + 1)
    return pd.DataFrame(dict(
        energy=(np.cumsum(sv ** 2) / np.sum(sv ** 2))[ks - 1],
        measured_rmse_px=[norm * np.sqrt(np.mean((Yc - (U[:, :k] * sv[:k]) @ Vt[:k]) ** 2)) for k in ks],
        eckart_young_rmse_px=[norm * np.sqrt(np.sum(sv[k:] ** 2) / Yc.size) for k in ks]), index=pd.Index(ks, name="k")), sv


# ---------------------------------------------------------------- Week 5: training-data evaluation -----------------------------------------------
def learning_curve(X, Y, ds, train_idx, val_idx, alpha, cfg, fractions=(0.1, 0.2, 0.35, 0.5, 0.75, 1.0), reps=3, seed=0):
    """Train/validation RMSE of Ridge versus training-set size. Returns DataFrame indexed by n_train."""
    rng = np.random.default_rng(seed); S = cfg["IMG_SIZE"]; rows = []
    for frac in fractions:
        for _ in range(reps if frac < 1 else 1):
            sub = rng.choice(train_idx, size=max(int(frac * len(train_idx)), 8), replace=False)
            m = fit_ridge_subset(X, Y, sub, alpha, cfg)
            rows.append(dict(n=len(sub), train=evaluate_predictions(predict_rows_sklearn(m, X[sub], S), ds["rows"][sub], ds["valid"][sub]).rmse_px.mean(),
                             val=evaluate_predictions(predict_rows_sklearn(m, X[val_idx], S), ds["rows"][val_idx], ds["valid"][val_idx]).rmse_px.mean()))
    return pd.DataFrame(rows).groupby("n").mean()


def season_experiment(X, Y, ds, train_idx, test_idx, alpha, cfg):
    """Does adding occluded seasons help? Train on winter-only vs all seasons, test on summer. Returns DataFrame or None if seasons are unknown."""
    seasons = ds["season"]
    if not {"winter", "summer"} <= set(seasons):
        return None
    te_summer = np.intersect1d(test_idx, np.flatnonzero(seasons == "summer")); rows = []
    for label, tr in {"winter only": np.intersect1d(train_idx, np.flatnonzero(seasons == "winter")), "all seasons": train_idx}.items():
        if len(te_summer) and len(tr) > 10:
            m = fit_ridge_subset(X, Y, tr, alpha, cfg)
            rows.append(dict(training_data=label, n_train=len(tr), summer_test_rmse_px=evaluate_predictions(
                predict_rows_sklearn(m, X[te_summer], cfg["IMG_SIZE"]), ds["rows"][te_summer], ds["valid"][te_summer]).rmse_px.mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Week 3: interpolation / over-fitting ------------------------------------------
def interpolation_overfit_demo(profile, valid, node_counts=(4, 6, 8, 12, 16, 24)):
    """RMSE of reconstructing a noise-free ground-truth branch from n equispaced samples with Newton polynomial / cubic spline / PCHIP.

    The Newton polynomial degrades for large n (Runge phenomenon) - the motivation for smoothing splines in post-processing."""
    idx = np.flatnonzero(valid); top, bot = idx[0], idx[-1]; q = np.arange(top, bot + 1); rows = []
    for n in node_counts:
        nodes = np.linspace(top, bot, n); f = np.interp(nodes, np.arange(len(profile)), profile)
        xs = (nodes - nodes[0]) / (nodes[-1] - nodes[0]); xq = (q - nodes[0]) / (nodes[-1] - nodes[0])
        err = lambda yhat: float(np.sqrt(np.mean((yhat - profile[q]) ** 2)))
        rows.append(dict(n_nodes=n, newton_poly=err(newton_eval(xs, newton_coefficients(xs, f), xq)),
                         cubic_spline=err(interpolate.CubicSpline(nodes, f)(q)), pchip=err(interpolate.PchipInterpolator(nodes, f)(q))))
    return pd.DataFrame(rows).set_index("n_nodes")


# ---------------------------------------------------------------- Weeks 6-7: verification of numerical methods ---------------------------------
def verify_finite_differences(ns=(16, 32, 64, 128, 256)):
    """Observed convergence order of forward / central first derivatives on sin(x). Expect (1, 2)."""
    hs = np.array([2 * np.pi / n for n in ns]); ec, ef = [], []
    for h in hs:
        x = np.arange(0, 2 * np.pi + h / 2, h)
        ec.append(np.max(np.abs(fd_first(np.sin(x), h, "central")[2:-2] - np.cos(x)[2:-2])))
        ef.append(np.max(np.abs(fd_first(np.sin(x), h, "forward")[2:-2] - np.cos(x)[2:-2])))
    return dict(forward=observed_order(hs, ef), central=observed_order(hs, ec))


def quadrature_convergence(levels=range(2, 9)):
    """|error| of trapezoid / Simpson / Romberg / Gauss-Legendre on the arc length of sin(3t) on [0,2] (reference: adaptive QUADPACK)."""
    f = lambda t: np.sqrt(1 + (3 * np.cos(3 * t)) ** 2)
    ref = integrate.quad(f, 0, 2, epsabs=1e-13, epsrel=1e-13, limit=500)[0]; rows = []
    for k in levels:
        n = 2 ** k + 1; t = np.linspace(0, 2, n); gn, gw = np.polynomial.legendre.leggauss(n)
        rows.append(dict(n=n, trapezoid=abs(integrate.trapezoid(f(t), t) - ref), simpson=abs(integrate.simpson(f(t), x=t) - ref),
                         romberg=abs(integrate.romb(f(t), dx=t[1] - t[0]) - ref), gauss_legendre=abs(np.sum(gw * f(t[0] + (gn + 1))) - ref)))
    return pd.DataFrame(rows).set_index("n")


def verify_ode_solvers(ns=(10, 20, 40, 80, 160)):
    """Observed order of Euler / Heun / RK4 on x' = -2x + t, x(0)=1 (exact x = t/2 - 1/4 + 5/4 e^-2t). Expect (1, 2, 4)."""
    f = lambda t, x: -2 * x + t; exact = lambda t: t / 2 - 0.25 + 1.25 * np.exp(-2 * t); ns = np.array(ns)
    return {m: observed_order(2.0 / ns, [abs(solve_ode(f, 1.0, 0, 2, n, m)[1][-1] - exact(2.0)) for n in ns]) for m in ("euler", "heun", "rk4")}


def ode_reconstruction_experiment(X, ds, train_idx, test_idx, alpha, cfg, max_test=100):
    """Regress slopes + base position, integrate with Euler/Heun/RK4. ``oracle`` rows use true slopes (isolates integration error from learning error)."""
    S = cfg["IMG_SIZE"]; norm = S - 1; n_pf, n_pt = pca_sizes(cfg, len(train_idx), X.shape[1])
    up = to_up(ds["rows"]); slope = np.stack([[fd_first(up[i, b], 1.0, "central") for b in range(2)] for i in range(len(up))])
    base = up[:, :, 0] / norm; te = test_idx[:max_test]
    slope_model = make_pipeline(Ridge(alpha=alpha), n_pf, n_pt, False, cfg["SEED"]).fit(X[train_idx], slope[train_idx].reshape(len(train_idx), -1))
    base_model = make_pipeline(Ridge(alpha=alpha), n_pf, 2, False, cfg["SEED"]).fit(X[train_idx], base[train_idx])
    ps = slope_model.predict(X[te]).reshape(len(te), 2, S); pb = base_model.predict(X[te]) * norm; rows = []
    for method in ("euler", "heun", "rk4"):
        pred = np.stack([reconstruct(pb[j], ps[j], method) for j in range(len(te))])
        oracle = np.stack([reconstruct(up[i, :, 0], slope[i], method) for i in te])
        rows.append(dict(method=method, test_rmse_px=evaluate_predictions(pred, ds["rows"][te], ds["valid"][te]).rmse_px.mean(),
                         oracle_rmse_px=evaluate_predictions(oracle, ds["rows"][te], ds["valid"][te]).rmse_px.mean()))
    return pd.DataFrame(rows).set_index("method")


# ---------------------------------------------------------------- Week 2: performance, profiling, numerical errors -----------------------------
def resolution_study(ds, train_idx, val_idx, alpha, cfg, sizes=(12, 16, 24, 32, 48, 64)):
    """Feature resolution vs accuracy / time / memory for Ridge."""
    S = cfg["IMG_SIZE"]; Y = ds["targets"]; rows = []
    for fs in sizes:
        t0 = time.perf_counter(); Xr = featurise(ds["rgb"], ds["dnorm"], fs); t_feat = time.perf_counter() - t0
        n_pf, n_pt = pca_sizes(cfg, len(train_idx), Xr.shape[1])
        m = make_pipeline(Ridge(alpha=alpha), n_pf, n_pt, False, cfg["SEED"]); t0 = time.perf_counter(); m.fit(Xr[train_idx], Y[train_idx]); t_fit = time.perf_counter() - t0
        rows.append(dict(feat_size=fs, n_features=Xr.shape[1], feature_MB=Xr.nbytes / 1e6, featurise_s=t_feat, fit_s=t_fit,
                         val_rmse_px=evaluate_predictions(predict_rows_sklearn(m, Xr[val_idx], S), ds["rows"][val_idx], ds["valid"][val_idx]).rmse_px.mean()))
    return pd.DataFrame(rows).set_index("feat_size").round(3)


def conditioning_study(X, Y, train_idx, lam=1e-6, n_cols=300):
    """Ridge by normal equations vs SVD in float64 / float32. Returns (relative errors vs float64-SVD reference, cond(A), singular values)."""
    ncol = max(min(n_cols, len(train_idx) // 2), 10)
    A64 = StandardScaler().fit_transform(X[train_idx])[:, :ncol].astype(np.float64); B64 = Y[train_idx][:, :8].astype(np.float64)
    ref = ridge_svd(A64, B64, lam); sv = np.linalg.svd(A64, compute_uv=False); rel = lambda W: float(np.linalg.norm(W - ref) / np.linalg.norm(ref))
    A32, B32 = A64.astype(np.float32), B64.astype(np.float32)
    errs = {"float64 normal equations": rel(ridge_normal_eq(A64, B64, lam)), "float32 SVD": rel(ridge_svd(A32, B32, np.float32(lam))),
            "float32 normal equations": rel(ridge_normal_eq(A32, B32, np.float32(lam)))}
    return errs, float(sv[0] / sv[-1]), sv


def profile_inference(model, X, size, reps=20, top=8):
    """cProfile of repeated prediction. Returns the formatted stats text."""
    prof = cProfile.Profile(); prof.enable()
    for _ in range(reps):
        predict_rows_sklearn(model, X, size)
    prof.disable(); st = io.StringIO(); pstats.Stats(prof, stream=st).sort_stats("cumulative").print_stats(top)
    return st.getvalue()


def peak_memory_mb(fn):
    """Peak extra memory (MB, tracemalloc) while running ``fn()``."""
    tracemalloc.start(); fn(); peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop(); return peak


def ridge_flops_and_intensity(model, X, size, n_pca_target, reps=50):
    """Analytic FLOPs/image, bytes moved/image, arithmetic intensity and achieved GFLOP/s for the linear Ridge pipeline.

    FLOPs = standardise + feature PCA + regression head + inverse target PCA (2 per multiply-add); bytes assume float64 weights are streamed once per image."""
    d_in = X.shape[1]; k_f = model.named_steps["pca"].n_components_; k_t = n_pca_target
    flops = 2 * d_in + 2 * d_in * k_f + 2 * k_f * k_t + 2 * k_t * 2 * size
    nbytes = 8 * (d_in * k_f + k_f * k_t + k_t * 2 * size + d_in)
    t0 = time.perf_counter()
    for _ in range(reps):
        predict_rows_sklearn(model, X, size)
    dt = (time.perf_counter() - t0) / (reps * len(X))
    return dict(mflop_per_image=flops / 1e6, mb_per_image=nbytes / 1e6, intensity_flop_per_byte=flops / nbytes, ms_per_image=1e3 * dt, achieved_gflops=flops / dt / 1e9)
