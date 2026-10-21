"""Smoke + sanity tests for the report experiments (they must run on small data and respect known theory)."""
import numpy as np
import pytest

from treeskel import experiments as ex
from treeskel.features import featurise, make_splits
from treeskel.ode_recon import reconstruct, to_up


@pytest.fixture(scope="module")
def setup(ds, cfg):
    X = featurise(ds["rgb"], ds["dnorm"], cfg["FEAT_SIZE"]); tr, va, te = make_splits(ds["group"], 0)
    return X, tr, va, te


def test_svd_reconstruction_matches_eckart_young(ds, cfg):
    tab, sv = ex.svd_target_analysis(ds["targets"], cfg["IMG_SIZE"])
    assert np.allclose(tab.measured_rmse_px, tab.eckart_young_rmse_px, rtol=1e-6) and tab.energy.is_monotonic_increasing and tab.energy.iloc[-1] <= 1 + 1e-12


def test_newton_polynomial_overfits_but_spline_does_not(ds):
    i = int(np.argmax([v.sum() for v in ds["valid"]])); t = ex.interpolation_overfit_demo(ds["rows"][i, 0], ds["valid"][i], (8, 24))
    assert t.loc[24, "newton_poly"] > 10 * t.loc[24, "cubic_spline"]          # Runge phenomenon at 24 equispaced nodes


def test_quadrature_convergence_ordering():
    q = ex.quadrature_convergence()
    assert q.simpson.iloc[-1] < q.trapezoid.iloc[-1] and q.romberg.iloc[-1] < 1e-8 and q.gauss_legendre.iloc[-1] < 1e-10


def test_ode_reconstruction_oracle_beats_learned(setup, ds, cfg):
    X, tr, va, te = setup
    t = ex.ode_reconstruction_experiment(X, ds, tr, te, alpha=100.0, cfg=cfg, max_test=10)
    assert (t.oracle_rmse_px < 1.0).all() and t.loc["rk4", "oracle_rmse_px"] <= t.loc["euler", "oracle_rmse_px"] + 1e-9


def test_reconstruct_recovers_straight_line():
    S = 32; slope = np.full((2, S), 0.5); out = reconstruct(np.array([10.0, 20.0]), slope, "rk4")          # up-ordered -> image rows reversed
    assert np.allclose(out[0][::-1], 10.0 + 0.5 * np.arange(S)) and np.allclose(to_up(out[None])[0, 0], out[0][::-1])


def test_learning_curve_and_season_experiment(setup, ds, cfg):
    X, tr, va, te = setup
    lc = ex.learning_curve(X, ds["targets"], ds, tr, va, 100.0, cfg, fractions=(0.5, 1.0), reps=1)
    assert len(lc) == 2 and (lc.val > 0).all()
    se = ex.season_experiment(X, ds["targets"], ds, tr, te, 100.0, cfg)
    assert se is None or "all seasons" in set(se.training_data)      # "winter only" is skipped when it has <= 10 training images


def test_conditioning_squares_condition_number(setup, ds):
    X, tr, va, te = setup; errs, cond, sv = ex.conditioning_study(X, ds["targets"], tr)
    assert errs["float32 normal equations"] > errs["float32 SVD"] and errs["float64 normal equations"] < 1e-6 and cond > 1


def test_resolution_study_runs(ds, cfg, setup):
    X, tr, va, te = setup; t = ex.resolution_study(ds, tr, va, 100.0, cfg, sizes=(8, 16))
    assert list(t.index) == [8, 16] and t.n_features.tolist() == [4 * 64, 4 * 256]
