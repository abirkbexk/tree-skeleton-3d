"""Re-create every table/figure used in the report from a trained pipeline.

Usage:  python scripts/run_experiments.py --data-mode synthetic --quick --no-cnn --out-dir outputs
Writes CSV tables and PNG figures to ``<out-dir>/experiments``.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from treeskel import experiments as ex
from treeskel import get_config, train_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mode", default="synthetic"); ap.add_argument("--local-path"); ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--no-cnn", action="store_true")
    a = ap.parse_args()
    cfg = get_config(DATA_MODE=a.data_mode, LOCAL_PATH=a.local_path, OUT_DIR=a.out_dir, QUICK_TEST=a.quick, RUN_CNN=not a.no_cnn)
    out = train_pipeline(cfg, out_dir=a.out_dir); ds, X = out["ds"], out["X"]; TR, VA, TE = out["splits"]; alpha = out["alpha"]
    d = Path(a.out_dir) / "experiments"; d.mkdir(parents=True, exist_ok=True)

    svd, sv = ex.svd_target_analysis(ds["targets"], cfg["IMG_SIZE"]); svd.to_csv(d / "svd_targets.csv")
    ex.learning_curve(X, ds["targets"], ds, TR, VA, alpha, cfg).to_csv(d / "learning_curve.csv")
    se = ex.season_experiment(X, ds["targets"], ds, TR, TE, alpha, cfg)
    if se is not None: se.to_csv(d / "season_experiment.csv", index=False)
    i = int(np.argmax([v.sum() for v in ds["valid"]])); ex.interpolation_overfit_demo(ds["rows"][i, 0], ds["valid"][i]).to_csv(d / "interpolation_overfit.csv")
    ex.quadrature_convergence().to_csv(d / "quadrature_convergence.csv")
    ex.ode_reconstruction_experiment(X, ds, TR, TE, alpha, cfg).to_csv(d / "ode_reconstruction.csv")
    ex.resolution_study(ds, TR, VA, alpha, cfg).to_csv(d / "resolution_study.csv")
    errs, cond, _ = ex.conditioning_study(X, ds["targets"], TR)
    (d / "verification.txt").write_text(f"finite-difference observed order: {ex.verify_finite_differences()}\nODE solver observed order: {ex.verify_ode_solvers()}\n"
                                        f"conditioning: cond(A)={cond:.3e}\nrelative errors vs float64 SVD: {errs}\n")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); ax[0].semilogy(sv[:40], "o-"); ax[0].set_title("singular values of centred targets")
    ax[1].plot(svd.index, svd.measured_rmse_px, "o-", label="measured"); ax[1].plot(svd.index, svd.eckart_young_rmse_px, "k--", label="Eckart-Young"); ax[1].legend()
    fig.savefig(d / "svd_targets.png", dpi=150, bbox_inches="tight")
    print("experiment outputs written to", d)


if __name__ == "__main__":
    main()
