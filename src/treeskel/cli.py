"""Command-line entry points: ``treeskel-train`` and ``treeskel-predict``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_config
from .pipeline import extract_skeleton_3d, train_pipeline


def train_main(argv=None):
    """Train + select a model and write ``skeleton_model.joblib`` and result tables to ``--out-dir``."""
    ap = argparse.ArgumentParser(description="Train the apple-tree skeleton model.")
    ap.add_argument("--data-mode", default="gdown", choices=["gdown", "drive", "local", "synthetic", "upload"])
    ap.add_argument("--local-path", help="folder of PNGs (with --data-mode local)")
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--quick", action="store_true", help="tiny/fast settings for a smoke test")
    ap.add_argument("--no-cnn", action="store_true", help="skip the PyTorch CNN")
    ap.add_argument("--group-regex", help="regex (1 group) extracting a tree id from the file key, to prevent split leakage")
    ap.add_argument("--skeleton-variant", help="'whole' / 'visible' when several annotation PNGs exist per image")
    ap.add_argument("--max-images", type=int); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    cfg = get_config(DATA_MODE=a.data_mode, LOCAL_PATH=a.local_path, DATA_DIR=a.data_dir, OUT_DIR=a.out_dir, QUICK_TEST=a.quick,
                     RUN_CNN=not a.no_cnn, GROUP_REGEX=a.group_regex, SKELETON_VARIANT=a.skeleton_variant, MAX_IMAGES=a.max_images, SEED=a.seed)
    out = train_pipeline(cfg, out_dir=a.out_dir)
    print(out["results"].to_string())
    return 0


def predict_main(argv=None):
    """Predict the 3D skeleton of one RGB(-D) PNG pair and export CSV/PLY."""
    ap = argparse.ArgumentParser(description="Extract a 3D skeleton from RGB(-D) PNGs.")
    ap.add_argument("rgb"); ap.add_argument("--depth"); ap.add_argument("--model", default="outputs/skeleton_model.joblib")
    ap.add_argument("--out-dir", default="outputs/predictions")
    a = ap.parse_args(argv)
    res = extract_skeleton_3d(a.rgb, a.depth, model=a.model, out_dir=a.out_dir)
    print(json.dumps({k: (v if not hasattr(v, "shape") and k != "points_3d" else None) for k, v in res.items() if k in ("length_m", "junction_row", "extent", "used_depth")}, default=float, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(train_main())
