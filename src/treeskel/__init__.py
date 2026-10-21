"""treeskel - 3D apple-tree skeleton extraction from RGB-D PNG images (MMA3001 individual project).

Typical use::

    from treeskel import get_config, train_pipeline, extract_skeleton_3d
    out = train_pipeline(get_config(DATA_MODE="local", LOCAL_PATH="path/to/pngs"), out_dir="outputs")
    res = extract_skeleton_3d("tree_rgb.png", "tree_depth.png", model=out["model"])
"""
from . import (cli, cnn, config, data, dataset, experiments, features, geometry, metrics, models, numerics, ode_recon, pipeline, postprocess, skeleton,
               synthetic)
from .config import DEFAULT_CONFIG, get_config
from .pipeline import SkeletonModel, extract_skeleton_3d, train_pipeline

__all__ = ["DEFAULT_CONFIG", "get_config", "SkeletonModel", "extract_skeleton_3d", "train_pipeline",
           # submodules (listed so that pdoc documents them)
           "cli", "cnn", "config", "data", "dataset", "experiments", "features", "geometry", "metrics", "models", "numerics", "ode_recon", "pipeline",
           "postprocess", "skeleton", "synthetic"]
__version__ = "0.1.0"
