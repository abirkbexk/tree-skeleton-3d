"""Configuration for the whole pipeline.

All tunable settings live in one flat dictionary so that a run is fully described by (config, random seed).
Use :func:`get_config` instead of editing :data:`DEFAULT_CONFIG` in place.
"""
from __future__ import annotations

DEFAULT_CONFIG: dict = dict(
    # ---- data --------------------------------------------------------------------------------
    DATA_MODE="gdown",            # 'gdown' | 'drive' | 'upload' | 'local' | 'synthetic'
    GDRIVE_FILE_ID="1-6akVXezq26ihtDrVH9czGUQDFd8Py81",   # Monash Y-shaped tree dataset - CHECK this is your Dataset 1 link
    DRIVE_PATH="/content/drive/MyDrive/MMA3001/dataset1",  # used when DATA_MODE == 'drive'
    LOCAL_PATH=None,              # used when DATA_MODE == 'local': folder that already contains the PNGs
    DATA_DIR="data",              # download / working folder (git-ignored)
    OUT_DIR="outputs",            # models, tables, exports (git-ignored)
    SKELETON_VARIANT=None,        # e.g. 'whole' / 'visible' if several annotation PNGs exist per image
    FOURTH_CHANNEL_IS_DEPTH=False,  # True if depth is packed into the alpha channel of the RGB PNG
    GROUP_REGEX=None,             # regex with 1 group extracting a tree id from the file key (prevents train/test leakage)
    MAX_IMAGES=None,              # cap number of samples (None = all)
    N_SYNTH=240,                  # number of synthetic trees when DATA_MODE == 'synthetic'
    # ---- sensor (Intel RealSense D435, 640x480 colour; replace with calibrated values if available) ----
    FX=615.0, FY=615.0,
    DEPTH_SCALE_M=0.001,          # uint16 depth PNG in millimetres -> metres
    DEPTH_MAX_M=4.0,              # readings beyond this (or == 0) are treated as missing
    # ---- representation / models -------------------------------------------------------------
    IMG_SIZE=256,                 # square working resolution (= number of skeleton rows)
    FEAT_SIZE=32,                 # down-sampled size for classical-ML features
    N_PCA_FEAT=100, N_PCA_TARGET=16,
    JUNCTION_TOL_PX=3.0,          # branches count as 'split' when |right-left| > tol
    SEED=42,
    RUN_CNN=True, CNN_EPOCHS=60, CNN_BATCH=16, CNN_LR=1e-3,
    QUICK_TEST=False,             # tiny, fast settings for debugging
)

_QUICK_OVERRIDES = dict(N_SYNTH=90, IMG_SIZE=128, FEAT_SIZE=24, N_PCA_FEAT=40, N_PCA_TARGET=10, CNN_EPOCHS=3, CNN_BATCH=16)


def get_config(**overrides) -> dict:
    """Return a fresh configuration dict.

    Args:
        **overrides: any key of :data:`DEFAULT_CONFIG`. ``QUICK_TEST=True`` applies small, fast settings first
            (explicit overrides still win).

    Raises:
        KeyError: if an override key is not a known setting (catches typos).
    """
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg = dict(DEFAULT_CONFIG)
    if overrides.get("QUICK_TEST", cfg["QUICK_TEST"]):
        cfg.update(_QUICK_OVERRIDES)
    cfg.update(overrides)
    return cfg
