"""Data acquisition, PNG discovery/pairing, input validation, loading and depth-hole filling (Week 1 data management, Week 3 2D interpolation)."""
import re
import tarfile
import time
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import interpolate, ndimage

from .synthetic import write_synthetic_dataset


_ROLE_TOKENS = {
    "depth": {"depth", "depthmap", "dep", "d"},
    "skeleton": {"skeleton", "skel", "gt", "label", "labels", "mask", "annotation", "annotations", "annot",
                 "whole", "position", "groundtruth", "truth", "seg"},
    "rgb": {"rgb", "color", "colour", "image", "img"},
}


_VARIANT_TOKENS = {"visible", "whole", "position"}


_SEASONS = ("summer", "autumn", "winter")


def acquire_data(cfg):
    """Return a folder containing the PNG dataset according to cfg['DATA_MODE'] (synthetic | local | drive | gdown | upload)."""
    mode, root = cfg["DATA_MODE"], Path(cfg["DATA_DIR"]); root.mkdir(parents=True, exist_ok=True)
    if mode == "synthetic":
        return write_synthetic_dataset(root / "synthetic", cfg["N_SYNTH"], cfg["SEED"])
    if mode == "local":
        if not cfg.get("LOCAL_PATH") or not Path(cfg["LOCAL_PATH"]).is_dir():
            raise FileNotFoundError(f"DATA_MODE='local' needs LOCAL_PATH to be an existing folder, got {cfg.get('LOCAL_PATH')!r}")
        return Path(cfg["LOCAL_PATH"])
    if mode == "drive":
        from google.colab import drive
        drive.mount("/content/drive")
        return Path(cfg["DRIVE_PATH"])
    if mode == "gdown":
        import gdown
        archive = root / "dataset_download"
        if not archive.exists():
            gdown.download(id=cfg["GDRIVE_FILE_ID"], output=str(archive), quiet=False)
        dest = root / "raw"
        if not dest.exists():
            if zipfile.is_zipfile(archive):
                zipfile.ZipFile(archive).extractall(dest)
            elif tarfile.is_tarfile(archive):
                tarfile.open(archive).extractall(dest)
            else:
                raise RuntimeError("Downloaded file is neither zip nor tar - open the Drive link manually and use DATA_MODE='drive' or 'upload'.")
        return dest
    if mode == "upload":
        from google.colab import files
        up = files.upload()
        dest = root / "uploaded"; dest.mkdir(exist_ok=True)
        for name, data in up.items():
            (dest / name).write_bytes(data)
            if name.lower().endswith(".zip"):
                zipfile.ZipFile(dest / name).extractall(dest)
        return dest
    raise ValueError(f"unknown DATA_MODE {mode!r}")


def _tokens(text):
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def classify_png(path, root):
    """Infer (role, match_key, variant, season) of a PNG from its path relative to `root`."""
    rel = Path(path).relative_to(root)
    dir_toks = [_tokens(p) for p in rel.parts[:-1]]
    name_toks = _tokens(rel.stem)
    role = None
    for r in ("depth", "skeleton", "rgb"):           # filename tokens take priority over folder names
        if any(t in _ROLE_TOKENS[r] for t in name_toks):
            role = r; break
    if role is None:
        for r in ("depth", "skeleton", "rgb"):
            if any(t in _ROLE_TOKENS[r] for toks in dir_toks for t in toks):
                role = r; break
    role = role or "rgb"
    all_role = set().union(*_ROLE_TOKENS.values())
    variant = next((t for t in name_toks + sum(dir_toks, []) if t in _VARIANT_TOKENS), "")
    drop = all_role | _VARIANT_TOKENS
    key = "/".join("_".join(t for t in toks if t not in drop) for toks in dir_toks) + "|" + "_".join(t for t in name_toks if t not in drop)
    # season: the FILENAME wins (a folder called "... Summer GT" also holds winter files), then folder names
    season = next((s for s in _SEASONS if s in name_toks), None) or next((s for s in _SEASONS if any(s in t for toks in dir_toks for t in toks)), "unknown")
    return role, key, variant, season


def discover_samples(root, cfg):
    """Pair RGB / depth / skeleton PNGs. Returns list of record dicts and prints a diagnostic summary."""
    root = Path(root)
    pngs = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".png" and "__MACOSX" not in p.parts)
    groups = {}
    for p in pngs:
        role, key, variant, season = classify_png(p, root)
        groups.setdefault(key, {"season": season}).setdefault(role, []).append((p, variant))
    recs, incomplete, variants = [], [], set()
    for key, g in groups.items():
        if "rgb" not in g or "skeleton" not in g:
            incomplete.append(key); continue
        skels = g["skeleton"]
        variants.update(v for _, v in skels)
        if cfg["SKELETON_VARIANT"]:
            skels = [s for s in skels if s[1] == cfg["SKELETON_VARIANT"]]
            if not skels:
                incomplete.append(key); continue
        group_id = key
        if cfg["GROUP_REGEX"]:
            m = re.search(cfg["GROUP_REGEX"], key)
            group_id = m.group(1) if m else key
        recs.append(dict(key=key, rgb=g["rgb"][0][0], skel=skels[0][0], depth=g["depth"][0][0] if "depth" in g else None,
                         season=g["season"], group=group_id))
    print(f"{len(pngs)} PNG files -> {len(recs)} usable samples, {len(incomplete)} incomplete groups")
    print("   with depth:", sum(r["depth"] is not None for r in recs), "| skeleton variants seen:", sorted(variants) or "none")
    print("   seasons:", pd.Series([r["season"] for r in recs]).value_counts().to_dict())
    for r in recs[:3]:
        print("   e.g.", {k: (str(v)[-60:] if isinstance(v, Path) else v) for k, v in r.items()})
    if incomplete:
        print("   first incomplete keys:", incomplete[:3])
    if not recs:
        raise RuntimeError("No (rgb, skeleton) pairs found - adjust _ROLE_TOKENS / SKELETON_VARIANT / DATA_DIR.")
    if cfg["MAX_IMAGES"]:
        rng = np.random.default_rng(cfg["SEED"]); recs = [recs[i] for i in rng.permutation(len(recs))[:cfg["MAX_IMAGES"]]]
    return recs


def _read(path, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return img


def _center_square(img):
    h, w = img.shape[:2]; side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side], side


def load_sample(rec, size, cfg, need_skeleton=True):
    """Load one observation.

    Returns dict with rgb (size,size,3) uint8, depth (size,size) float32 metres (NaN = missing, or None),
    mask (size,size) bool (or None) and scale = size / crop_side (needed to scale camera intrinsics)."""
    rgb = _read(rec["rgb"])
    depth = None
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    if rgb.shape[2] == 4:
        if cfg["FOURTH_CHANNEL_IS_DEPTH"] and rec.get("depth") is None:
            depth = rgb[..., 3].astype(np.float32)
        rgb = rgb[..., :3]
    if rgb.dtype == np.uint16:
        rgb = (rgb / 257).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    if rec.get("depth") is not None:
        d = _read(rec["depth"])
        if d.ndim == 3:
            d = d[..., 0]
        if d.dtype == np.uint8:
            # 8-bit depth: the real metric scale is NOT stored in the file. Assumed linear 0..255 <-> 0..DEPTH_MAX_M metres; DEPTH_8BIT_INVERTED=True means
            # bright = NEAR (inferred for the Monash data: foliage is brighter than the wall behind it). 0 always means "no reading". Treat metres as approximate.
            warnings.warn("8-bit depth PNG: metric scale is assumed (linear 0..255 -> 0..DEPTH_MAX_M, inverted=%s); 3D lengths in metres are approximate"
                          % cfg.get("DEPTH_8BIT_INVERTED", True))
            v8 = d.astype(np.float32)
            depth = cfg["DEPTH_MAX_M"] * ((1.0 - v8 / 255.0) if cfg.get("DEPTH_8BIT_INVERTED", True) else v8 / 255.0)
            depth[v8 == 0] = np.nan
        else:
            depth = d.astype(np.float32) * cfg["DEPTH_SCALE_M"]
    elif depth is not None:
        depth = depth * cfg["DEPTH_SCALE_M"]
    if depth is not None:
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = np.where((depth <= 0) | (depth > cfg["DEPTH_MAX_M"]), np.nan, depth).astype(np.float32)
    mask = None
    if need_skeleton:
        m = _read(rec["skel"])
        if m.ndim == 3:
            m = m[..., :3].max(axis=2)
        m = m.astype(np.float32)
        if m.max() <= 0:
            raise ValueError(f"empty skeleton mask: {rec['skel']}")
        m = (m > 0.5 * m.max()).astype(np.float32)
        if m.mean() > 0.5:                      # black line on white background -> invert
            m = 1.0 - m
        if m.shape != (H, W):
            if m.shape[0] == m.shape[1] and H != W:
                # square annotation of a non-square image = the CENTRE SQUARE crop (Monash data: 256x256 GT for 480x640 RGB): put it back where it came from
                side_ = min(H, W); full = np.zeros((H, W), np.float32)
                y0_, x0_ = (H - side_) // 2, (W - side_) // 2
                full[y0_:y0_ + side_, x0_:x0_ + side_] = cv2.resize(m, (side_, side_), interpolation=cv2.INTER_NEAREST)
                m = full
            else:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = m
    rgb, side = _center_square(rgb)
    out = dict(rgb=cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA), scale=size / side)
    out["depth"] = None if depth is None else cv2.resize(_center_square(depth)[0], (size, size), interpolation=cv2.INTER_NEAREST)
    out["mask"] = None if mask is None else cv2.resize(_center_square(mask)[0], (size, size), interpolation=cv2.INTER_AREA) > 0.3
    return out


def fill_nearest(d):
    """Replace NaNs by the nearest valid value (Euclidean distance transform; O(N), very fast)."""
    m = ~np.isfinite(d)
    if m.all():
        return np.zeros_like(d)
    if not m.any():
        return d
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    return d[tuple(idx)]


def fill_depth_griddata(d, method="linear", stride=2):
    """Fill NaNs of a depth map by scattered-data interpolation (Delaunay linear / Clough-Tocher cubic), nearest as fall-back."""
    m = ~np.isfinite(d)
    if m.all() or not m.any():
        return fill_nearest(d)
    vv, uu = np.nonzero(~m)
    sel = np.arange(len(vv)) % stride == 0
    pts, vals = np.column_stack([uu[sel], vv[sel]]), d[vv[sel], uu[sel]]
    qv, qu = np.nonzero(m)
    est = interpolate.griddata(pts, vals, np.column_stack([qu, qv]), method=method)
    nn = interpolate.griddata(pts, vals, np.column_stack([qu, qv]), method="nearest")
    out = d.copy(); out[qv, qu] = np.where(np.isfinite(est), est, nn)
    return out


def depth_normalised(d):
    """Per-image normalised depth channel for ML: nearest-filled, median-centred, scaled by 0.5 m, clipped to +-3."""
    if d is None:
        return None
    f = fill_nearest(d)
    return np.clip((f - np.median(f)) / 0.5, -3, 3).astype(np.float32)


def depth_fill_experiment(records, cfg, n_img=12, size=64, block=6, frac_blocks=0.08, seed=0):
    """Validate depth-hole filling: hide blocks of KNOWN depth, refill with nearest / linear / cubic interpolation, report RMSE (mm) and time (ms) per method (Week 3)."""
    rng = np.random.default_rng(seed); rows = []
    cand = [r for r in records if r["depth"] is not None][:n_img]
    for rec in cand:
        d = load_sample(rec, size, cfg, need_skeleton=False)["depth"]
        valid = np.isfinite(d)
        if valid.mean() < 0.5:
            continue
        hide = np.zeros_like(valid)
        for _ in range(int(frac_blocks * size * size / block ** 2)):
            y, x = rng.integers(0, size - block, 2); hide[y:y + block, x:x + block] = True
        hide &= valid
        d_in = d.copy(); d_in[hide] = np.nan
        for method in ("nearest", "linear", "cubic"):
            t0 = time.perf_counter()
            f = fill_nearest(d_in) if method == "nearest" else fill_depth_griddata(d_in, method, stride=1)
            rows.append(dict(method=method, rmse_mm=1000 * np.sqrt(np.mean((f[hide] - d[hide]) ** 2)), time_ms=1000 * (time.perf_counter() - t0)))
    return pd.DataFrame(rows).groupby("method").mean().round(2)
