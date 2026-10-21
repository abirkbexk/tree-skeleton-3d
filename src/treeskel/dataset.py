"""Dataset assembly: load every (RGB, depth, skeleton) record into arrays and derive regression targets."""
import numpy as np
from tqdm.auto import tqdm

from .data import depth_normalised, load_sample
from .skeleton import fill_rows, skeleton_to_rows


def build_dataset(records, cfg, min_rows=10):
    """Load all records into arrays. Samples with unreadable files or < min_rows skeleton rows are rejected (and counted)."""
    S = cfg["IMG_SIZE"]
    keep, rgb, depth, rows, valid, scale, rejected = [], [], [], [], [], [], []
    for rec in tqdm(records, desc="loading"):
        try:
            s = load_sample(rec, S, cfg)
            r, v = skeleton_to_rows(s["mask"])
            if v.sum() < min_rows:
                raise ValueError(f"only {int(v.sum())} skeleton rows")
        except Exception as e:                                      # invalid input handling: skip + report
            rejected.append((str(rec["rgb"].name), str(e))); continue
        keep.append(rec); rgb.append(s["rgb"]); scale.append(s["scale"])
        depth.append(np.full((S, S), np.nan, np.float32) if s["depth"] is None else s["depth"])
        rows.append(np.stack([fill_rows(r[0], v), fill_rows(r[1], v)])); valid.append(v)
    print(f"loaded {len(keep)} samples, rejected {len(rejected)}", rejected[:3])
    has_depth = np.array([r["depth"] is not None or cfg["FOURTH_CHANNEL_IS_DEPTH"] for r in keep])
    return dict(rgb=np.stack(rgb), depth=np.stack(depth), rows=np.stack(rows).astype(np.float32), valid=np.stack(valid),
                scale=np.array(scale), has_depth=has_depth, season=np.array([r["season"] for r in keep]),
                group=np.array([r["group"] for r in keep]), key=np.array([r["key"] for r in keep]))


def assemble_dataset(records, cfg):
    """Build the full in-memory dataset used for training and evaluation.

    Args:
        records: list of record dicts from :func:`treeskel.data.discover_samples`.
        cfg: configuration dict (see :mod:`treeskel.config`).

    Returns:
        dict with everything from :func:`build_dataset` plus

        * ``dnorm`` (N,S,S) float16: per-image normalised depth channel (zeros when no depth),
        * ``targets`` (N,2S) float32: branch x-positions divided by ``S-1``, layout ``[left(S), right(S)]``,
        * ``extent`` (N,2) float32: first/last annotated row divided by ``S-1``.
    """
    S = cfg["IMG_SIZE"]
    ds = build_dataset(records, cfg)
    n = len(ds["rgb"])
    ds["dnorm"] = np.stack([depth_normalised(d) if h else np.zeros((S, S), np.float32)
                            for d, h in zip(ds["depth"], ds["has_depth"])]).astype(np.float16)
    ds["targets"] = (ds["rows"] / (S - 1)).reshape(n, 2 * S).astype(np.float32)
    ds["extent"] = np.array([[np.flatnonzero(v)[0], np.flatnonzero(v)[-1]] for v in ds["valid"]], dtype=np.float32) / (S - 1)
    return ds
