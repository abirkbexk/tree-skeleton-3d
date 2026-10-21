"""Classical-ML feature extraction and grouped train/validation/test splitting (Week 5: training-data evaluation)."""
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def featurise(rgb, dnorm, fsize):
    """Classical-ML features: area-down-sampled RGB (3) + normalised depth (1) channels, flattened. -> (N, 4*fsize*fsize) float32"""
    out = np.empty((len(rgb), 4 * fsize * fsize), np.float32)
    for i in range(len(rgb)):
        a = cv2.resize(rgb[i], (fsize, fsize), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        d = cv2.resize(dnorm[i].astype(np.float32), (fsize, fsize), interpolation=cv2.INTER_AREA)
        out[i] = np.concatenate([a.reshape(-1), d.reshape(-1)])
    return out


def make_splits(groups, seed, frac_val=0.15, frac_test=0.15):
    """Grouped random split into (train, val, test) index arrays."""
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=frac_test, random_state=seed)
    trv, te = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=frac_val / (1 - frac_test), random_state=seed + 1)
    tr, va = next(gss2.split(trv, groups=groups[trv]))
    return trv[tr], trv[va], te
