"""Small convolutional regressor (PyTorch) mapping RGB-D images to the 2*S branch positions.

PyTorch is imported lazily so the rest of the package works without it. Training uses a *masked* MSE (only annotated rows
contribute), random horizontal flips as augmentation, ReduceLROnPlateau and early stopping on validation RMSE.
"""
from __future__ import annotations

import time

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyTorch is required for the CNN (pip install torch) - or set RUN_CNN=False") from e
    return torch, nn


def build_cnn(n_out, bias_init=None):
    """Conv stem + 4 conv-BN-ReLU-pool blocks + adaptive pooling + 2-layer regression head. Output: ``n_out`` values (normalised x / (S-1))."""
    torch, nn = _torch()

    def blk(i, o):
        """conv3x3 -> batch-norm -> ReLU -> 2x2 max-pool block."""
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    net = nn.Sequential(nn.Conv2d(4, 16, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                        blk(16, 32), blk(32, 64), blk(64, 96), blk(96, 128), nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(),
                        nn.Linear(128 * 36, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_out))
    if bias_init is not None:
        with torch.no_grad():
            net[-1].bias.copy_(torch.as_tensor(bias_init, dtype=torch.float32))
    return net


def make_batch(ds, idx, size, flip=None):
    """Numpy dataset -> torch tensors ``x (B,4,S,S)``, targets ``y (B,2S)``, masks ``m (B,2S)``.

    ``flip`` (bool array): mirror those images; left/right targets swap and map ``t -> 1 - t`` (exact because targets are x/(S-1))."""
    torch, _ = _torch()
    x = np.concatenate([ds["rgb"][idx].astype(np.float32) / 255.0, ds["dnorm"][idx].astype(np.float32)[..., None]], -1)
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(ds["targets"][idx]).clone()
    m = torch.from_numpy(np.concatenate([ds["valid"][idx], ds["valid"][idx]], axis=1).astype(np.float32))
    if flip is not None and flip.any():
        f = torch.from_numpy(flip); S = size
        x[f] = torch.flip(x[f], dims=[3])
        yl, yr = y[f, :S].clone(), y[f, S:].clone()
        y[f, :S], y[f, S:] = 1 - yr, 1 - yl
        ml, mr = m[f, :S].clone(), m[f, S:].clone()
        m[f, :S], m[f, S:] = mr, ml
    return x, y, m


def predict_cnn(net, rgb, dnorm, size, device="cpu", batch=64):
    """Predicted branch positions in pixels, shape (N, 2, size), for ``rgb (N,S,S,3) uint8`` and ``dnorm (N,S,S)``."""
    torch, _ = _torch()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(rgb), batch):
            x = np.concatenate([rgb[i:i + batch].astype(np.float32) / 255.0, dnorm[i:i + batch].astype(np.float32)[..., None]], -1)
            out.append(net(torch.from_numpy(x).permute(0, 3, 1, 2).to(device)).cpu().numpy())
    return np.concatenate(out).reshape(-1, 2, size) * (size - 1)


def train_cnn(ds, train_idx, val_idx, cfg, device=None, verbose=True):
    """Train the CNN with early stopping.

    Returns:
        (net with best-validation weights loaded, history array [(epoch, train_rmse_px, val_rmse_px)], best_epoch, fit_seconds).
    """
    torch, _ = _torch()
    S = cfg["IMG_SIZE"]; norm = S - 1
    torch.manual_seed(cfg["SEED"]); device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = build_cnn(2 * S, bias_init=ds["targets"][train_idx].mean(0)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["CNN_LR"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    rng = np.random.default_rng(cfg["SEED"]); best = (1e9, None, 0); hist = []; t0 = time.perf_counter(); bs = cfg["CNN_BATCH"]
    for epoch in range(cfg["CNN_EPOCHS"]):
        net.train(); perm = rng.permutation(train_idx); tl = []
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            x, y, m = make_batch(ds, b, S, flip=rng.random(len(b)) < 0.5)
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = (m * (net(x) - y) ** 2).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        pv = predict_cnn(net, ds["rgb"][val_idx], ds["dnorm"][val_idx], S, device); v = ds["valid"][val_idx][:, None, :]
        val_rmse = float(np.sqrt(np.sum(((pv - ds["rows"][val_idx]) ** 2) * v) / (2 * v.sum())))
        sched.step(val_rmse); hist.append((epoch, float(np.sqrt(np.mean(tl)) * norm) if tl else np.nan, val_rmse))
        if val_rmse < best[0]:
            best = (val_rmse, {k: t.detach().clone() for k, t in net.state_dict().items()}, epoch)
        if epoch - best[2] >= 12:
            if verbose: print("early stop at epoch", epoch)
            break
        if verbose and (epoch % 5 == 0 or epoch == cfg["CNN_EPOCHS"] - 1):
            print(f"epoch {epoch:3d}  train RMSE {hist[-1][1]:6.2f}px   val RMSE {val_rmse:6.2f}px")
    net.load_state_dict(best[1])
    return net, np.array(hist), best[2], time.perf_counter() - t0


def count_flops_and_params(net, size, device="cpu"):
    """Analytic forward-pass FLOPs (2 per multiply-add) of conv and linear layers for one image, and parameter count."""
    torch, nn = _torch()
    fl = [0]; hooks = []
    for mod in net.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(lambda m_, i_, o_: fl.__setitem__(0, fl[0] + 2 * o_.numel() * (m_.in_channels // m_.groups) * m_.kernel_size[0] * m_.kernel_size[1])))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m_, i_, o_: fl.__setitem__(0, fl[0] + 2 * m_.in_features * m_.out_features)))
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, 4, size, size, device=device))
    for h in hooks:
        h.remove()
    return fl[0], sum(p.numel() for p in net.parameters())
