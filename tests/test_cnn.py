import numpy as np
import pytest

torch = pytest.importorskip("torch")
from treeskel.cnn import build_cnn, make_batch, predict_cnn


def test_flip_batch_swaps_and_mirrors_targets(ds, cfg):
    S = cfg["IMG_SIZE"]; idx = np.arange(3); flip = np.array([True, False, True])
    x, y, m = make_batch(ds, idx, S, flip=flip); x0, y0, m0 = make_batch(ds, idx, S)
    assert torch.allclose(y[0, :S], 1 - y0[0, S:], atol=1e-6) and torch.allclose(y[1], y0[1]) and torch.allclose(x[0], torch.flip(x0[0], dims=[2]))
    x2, y2, _ = make_batch(ds, idx, S, flip=flip); assert torch.equal(y, y2)


def test_cnn_output_shape_and_bias_init(ds, cfg):
    S = cfg["IMG_SIZE"]; net = build_cnn(2 * S, bias_init=ds["targets"].mean(0))
    out = predict_cnn(net, ds["rgb"][:3], ds["dnorm"][:3], S); assert out.shape == (3, 2, S) and np.isfinite(out).all()
