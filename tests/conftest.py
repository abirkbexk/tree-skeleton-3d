"""Shared fixtures: a tiny synthetic dataset and a quick configuration (no real data needed)."""
import numpy as np
import pytest

from treeskel import get_config
from treeskel.data import discover_samples
from treeskel.dataset import assemble_dataset
from treeskel.synthetic import write_synthetic_dataset


@pytest.fixture(scope="session")
def cfg():
    return get_config(QUICK_TEST=True, RUN_CNN=False, DATA_MODE="synthetic", IMG_SIZE=64, FEAT_SIZE=16, N_PCA_FEAT=20, N_PCA_TARGET=8, N_SYNTH=48)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory, cfg):
    return write_synthetic_dataset(tmp_path_factory.mktemp("synth"), cfg["N_SYNTH"], seed=0)


@pytest.fixture(scope="session")
def records(synth_dir, cfg):
    return discover_samples(synth_dir, cfg)


@pytest.fixture(scope="session")
def ds(records, cfg):
    return assemble_dataset(records, cfg)
