"""Checks on the TreeID images bundled in ``data/TreeID`` (skipped if the folder is absent)."""
import pytest

from treeskel import get_config
from treeskel.data import discover_samples, repo_data_path

try:
    ROOT = repo_data_path()
except FileNotFoundError:  # pragma: no cover
    ROOT = None

pytestmark = pytest.mark.skipif(ROOT is None, reason="bundled data/TreeID not present")


def test_repo_mode_defaults():
    cfg = get_config()
    assert cfg["DATA_MODE"] == "repo" and cfg["GROUP_REGEX"] == r"(r\d+n\d+)"


def test_bundled_samples_complete_and_grouped():
    recs = discover_samples(ROOT, get_config())
    assert len(recs) == 400
    assert all(r["depth"] is not None and r["season"] == "summer" for r in recs)
    assert len({r["group"] for r in recs}) == 400          # one group per tree, no leakage across RGB/D/GT
