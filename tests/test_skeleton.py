import numpy as np

from treeskel.skeleton import fill_rows, junction_row, skeleton_to_rows


def test_single_and_double_run():
    m = np.zeros((40, 100), bool); m[5:30, 48:53] = True          # run centre = 50
    rows, valid = skeleton_to_rows(m)
    assert valid.sum() == 25 and np.allclose(rows[:, 5:30], 50.0)
    m[5:15, 70:75] = True
    rows, _ = skeleton_to_rows(m)
    assert np.allclose(rows[0, 5:15], 50.0) and np.allclose(rows[1, 5:15], 72.0)


def test_fill_rows_reproduces_linear_profile():
    x = np.linspace(0, 10, 30); valid = np.ones(30, bool); valid[10:14] = False; valid[:3] = False
    out = fill_rows(np.where(valid, x, np.nan), valid)
    assert np.allclose(out[3:], x[3:]) and np.allclose(out[:3], x[3])


def test_fill_rows_without_valid_rows_is_nan():
    assert np.isnan(fill_rows(np.full(5, np.nan), np.zeros(5, bool))).all()


def test_junction_row():
    left = np.array([10, 10, 10, 8, 5.0]); right = np.array([10, 10, 10, 12, 15.0])
    assert junction_row(left, right, 3.0) == 4.0 and np.isnan(junction_row(left, left, 3.0))
