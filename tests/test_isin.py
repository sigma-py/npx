import numpy as np

import npx


def test_isin():
    a = [[0, 3], [1, 0]]
    b = [[1, 0], [7, 12], [-1, 5]]

    out = npx.isin_rows(a, b)
    assert np.all(out == [False, True])


def test_scalar():
    a = [0, 3, 5]
    b = [-1, 6, 5, 0, 0, 0]

    out = npx.isin_rows(a, b)
    assert np.all(out == [True, False, True])
