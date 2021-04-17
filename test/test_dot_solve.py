import numpy as np

import npx

np.random.seed(0)


def test_dot():
    a = np.random.rand(1, 2, 3)
    b = np.random.rand(3, 4, 5)
    c = npx.dot(a, b)
    assert c.shape == (1, 2, 4, 5)


def test_solve():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 4, 5)
    c = npx.solve(a, b)
    assert c.shape == b.shape
