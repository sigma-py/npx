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


def test_sum_at():
    a = [1.0, 2.0, 3.0]
    idx = [0, 1, 0]
    out = npx.sum_at(a, idx, minlength=4)

    tol = 1.0e-13
    ref = np.array([4.0, 2.0, 0.0, 0.0])
    assert np.all(np.abs(out - ref) < (1 + np.abs(ref)) * tol)
