import numpy as np

import npx

rng = np.random.default_rng(0)


def test_dot():
    a = rng.random(1, 2, 3)
    b = rng.random(3, 4, 5)
    c = npx.dot(a, b)
    assert c.shape == (1, 2, 4, 5)


def test_solve():
    a = rng.random(3, 3)
    b = rng.random(3, 4, 5)
    c = npx.solve(a, b)
    assert c.shape == b.shape


def test_outer():
    a = rng.random(1, 2)
    b = rng.random(3, 4)
    c = npx.outer(a, b)
    assert c.shape == (1, 2, 3, 4)
