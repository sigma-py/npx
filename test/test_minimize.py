import numpy as np
import pytest

import npx


def test_0d():
    def f(x):
        return (x ** 2 - 2) ** 2

    x0 = 1.5
    out = npx.minimize(f, x0)

    assert out.x.shape == np.asarray(x0).shape
    assert np.asarray(out.fun).shape == ()


def test_2d():
    def f(x):
        return (np.sum(x ** 2) - 2) ** 2

    x0 = np.ones((4, 3), dtype=float)
    out = npx.minimize(f, x0, method="Powell")

    assert out.x.shape == np.asarray(x0).shape
    assert np.asarray(out.fun).shape == ()


def test_error():
    def f(x):
        return x - 2

    x0 = [1.5]
    with pytest.raises(ValueError):
        npx.minimize(f, x0)
