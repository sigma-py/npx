import numpy as np

import npx


def test_1d():
    def f(x):
        return (x ** 2 - 2) ** 2

    x0 = 1.5
    out = npx.minimize(f, x0)

    assert out.x.shape == np.asarray(x0).shape
    assert np.asarray(out.fun).shape == ()
