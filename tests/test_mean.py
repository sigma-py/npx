import numpy as np
import pytest

import npx


@pytest.mark.parametrize(
    "p,ref",
    [
        (-np.inf, 1.0),  # min
        (-20000, 1.0000693171203765),
        (-1, 1.9672131147540985),  # harmonic mean
        (-0.1, 2.3000150292740735),
        (0, 2.3403473193207156),  # geometric mean
        (0.1, 2.3810581190184337),
        (1, 2.75),  # arithmetic mean
        (2, np.sqrt(9.75)),  # root mean square
        (10000, 4.999306900862521),
        (np.inf, 5.0),  # max
    ],
)
def test_mean_pos(p, ref):
    a = [1.0, 2.0, 3.0, 5.0]
    val = npx.mean(a, p)
    print(p, val)
    assert abs(val - ref) < 1.0e-13 * abs(ref)


@pytest.mark.parametrize(
    "p,ref",
    [
        (-np.inf, 1.0),  # absmin
        (-1, -1.9672131147540985),  # harmonic mean
        # (0, 2.3403473193207156), # geometric mean
        (1, -2.75),  # arithmetic mean
        (2, np.sqrt(9.75)),  # root mean square
        (np.inf, 5.0),  # absmax
    ],
)
def test_mean_neg(p, ref):
    a = [-1.0, -2.0, -3.0, -5.0]
    val = npx.mean(a, p)
    print(p, val)
    assert abs(val - ref) < 1.0e-13 * abs(ref)


def test_errors():
    a = [-1.0, -2.0, -3.0, -5.0]
    with pytest.raises(ValueError):
        npx.mean(a, 0.5)

    with pytest.raises(ValueError):
        npx.mean(a, 0)
