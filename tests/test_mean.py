import npx
import numpy as np

import pytest


@pytest.mark.parametrize(
    "p,ref", [
    (-np.inf, 1.0),  # min
    (-1, 1.9672131147540985),  # harmonic mean
    (0, 2.3403473193207156), # geometric mean
    (1, 2.75), # arithmetic mean
    (2, np.sqrt(9.75)),  # root mean square
    (np.inf, 5.0),  # max
])
def test_mean_pos(p, ref):
    a = [1.0, 2.0, 3.0, 5.0]
    val = npx.mean(a, p)
    assert abs(val - ref) < 1.0e-13 * abs(ref)


@pytest.mark.parametrize(
    "p,ref", [
    (-np.inf, 1.0),  # absmin
    # (-1, -6.315789473684211),  # harmonic mean
    # (0, 2.3403473193207156), # geometric mean
    (1, -2.75), # arithmetic mean
    (2, np.sqrt(9.75)),  # root mean square
    (np.inf, 5.0),  # absmax
])
def test_mean_neg(p, ref):
    a = [-1.0, -2.0, -3.0, -5.0]
    print(a)
    print(p)
    val = npx.mean(a, p)
    print(val)
    assert abs(val - ref) < 1.0e-13 * abs(ref)
