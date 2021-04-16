import numpy as np
import pytest

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


def test_add_at():
    a = [1.0, 2.0, 3.0]
    idx = [0, 1, 0]
    out = np.zeros(2)
    npx.add_at(out, idx, a)

    tol = 1.0e-13
    ref = np.array([4.0, 2.0])
    assert np.all(np.abs(out - ref) < (1 + np.abs(ref)) * tol)


def test_subtract_at():
    a = [1.0, 2.0, 3.0]
    idx = [0, 1, 0]
    out = np.ones(2)
    npx.subtract_at(out, idx, a)

    tol = 1.0e-13
    ref = np.array([-3.0, -1.0])
    assert np.all(np.abs(out - ref) < (1 + np.abs(ref)) * tol)


def test_unique_rows():
    a = [1, 2, 1]
    a_unique = npx.unique_rows(a)
    assert np.all(a_unique == [1, 2])

    a = [[1, 2], [1, 4], [1, 2]]
    a_unique, inv, count = npx.unique_rows(a, return_inverse=True, return_counts=True)
    assert np.all(a_unique == [[1, 2], [1, 4]])
    assert np.all(inv == [0, 1, 0])
    assert np.all(count == [2, 1])

    a_unique = npx.unique_rows(a)
    assert np.all(a_unique == [[1, 2], [1, 4]])

    # entries are matrices
    # fails for some reason. keep an eye on
    # <https://stackoverflow.com/q/67128631/353337>
    # a = [[[3, 4], [-1, 2]], [[3, 4], [-1, 2]]]
    # a_unique = npx.unique_rows(a)
    # print(a_unique)
    # assert np.all(a_unique == [[[3, 4], [-1, 2]]])

    a = [1.1, 2.2]
    with pytest.raises(ValueError):
        a_unique = npx.unique_rows(a)
