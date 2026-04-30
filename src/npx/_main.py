from math import prod

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import deprecated


def dot(a: ArrayLike, b: np.ndarray) -> np.ndarray:
    """Take arrays `a` and `b` and form the dot product between the last axis
    of `a` and the first of `b`.
    """
    return np.tensordot(a, b, 1)


def outer(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Compute the outer product of two arrays `a` and `b` such that the shape
    of the resulting array is `(*a.shape, *b.shape)`.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.outer(a, b).reshape(*a.shape, *b.shape)


def solve(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Solves a linear equation system with a matrix of shape (n, n) and an array of
    shape (n, ...). The output has the same shape as the second argument.
    """
    # https://stackoverflow.com/a/48387507/353337
    x = np.asarray(x)
    return np.linalg.solve(A, x.reshape(x.shape[0], -1)).reshape(x.shape)


@deprecated("np.add.at is now equally fast. Use that.")
def sum_at(a: ArrayLike, indices: ArrayLike, minlength: int):
    """Sums up values `a` with `indices` into an output array of at least
    length `minlength` while treating dimensionality correctly. It's a lot
    faster than numpy's own np.add.at (see
    https://github.com/numpy/numpy/issues/5922#issuecomment-511477435).

    Typically, `indices` will be a one-dimensional array; `a` can have any
    dimensionality. In this case, the output array will have shape (minlength,
    a.shape[1:]).

    `indices` may have arbitrary shape, too, but then `a` has to start out the
    same. (Those dimensions are flattened out in the computation.)
    """
    return _sum_at(a, indices, minlength)


def _sum_at(a: ArrayLike, indices: ArrayLike, minlength: int):
    a = np.asarray(a)
    indices = np.asarray(indices)

    if len(a.shape) < len(indices.shape):
        msg = (
            f"a.shape = {a.shape}, indices.shape = {indices.shape}, "
            "but len(a.shape) >= len(indices.shape) is required."
        )
        raise RuntimeError(msg)

    m = len(indices.shape)
    assert indices.shape == a.shape[:m]

    out_shape = (minlength, *a.shape[m:])

    indices = indices.reshape(-1)
    a = a.reshape(prod(a.shape[:m]), prod(a.shape[m:]))

    # Cast to int; bincount doesn't work for uint64 yet
    # https://github.com/numpy/numpy/issues/17760
    indices = indices.astype(int)

    return np.array(
        [
            np.bincount(indices, weights=a[:, k], minlength=minlength)
            for k in range(a.shape[1])
        ],
    ).T.reshape(out_shape)


@deprecated("np.add.at is now equally fast. Use that.")
def add_at(a: ArrayLike, indices: ArrayLike, b: ArrayLike):
    return _add_at(a, indices, b)


def _add_at(a: ArrayLike, indices: ArrayLike, b: ArrayLike):
    a = np.asarray(a)
    indices = np.asarray(indices)
    b = np.asarray(b)

    m = len(indices.shape)
    assert a.shape[1:] == b.shape[m:]
    a += _sum_at(b, indices, a.shape[0])


@deprecated("np.subtract.at is now equally fast. Use that.")
def subtract_at(a: ArrayLike, indices: ArrayLike, b: ArrayLike):
    b = np.asarray(b)
    _add_at(a, indices, -b)
