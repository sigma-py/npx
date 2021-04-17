from functools import reduce
from operator import mul

import numpy as np


# math.prod in 3.8
# https://docs.python.org/3/library/math.html#math.prod
def _prod(a):
    return reduce(mul, a, 1)


def dot(a, b):
    """Take arrays `a` and `b` and form the dot product between the last axis of `a` and
    the first of `b`.
    """
    b = np.asarray(b)
    return np.dot(a, b.reshape(b.shape[0], -1)).reshape(a.shape[:-1] + b.shape[1:])


def solve(A, x):
    """Solves a linear equation system with a matrix of shape (n, n) and an array of
    shape (n, ...). The output has the same shape as the second argument.
    """
    # https://stackoverflow.com/a/48387507/353337
    x = np.asarray(x)
    return np.linalg.solve(A, x.reshape(x.shape[0], -1)).reshape(x.shape)


def sum_at(a, indices, minlength: int):
    """Sums up values `a` with `indices` into an output array of at least length
    `minlength` while treating dimensionality correctly. It's a lot faster than numpy's
    own np.add.at (see
    https://github.com/numpy/numpy/issues/5922#issuecomment-511477435).

    Typically, `indices` will be a one-dimensional array; `a` can have any
    dimensionality. In this case, the output array will have shape (minlength,
    a.shape[1:]).

    `indices` may have arbitrary shape, too, but then `a` has to start out the same.
    (Those dimensions are flattened out in the computation.)
    """
    a = np.asarray(a)
    indices = np.asarray(indices)

    if len(a.shape) < len(indices.shape):
        raise RuntimeError(
            f"a.shape = {a.shape}, indices.shape = {indices.shape}, "
            "but len(a.shape) >= len(indices.shape) is required."
        )

    m = len(indices.shape)
    assert indices.shape == a.shape[:m]

    out_shape = (minlength, *a.shape[m:])

    indices = indices.reshape(-1)
    a = a.reshape(_prod(a.shape[:m]), _prod(a.shape[m:]))

    return np.array(
        [
            np.bincount(indices, weights=a[:, k], minlength=minlength)
            for k in range(a.shape[1])
        ]
    ).T.reshape(out_shape)


def add_at(a, indices, b):
    a = np.asarray(a)
    indices = np.asarray(indices)
    b = np.asarray(b)

    m = len(indices.shape)
    assert a.shape[1:] == b.shape[m:]
    a += sum_at(b, indices, a.shape[0])


def subtract_at(a, indices, b):
    b = np.asarray(b)
    add_at(a, indices, -b)


def unique_rows(a, return_inverse: bool = False, return_counts: bool = False):
    # The numpy alternative `np.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"Input array must be integer type, got {a.dtype}.")

    a_shape = a.shape
    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))

    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    out = np.unique(b, return_inverse=return_inverse, return_counts=return_counts)
    # out[0] are the sorted, unique rows
    if isinstance(out, tuple):
        out = (out[0].view(a.dtype).reshape(out[0].shape[0], *a_shape[1:]), *out[1:])
    else:
        out = out.view(a.dtype).reshape(out.shape[0], *a_shape[1:])

    return out
