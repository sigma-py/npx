import math
import numpy as np


def add_at(a, idx, minlength: int):
    """A fancy (and correct) way of summing up vals into an array of out_shape according
    to idx. np.add.at is thought out for this, but is really slow. np.bincount is a lot
    faster (https://github.com/numpy/numpy/issues/5922#issuecomment-511477435), but
    doesn't handle dimensionality. This function does.

    vals has to have shape (idx.shape, ...),
    """
    assert len(a.shape) >= len(idx.shape)
    m = len(idx.shape)
    assert idx.shape == a.shape[:m]

    out_shape = (minlength, *a.shape[m:])

    idx = idx.reshape(-1)
    a = a.reshape(math.prod(a.shape[:m]), math.prod(a.shape[m:]))

    return np.array(
        [
            np.bincount(idx, weights=a[:, k], minlength=minlength)
            for k in range(a.shape[1])
        ]
    ).T.reshape(out_shape)


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
