from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


def _unique_tol(
    unique_fun: Callable, a: ArrayLike, tol: float, **kwargs
) -> np.ndarray | tuple[np.ndarray, ...]:
    a = np.asarray(a)
    # compute 1/tol first. Difference:
    #
    #   int(3.3 / 0.1) = int(32.99999999999999) = 32
    #   int(3.3 * (1.0 / 0.1)) = int(33.0) = 33
    #
    aint = (a * (1.0 / tol)).astype(int)

    return_index = kwargs.pop("return_index", False)

    _, idx, *out = unique_fun(aint, return_index=True, **kwargs)
    unique_a = a[idx]

    if return_index:
        out = [idx, *out]

    if len(out) == 0:
        return unique_a

    return (unique_a, *out)


def unique(
    a: ArrayLike, tol: float = 0.0, **kwargs
) -> np.ndarray | tuple[np.ndarray, ...]:
    assert tol >= 0.0
    if tol > 0.0:
        return _unique_tol(np.unique, a, tol, **kwargs)

    return np.unique(a, **kwargs)


def unique_rows(
    a: ArrayLike,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    # The numpy alternative `np.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    a = np.asarray(a)

    a_shape = a.shape
    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))

    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    out = np.unique(
        b,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    # out[0] are the sorted, unique rows
    if isinstance(out, tuple):
        out = (out[0].view(a.dtype).reshape(out[0].shape[0], *a_shape[1:]), *out[1:])
    else:
        out = out.view(a.dtype).reshape(out.shape[0], *a_shape[1:])

    return out
