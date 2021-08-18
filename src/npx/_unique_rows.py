from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def unique_rows(
    a: ArrayLike, return_inverse: bool = False, return_counts: bool = False
) -> Union[np.ndarray, Tuple]:
    # The numpy alternative `np.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    a = np.asarray(a)

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
