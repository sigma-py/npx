import numpy as np
from numpy.typing import ArrayLike


def isin_rows(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"Input array must be integer type, got {a.dtype}.")
    if not np.issubdtype(b.dtype, np.integer):
        raise ValueError(f"Input array must be integer type, got {b.dtype}.")

    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))
    b = b.reshape(b.shape[0], np.prod(b.shape[1:], dtype=int))

    a_view = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    )
    b_view = np.ascontiguousarray(b).view(
        np.dtype((np.void, b.dtype.itemsize * b.shape[1]))
    )

    out = np.isin(a_view, b_view)

    return out.reshape(a.shape[0])
