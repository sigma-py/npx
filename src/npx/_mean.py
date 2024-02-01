import numpy as np
from numpy.typing import ArrayLike


# There also is
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html>,
# but implementation is easy enough
def _logsumexp(x: ArrayLike):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def mean(x: ArrayLike, p: float = 1) -> np.ndarray:
    """Generalized mean.

    See <https://github.com/numpy/numpy/issues/19341> for the numpy issue.
    """
    x = np.asarray(x)

    n = len(x)
    if p == 1:
        return np.mean(x)

    if p == -np.inf:
        return np.min(np.abs(x))

    if p == 0:
        # first compute the root, then the product, to avoid numerical
        # difficulties with too small values of prod(x)
        if np.any(x < 0.0):
            msg = "p=0 only works with nonnegative x."
            raise ValueError(msg)
        return np.prod(np.power(x, 1 / n))
        # alternative:
        # return np.exp(np.mean(np.log(x)))

    if p == np.inf:
        return np.max(np.abs(x))

    if np.all(x > 0.0):
        # logsumexp trick to avoid overflow for large p
        # only works for positive x though
        return np.exp((_logsumexp(p * np.log(x)) - np.log(n)) / p)

    if not isinstance(p, (int, np.integer)):
        msg = f"Non-integer p (={p}) only work with nonnegative x."
        raise TypeError(msg)

    return (np.sum(x**p) / n) ** (1.0 / p)
