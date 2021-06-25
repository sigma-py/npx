import numpy as np
import numpy.typing as npt


def mean(x: npt.ArrayLike, p: float = 1):
    x = np.asarray(x)
    n = len(x)
    if p == 1:
        return np.mean(x)
    elif p == -np.inf:
        return np.min(np.abs(x))
    elif p == 0:
        # first compute the root, then the product, to avoid numerical difficulties with
        # too small values of prod(x)
        return np.prod(np.power(x, 1 / n))
    elif p == np.inf:
        return np.max(np.abs(x))

    return (np.sum(x ** p) / n) ** (1.0 / p)
