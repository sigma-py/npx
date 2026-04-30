# /// script
# dependencies = [
#   "npx",
#   "numpy",
#   "perfplot",
# ]
# ///
import numpy as np
import perfplot

import npx

rng = np.random.default_rng(0)

m = 100


def setup(n):
    idx = rng.integers(0, m, size=n)
    b = rng.random(n)
    return idx, b


def np_add_at(idx, b):
    a = np.zeros(m)
    np.add.at(a, idx, b)
    return a


def npx_add_at(idx, b):
    a = np.zeros(m)
    npx.add_at(a, idx, b)
    return a


def npx_sum_at(idx, b):
    return npx.sum_at(b, idx, minlength=m)


def bincount(idx, b):
    return np.bincount(idx, weights=b, minlength=m)


b = perfplot.bench(
    setup=setup,
    kernels=[np_add_at, npx_add_at, npx_sum_at, bincount],
    n_range=[2**k for k in range(23)],
)
b.save("perf-add-at.png", transparent=False)
