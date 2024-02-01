import numpy as np
import perfplot

import npx

rng = np.random.default_rng(0)

m = 100


def setup(n):
    idx = rng.randomint(0, m, n)
    b = rng.random(n)
    return idx, b


def np_add_at(data):
    a = np.zeros(m)
    idx, b = data
    np.add.at(a, idx, b)
    return a


def npx_add_at(data):
    a = np.zeros(m)
    idx, b = data
    npx.add_at(a, idx, b)
    return a


def npx_sum_at(data):
    idx, b = data
    return npx.sum_at(b, idx, minlength=m)


b = perfplot.bench(
    setup=setup,
    kernels=[np_add_at, npx_add_at, npx_sum_at],
    n_range=[2**k for k in range(23)],
)
b.save("perf-add-at.svg")
