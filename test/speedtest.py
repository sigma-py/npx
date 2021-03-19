import numpy as np
import npx
import perfplot


def setup(n):
    a = np.random.rand(n)
    idx = np.random.randint(0, n, n)
    b = np.random.rand(n)
    return a, idx, b


def np_add_at(data):
    a, idx, b = data
    np.add.at(a, idx, b)
    return a


def npx_add_at(data):
    a, idx, b = data
    npx.add_at(a, idx, b)
    return a


perfplot.show(
    setup=setup,
    kernels=[np_add_at, npx_add_at],
    n_range=[2 ** k for k in range(23)]
)
