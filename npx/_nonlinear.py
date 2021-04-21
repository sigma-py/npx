from typing import Callable

import numpy as np


def _dot_last(a, b):
    a_shape = a.shape
    a = a.reshape(*a.shape[:-1], -1)
    b = b.reshape(*b.shape[:-1], -1)
    out = np.dot(a.T, b)
    out = out.reshape(a_shape[:-1])
    return out


def bisect(f: Callable, a, b, tol: float, max_num_steps=np.infty):
    a = np.asarray(a)
    b = np.asarray(b)

    fa = np.asarray(f(a))
    fb = np.asarray(f(b))

    assert np.all(np.logical_xor(fa > 0, fb > 0))
    is_fa_positive = fa > 0
    # sort points such that f(a) is negative, f(b) positive
    tmp = fa[is_fa_positive]
    fa[is_fa_positive] = fb[is_fa_positive]
    fb[is_fa_positive] = tmp
    #
    tmp = a[..., is_fa_positive]
    a[..., is_fa_positive] = b[..., is_fa_positive]
    b[..., is_fa_positive] = tmp

    k = 0
    while True:
        diff = a - b
        dist2 = _dot_last(diff, diff)

        if np.all(dist2 < tol ** 2):
            break

        if k >= max_num_steps:
            break

        c = (a + b) / 2
        fc = np.asarray(f(c))

        is_fc_positive = fc > 0
        a[..., ~is_fc_positive] = c[..., ~is_fc_positive]
        b[..., is_fc_positive] = c[..., is_fc_positive]

        k += 1

    return a, b


def regula_falsi(f: Callable, a, b, tol: float, max_num_steps=np.infty):
    a = np.asarray(a)
    b = np.asarray(b)
    fa = np.asarray(f(a))
    fb = np.asarray(f(b))

    assert np.all(np.logical_xor(fa > 0, fb > 0))
    # sort points such that f(a) is negative, f(b) positive
    is_fa_positive = fa > 0
    fa[is_fa_positive], fb[is_fa_positive] = fb[is_fa_positive], fa[is_fa_positive]
    a[..., is_fa_positive], b[..., is_fa_positive] = (
        b[..., is_fa_positive],
        a[..., is_fa_positive],
    )

    k = 0
    while True:
        if k >= max_num_steps:
            break

        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)

        is_fc_positive = fc > 0
        a[~is_fc_positive] = c[~is_fc_positive]
        b[is_fc_positive] = c[is_fc_positive]

        diff = a - b
        dist2 = _dot_last(diff, diff)

        if np.all(dist2 < tol ** 2):
            break

        k += 1

    return a, b
