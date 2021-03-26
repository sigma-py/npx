from collections import namedtuple
from typing import Optional

import numpy as np
import scipy
import scipy.sparse.linalg

Info = namedtuple("KrylovInfo", ["success", "xk", "resnorms", "errnorms"])


def cg(
    A,
    b,
    x0=None,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M=None,
    callback=None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    resnorms = []

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(np.dot(err, err))]

    def cb(xk):
        if callback is not None:
            callback(xk)

        r = b - A @ xk
        Mr = r if M is None else M @ r
        resnorms.append(np.sqrt(np.dot(r, Mr)))

        if exact_solution is not None:
            err = exact_solution - x0
            errnorms.append(np.sqrt(np.dot(err, err)))

    x, info = scipy.sparse.linalg.cg(
        A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, atol=atol, callback=cb
    )

    success = info == 0

    resnorms = np.array(resnorms)
    if errnorms is not None:
        errnorms = np.array(errnorms)

    return x if success else None, Info(success, x, resnorms, errnorms)


def gmres(
    A,
    b,
    x0=None,
    tol: float = 1e-05,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback=None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    resnorms = []

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(np.dot(err, err))]

    def cb(xk):
        if callback is not None:
            callback(xk)

        r = b - A @ xk
        Mr = r if M is None else M @ r
        resnorms.append(np.sqrt(np.dot(r, Mr)))

        if exact_solution is not None:
            err = exact_solution - x0
            errnorms.append(np.sqrt(np.dot(err, err)))

    x, info = scipy.sparse.linalg.gmres(
        A,
        b,
        x0=x0,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        atol=atol,
        callback=cb,
        callback_type="x",
    )

    success = info == 0

    resnorms = np.array(resnorms)
    if errnorms is not None:
        errnorms = np.array(errnorms)

    return x if success else None, Info(success, x, resnorms, errnorms)


def minres(
    A,
    b,
    x0=None,
    shift: float = 0.0,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M=None,
    callback=None,
    exact_solution=None,
):
    resnorms = []

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(np.dot(err, err))]

    def cb(xk):
        if callback is not None:
            callback(xk)

        r = b - A @ xk
        Mr = r if M is None else M @ r
        resnorms.append(np.sqrt(np.dot(r, Mr)))

        if exact_solution is not None:
            err = exact_solution - x0
            errnorms.append(np.sqrt(np.dot(err, err)))

    x, info = scipy.sparse.linalg.minres(
        A, b, x0=x0, shift=shift, tol=tol, maxiter=maxiter, M=M, callback=cb
    )

    success = info == 0

    resnorms = np.array(resnorms)
    if errnorms is not None:
        errnorms = np.array(errnorms)

    return x if success else None, Info(success, x, resnorms, errnorms)
