from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import scipy
import scipy.sparse.linalg

Info = namedtuple("KrylovInfo", ["success", "xk", "numsteps", "resnorms", "errnorms"])


def cg(
    A,
    b,
    x0=None,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

    # initial residual
    resnorms = []
    r = b - A @ x0
    Mr = r if M is None else M @ r
    resnorms.append(np.sqrt(np.dot(r, Mr)))

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(np.dot(err, err))]

    num_steps = 0

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

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

    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


def gmres(
    A,
    b,
    x0=None,
    tol: float = 1e-05,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

    # scipy.gmres() apparently calls the callback before the start of the iteration such
    # that the initial residual is automatically contained
    resnorms = []
    num_steps = -1

    if exact_solution is None:
        errnorms = None
    else:
        errnorms = []

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

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

    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


def minres(
    A,
    b,
    x0=None,
    shift: float = 0.0,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

    # initial residual
    resnorms = []
    r = b - A @ x0
    Mr = r if M is None else M @ r
    resnorms.append(np.sqrt(np.dot(r, Mr)))

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(np.dot(err, err))]

    num_steps = 0

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

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

    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)
