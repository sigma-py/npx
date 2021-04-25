import numpy as np
import scipy.sparse.linalg

import npx


def _run(method, resnorms1, resnorms2, tol=1.0e-13):
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)

    exact_solution = scipy.sparse.linalg.spsolve(A, b)

    x0 = np.zeros(A.shape[1])
    sol, info = method(A, b, x0, exact_solution=exact_solution, callback=lambda _: None)
    assert sol is not None
    assert info.success
    print(info)
    assert len(info.resnorms) == info.numsteps + 1
    assert len(info.errnorms) == info.numsteps + 1
    print(info.resnorms)
    print()
    resnorms1 = np.asarray(resnorms1)
    assert np.all(np.abs(info.resnorms - resnorms1) < tol * (1 + resnorms1))
    # make sure the initial resnorm and errnorm are correct
    assert abs(np.linalg.norm(A @ x0 - b, 2) - info.resnorms[0]) < 1.0e-13
    assert abs(np.linalg.norm(x0 - exact_solution, 2) - info.errnorms[0]) < 1.0e-13

    # with "preconditioning"
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda x: 0.5 * x)
    sol, info = method(A, b, M=M)

    assert sol is not None
    assert info.success
    print(info.resnorms)
    resnorms2 = np.asarray(resnorms2)
    assert np.all(np.abs(info.resnorms - resnorms2) < tol * (1 + resnorms2))


def test_cg():
    _run(
        npx.cg,
        [
            3.1622776601683795,
            6.324555320336759,
            4.898979485566356,
            3.4641016151377544,
            2.0,
            0.0,
        ],
        [
            2.23606797749979,
            4.47213595499958,
            3.4641016151377544,
            2.449489742783178,
            1.4142135623730951,
            0.0,
        ],
    )


def test_gmres():
    _run(
        npx.gmres,
        [3.162277660168380e00, 7.160723346098895e-15],
        [2.236067977499790e00, 5.063396036227354e-15],
    )


def test_minres():
    _run(
        npx.minres,
        [
            3.1622776601683795,
            2.8284271247461903,
            2.449489742783178,
            2.0,
            1.4142135623730951,
            8.747542958250513e-15,
        ],
        [
            2.23606797749979,
            2.0,
            1.7320508075688772,
            1.4142135623730951,
            1.0,
            5.475099487534308e-15,
        ],
    )


if __name__ == "__main__":
    test_gmres()
