import numpy as np
import scipy.sparse.linalg

import npx


def _run(fun, resnorms1, resnorms2, tol=1.0e-13):
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)

    sol, info = fun(A, b)
    assert sol is not None
    assert info.success
    resnorms1 = np.asarray(resnorms1)
    for x in info.resnorms:
        print(f"{x:.15e}")
    print()
    assert np.all(np.abs(info.resnorms - resnorms1) < tol * (1 + resnorms1))

    # with "preconditioning"
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda x: 0.5 * x)
    sol, info = fun(A, b, M=M)

    assert sol is not None
    assert info.success
    resnorms2 = np.asarray(resnorms2)
    for x in info.resnorms:
        print(f"{x:.15e}")
    assert np.all(np.abs(info.resnorms - resnorms2) < tol * (1 + resnorms2))


def test_cg():
    _run(
        npx.cg,
        [
            6.324555320336759e00,
            4.898979485566356e00,
            3.464101615137754e00,
            2.000000000000000e00,
            0.000000000000000e00,
        ],
        [
            4.472135954999580e00,
            3.464101615137754e00,
            2.449489742783178e00,
            1.414213562373095e00,
            0.000000000000000e00,
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
            2.828427124746190e00,
            2.449489742783178e00,
            2.000000000000000e00,
            1.414213562373095e00,
            8.747542958250513e-15,
        ],
        [
            2.000000000000000e00,
            1.732050807568877e00,
            1.414213562373095e00,
            1.000000000000000e00,
            5.475099487534308e-15,
        ],
    )


if __name__ == "__main__":
    test_gmres()
