import numpy as np
import scipy.sparse.linalg

import npx


def test_cg():
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)

    sol, info = npx.cg(A, b)
    assert sol is not None
    assert info.success
    ref = np.array(
        [
            6.324555320336759e00,
            4.898979485566356e00,
            3.464101615137754e00,
            2.000000000000000e00,
            0.000000000000000e00,
        ]
    )
    tol = 1.0e-13
    assert np.all(np.abs(info.resnorms - ref) < tol * (1 + ref))

    # with "preconditioning"
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda x: 0.5 * x)
    sol, info = npx.cg(A, b, M=M)

    assert sol is not None
    assert info.success
    ref = np.array(
        [
            3.162277660168380e00,
            2.449489742783178e00,
            1.732050807568877e00,
            1.000000000000000e00,
            0.000000000000000e00,
        ]
    )
    tol = 1.0e-13
    assert np.all(np.abs(info.resnorms - ref) < tol * (1 + ref))


if __name__ == "__main__":
    test_cg()
