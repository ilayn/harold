from numpy import eye, array, sort
from scipy.linalg import block_diag, eigvals
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import raises as assert_raises
from harold import lqr, ackermann, State, Transfer, haroldcompanion


def test_lqr_arguments():
    # First arg is not LTI
    assert_raises(ValueError, lqr, 1, 1)
    # Static Gain
    assert_raises(ValueError, lqr, State(1), 1)
    # Wrong string
    assert_raises(ValueError, lqr, Transfer(1, [1, 1]), 1, weight_on='asdf')
    # scalar matrices
    H = Transfer(1, [1, 1])
    k, x, e = lqr(H, 3)
    assert_almost_equal(array([k[0, 0], x[0, 0], e[0]]), [1, 1, -2+0j])


def test_simple_lqr():
    # Example taken from M. de Oliveira's MAE280B lecture notes
    H = State([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [4.03428022844288e-06, 0, 0, 0.0515652322798669],
               [0, 0, -0.000104315254033883, 0]],
              [[0, 0], [1e-5/3, 0], [0, 0], [0, 0.01]],
              eye(4))
    k, _, _ = lqr(H[:, 1], eye(4))
    H.a = H.a.T
    f, _, _ = lqr(H[:, 0], block_diag(0, 0, 1e-5, 1e-5), 0.1)
    assert_almost_equal(k, array([[1.00554916, -1, 52.52180106, 18.51107167]]))
    assert_almost_equal(f,  array([[-577.370350, 173.600463,
                                    0.383744946, 0.050228534]]), decimal=5)


def test_simple_lqry():
    # Scalar matrices
    H = State(1, 1, 1, 1)
    k, x, e = lqr(H, Q=3, weight_on='output')
    assert_almost_equal(array([k[0, 0], x[0, 0], e[0]]), [1.5, 3, -0.5+0j])
    # Wrong S shape
    assert_raises(ValueError, lqr, H, Q=3, S=eye(2), weight_on='output')


def test_simple_dlqr():
    # Example taken from M. de Oliveira's MAE280B lecture notes
    H = State([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [4.03428022844288e-06, 0, 0, 0.0515652322798669],
               [0, 0, -0.000104315254033883, 0]],
              [[0, 0], [1e-5/3, 0], [0, 0], [0, 0.01]],
              eye(4), dt=0.1)
    k, _, _ = lqr(H[:, 1], eye(4))
    H.a = H.a.T
    f, _, _ = lqr(H[:, 0], block_diag(0, 0, 1e-5, 1e-5), 0.1)
    assert_almost_equal(k, array([[0, 0, -2.08727337333631e-06, 0]]))
    assert_almost_equal(f,  array([[1.71884123e-11, 0, 0, -1.79301359e-15]]))


def test_ackermann_args():
    # Not SIxO system
    G = State(eye(2), eye(2), eye(2))
    assert_raises(ValueError, ackermann, G, [1, 2])
    # Wrong # of poles
    G = State(eye(2), [[1], [0]], [1, 0])
    assert_raises(ValueError, ackermann, G, [1, 2, 3])


def test_ackermann_controllable():
    #
    A = haroldcompanion([1, 6, 5, 1])
    B = eye(3)[:, [-1]]
    p = [-10, -9, -8]
    K = ackermann((A, B), p)
    pa = eigvals(A - B@K)
    assert_array_almost_equal(array(p, dtype=complex), sort(pa))


def test_ackermann_uncontrollable():
    A = block_diag(haroldcompanion([1, 6, 5, 1]), 1)
    B = eye(4)[:, [-2]]
    p = [-10, -9, -8, -7]
    assert_raises(ValueError, ackermann, (A, B), p)
