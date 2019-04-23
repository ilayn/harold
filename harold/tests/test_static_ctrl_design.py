from numpy import eye, array, sort, empty
from scipy.linalg import block_diag, eigvals
from scipy.signal.filter_design import _cplxpair
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)

from pytest import raises as assert_raises
from harold import lqr, ackermann, State, Transfer, haroldcompanion
from harold._static_ctrl_design import _get_pole_reps


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


def byersnash_A_B_test_pairs():
    ABs = [
           # Chemical Reactor (Munro 1979)
           (array([[1.38, -0.2077, 6.715, -5.676],
                   [-0.5814, -4.29, 0, 0.675],
                   [1.067, 4.273, -6.654, 5.893],
                   [0.048, 4.273, 1.343, -2.104]]),
            array([[0, 0],
                   [5.679, 0],
                   [1.136, -3.146],
                   [1.136, 0]])),
           # Distillation Column (Klein, Moore 1977)
           (array([[-0.1094, 0.0628, 0, 0, 0],
                   [1.306, -2.132, 0.9807, 0, 0],
                   [0, 1.595, -3.149, 1.547, 0],
                   [0, 0.0355, 2.632, -4.257, 1.855],
                   [0, 0.0023, 0, 0.1636, -0.1625]]),
            array([[0, 0],
                   [0.638, 0],
                   [0.0838, -0.1396],
                   [0.1004, -0.206],
                   [0.0063, -0.0128]])),
           # Nuclear rocket engine (Davison, Chow 1974)
           (array([[-65.0, 65, -19.5, 19.5],
                   [0.1, -0.1, 0, 0],
                   [1, 0, -0.5, -1],
                   [0, 0, 0.4, 0]]),
            array([[65., 0],
                   [0, 0],
                   [0, 0],
                   [0, 0.4]])),
           # MIMO system (Atkinson, 1985)
           (array([[0, 1, 0],
                   [0, 0, 1],
                   [-6, -11, -6]]),
            array([[1, 1],
                   [0, 1],
                   [1, 1]])),
           # Drum boiler (Bengtsson 1973)
           (array([[-0.129, 0, 0.396, 0.25, 0.00191],
                   [0.0329, 0, -0.00779, 0.0122, -0.621],
                   [0.00718, 0, -0.1, 0.000887, -0.0385],
                   [0.00411, 0, 0, -0.0822, 0],
                   [0.00351, 0, 0.0035, 0.00426, -0.0743]]),
            array([[0, 0.1390],
                   [0, 0.0359],
                   [0, -0.0989],
                   [0.0249, 0],
                   [0, -0.00534]])),
           # Miminis random example #1
           (array([[5.8765, 9.3456, 4.5634, 9.3520],
                   [6.6526, 0.5867, 3.5829, 0.6534],
                   [0.0000, 9.6738, 7.4876, 4.7654],
                   [0.0000, 0.0000, 6.6784, 2.5678]]),
            array([[3.9878, 0.5432],
                   [0.0000, 2.7650],
                   [0.0000, 0.0000],
                   [0.0000, 0.0000]])),
           # Miminis random example #2
           (array([[.5257, .8544, .5596, .5901, .0259, .6213, .7227, .5617],
                   [.9931, .0643, .1249, .3096, .5174, .3455, .8977, .4682],
                   [.6489, .8279, .7279, .2552, .3917, .7065, .2428, .7795],
                   [.9923, .9262, .2678, .6252, .2414, .5211, .4338, .9677],
                   [.0000, .5667, .5465, .1157, .5064, .2870, .7901, .9809],
                   [.0000, .0000, .8672, .6117, .4236, .6503, .5069, .8187],
                   [.0000, .0000, .0000, .0000, .2894, .0881, .5233, .4257],
                   [.0000, .0000, .0000, .0000, .0000, .4499, .5597, .2462]]),
            array([[0.9230, 0.3950, 0.8325],
                   [0.0000, 0.0366, 0.6105],
                   [0.0000, 0.0000, 0.1871],
                   [0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000],
                   [0.0000, 0.0000, 0.0000]])),
           # Aircraft control example I (Kautsky and Nichols 1983)
           (array([[0, 1, 0, 0],
                   [1.40e-4, -2.04, -1.95, -1.33e-2],
                   [-2.51e-4, 1, -1.32, -2.38e-2],
                   [-5.61e-1, 0, 0.358, -2.79e-1]]),
            array([[0, 0, 0],
                   [-5.33, 6.45e-3, -2.67e-1],
                   [-1.60e-1, -1.16e-2, -2.51e-1],
                   [0, 1.06e-1, 8.62e-2]])),
           # Aircraft control example II (Kautsky and Nichols 1983)
           (array([[0, 1, 0, 0],
                   [5.32e-7, -4.18e-1, -0.12, -2.32e-2],
                   [-4.62e-9, 1, -0.752, -2.39e-2],
                   [-5.61e-1, 0, 0.3, -1.74e-2]]),
            array([[0, 0],
                   [-1.72e-1, 7.45e-6],
                   [-2.82e-2, -7.78e-5],
                   [0, 3.69e-3]])),
           # Symmetric example (Kautsky and Nichols 1983)
           (array([[-3.624, 4.9567e-2, -2.4564e-1, 1.3853e-2],
                   [3.3486e-1, -1.8875, -8.1251e-1, -2.8102e-1],
                   [-1.9958e-1, -1.1335, -2.2039, -4.5523e-1],
                   [1.3784e-1, -4.7140e-1, -3.3229e-1, -4.0605]]),
            array([[2.3122e-1, 3.0761e-1, 3.6164e-1, 3.3217e-1],
                   [8.8339e-1, 2.1460e-1, 5.6642e-1, 5.0153e-1]]).T),
           # Ad-hoc ill-conditioned example (Byers and Nash 1989)
           (array([[0, 0, 0, 0],
                   [1, 10, 100, 1000],
                   [0, 1, 10, 100],
                   [0, 0, 1, 10]]),
            array([[1, 0],
                   [0, 1],
                   [0, 0],
                   [0, 0]]))
        ]

    # Return a generator
    return (x for x in ABs)


def _test_get_pole_reps():

    # Only complex
    p = array([1.+1j, 1-1j, 2.+1j, 2-1j])
    pr, nc, nr = _get_pole_reps(p)
    for x in range(2):
        assert_array_equal(pr[x], empty((0, 2)))
    assert nc == 4
    assert nr == 0

    # Only real
    p = array([1, 2, 3])
    pr, nc, nr = _get_pole_reps(p)
    for x in range(2):
        assert_array_equal(pr[x], empty((0, 2)))
    assert nc == 0
    assert nr == 3

    # Mixed, no reps
    p = array([1.+1j, 1-1j, 3])
    pr, nc, nr = _get_pole_reps(p)
    for x in range(2):
        assert_array_equal(pr[x], empty((0, 2)))
    assert nc == 2
    assert nr == 1

    # Mixed, complex reps
    p = array([1.+1j, 1-1j, 1.+1j, 1-1j, 3])
    p = _cplxpair(p).conj()
    pr, nc, nr = _get_pole_reps(p)
    assert_array_equal(pr[0], array([[0, 2]]))
    assert_array_equal(pr[1], empty((0, 2)))
    assert nc == 4
    assert nr == 1

    # Mixed real reps
    p = array([1.+1j, 1-1j, 1., 1])
    p = _cplxpair(p).conj()
    pr, nc, nr = _get_pole_reps(p)
    assert_array_equal(pr[0], empty((0, 2)))
    assert_array_equal(pr[1], array([[2, 4]]))
    assert nc == 2
    assert nr == 2

    # Mixed real reps, real dangling
    p = array([1.+1j, 1-1j, 1., 1, 0.54, 3.8])
    p = _cplxpair(p).conj()
    pr, nc, nr = _get_pole_reps(p)
    assert_array_equal(pr[0], empty((0, 2)))
    assert_array_equal(pr[1], array([[3, 5]]))
    assert nc == 2
    assert nr == 4

    # Mixed complex reps, complex dangling
    p = array([1.+1j, 1-1j, 1.+1j, 1-1j, 0.+1j, 0-1j, 0.5, 3.])
    p = _cplxpair(p).conj()
    pr, nc, nr = _get_pole_reps(p)
    assert_array_equal(pr[0], array([[1, 3]]))
    assert_array_equal(pr[1], empty((0, 2)))
    assert nc == 6
    assert nr == 2

    # Mixed reps and dangling
    p = array([1.+1j, 1-1j, 1.+1j, 1-1j,
               2.+1j, 2-1j,
               3.+1j, 3-1j, 3.+1j, 3-1j, 3.+1j, 3-1j,
               4.+1j, 4-1j,
               0,
               0.5, 0.5,
               3.,
               6, 6, 6])
    p = _cplxpair(p).conj()
    pr, nc, nr = _get_pole_reps(p)
    assert_array_equal(pr[0], array([[0, 2],
                                     [3, 6]]))
    assert_array_equal(pr[1], array([[15, 17],
                                     [18, 21]]))
    assert nc == 14
    assert nr == 7
