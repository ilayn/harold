from harold import (State, Transfer, feedback, lqr, matrix_slice,
                    concatenate_state_matrices)

from numpy import array, eye
from numpy.testing import assert_equal, assert_almost_equal, assert_raises


def test_feedback_wrong_inputs():
    G = Transfer(1, [1, 1])
    H = Transfer([3, 1], [1, 2], dt=0.01)
    assert_raises(ValueError, feedback, G, H)
    assert_raises(ValueError, feedback, G, [1, 2])
    assert_raises(ValueError, feedback, G, 5+1j)


def test_feedback_wellposedness():
    G = State(eye(3), [[1], [1], [1]], [1, 1, 1], [1])
    assert_raises(ValueError, feedback, G, 1+0j)


def test_feedback_static_static():
    G = State(5)
    H = Transfer(4)
    assert_almost_equal(feedback(G, G).d, array([[10.208333333333334]]))
    assert_almost_equal(feedback(G, H).d, array([[10.263157894736842]]))
    assert_almost_equal(feedback(H, G).d, array([[8.210526315789473]]))
    assert_almost_equal(feedback(H, H).num, array([[8.266666666666666]]))


def test_feedback_dynamic_static():
    M = array([[2, 2, 1, -1],
               [3, 2, -2, 3],
               [1, -1, 1, -3],
               [0, -1, 2, 0]])
    a, b, c, d = matrix_slice(M, [3, 3])
    G = State(a, b, eye(3))
    H, _, _ = lqr(G, eye(3))
    CL = feedback(G, H)
    assert_almost_equal(CL.poles,
                        array([-5.42998305, -3.75504185, -1.55400925]))
    G = State(a, b, c, d)
    CL = feedback(G, 5+0j)
    assert_almost_equal(concatenate_state_matrices(CL),
                        array([[2., -3., 11., -1.],
                               [3., 17., -32., 3.],
                               [1., -16., 31., -3.],
                               [0., -1., 2., 0.]]))
    CL = feedback(5, G)
    assert_almost_equal(concatenate_state_matrices(CL),
                        array([[2., -3., 11., -5.],
                               [3., 17., -32., 15.],
                               [1., -16., 31., -15.],
                               [0., 5., -10., 5.]]))


def test_feedback_dynamic_dynamic():
    M = array([[2, 2, 1, -1],
               [3, 2, -2, 3],
               [1, -1, 1, -3],
               [0, -1, 2, 0]])
    N = array([[2, 0, 2, 1],
               [-2, 1, -2, -2],
               [0, 1, 1, -3],
               [0, 3, -1, 2]])
    a, b, c, d = matrix_slice(M, [3, 3])
    k, l, m, n = matrix_slice(N, [3, 3])
    G = State(a, b, c, d)
    H = State(k, l, m, n)
    CL = feedback(G, H)
    acl = array([[2., 0., 5., 0., 3., -1.],
                 [3., 8., -14., 0., -9., 3.],
                 [1., -7., 13., 0., 9., -3.],
                 [0., -1., 2., 2., 0., 2.],
                 [0., 2., -4., -2., 1., -2.],
                 [0., 3., -6., 0., 1., 1.]])
    bcl = array([[-1, 3, -3, 0, 0, 0]]).T
    ccl = array([[0, -1, 2, 0, 0, 0]])
    dcl = array([[0.]])
    assert_equal(CL.a, acl)
    assert_equal(CL.b, bcl)
    assert_equal(CL.c, ccl)
    assert_equal(CL.d, dcl)
    G = State(a, b, c, 1)
    CL = feedback(G, H)
    acl = array([[2., 4., -3., 0., -3., 1.],
                 [3., -4., 10., 0., 9., -3.],
                 [1., 5., -11., 0., -9., 3.],
                 [0., -3., 6., 2., 3., 1.],
                 [0., 6., -12., -2., -5., 0.],
                 [0., 9., -18., 0., -8., 4.]])
    bcl = array([[-3.], [9.], [-9.], [3.], [-6.], [-9.]])
    ccl = array([[0., -3., 6., 0., 3., -1.]])
    dcl = array([[3.]])
    assert_equal(CL.a, acl)
    assert_equal(CL.b, bcl)
    assert_equal(CL.c, ccl)
    assert_equal(CL.d, dcl)
