import numpy as np
from numpy.linalg import LinAlgError
from numpy.random import seed
from harold import (Transfer, State, e_i, haroldcompanion,
                    transmission_zeros, state_to_transfer, transfer_to_state,
                    random_state_model, concatenate_state_matrices)

from numpy.testing import (assert_,
                           assert_allclose,
                           assert_equal,
                           assert_array_equal,
                           assert_almost_equal,
                           assert_array_almost_equal)
from pytest import raises as assert_raises


def test_concatenate_state_matrices():
    G = State(1, 2, 3, 4)
    M = concatenate_state_matrices(G)
    assert_array_equal(M, np.array([[1, 2], [3, 4]]))  # 1
    G = State(np.eye(4))
    assert_array_equal(concatenate_state_matrices(G), np.eye(4))


def test_Transfer_Instantiations():
    assert_raises(TypeError, Transfer)
    # Double list is MIMO, num is SISO --> Static!
    G = Transfer(1, [[1, 2, 1]])
    assert_equal(len(G.num), 1)
    assert_equal(len(G.num[0]), 3)
    assert_equal(G.shape, (1, 3))
    assert_(G._isgain)
    G = Transfer(1, np.array([[2, 1, 1]]))
    assert_(G._isSISO)

    G = Transfer([[1]], [1, 2, 1])
    assert_(G._isSISO)
    assert_equal(G.num.shape, (1, 1))
    assert_equal(G.den.shape, (1, 3))

    G = Transfer([[1, 2]], [1, 2, 1])
    assert_(not G._isSISO)
    assert_equal(len(G.num[0]), 2)
    assert_equal(len(G.den[0]), 2)
    assert_equal(G.shape, (1, 2))

    num = [[np.array([1, 2]), 1], [np.array([1, 2]), 0]]
    den = [1, 4, 4]
    G = Transfer(num, den)
    assert_equal(len(G.num), 2)
    assert_equal(len(G.num[0]), 2)
    assert_equal(G.shape, (2, 2))

    G = Transfer(1)
    assert_(G._isSISO)
    assert_equal(G.num, np.array([[1]]))
    assert_equal(G.den, np.array([[1]]))

    G = Transfer(None, 1)
    assert_(G._isSISO)
    assert_equal(G.num, np.array([[1]]))
    assert_equal(G.den, np.array([[1]]))

    G = Transfer(np.random.rand(3, 2))
    assert_(not G._isSISO)
    assert_equal(G.shape, (3, 2))
    assert_equal(G.poles.size, 0)

    assert_raises(IndexError, Transfer, np.ones((3, 2)), [[[1, 2], [1, 1]]])

    # user-reported gh-64
    G = Transfer([[[1.0], [1.0, 0.0]]], [[[1.0], [1.0, 0.0]]], 0.02)
    assert_(not G._isgain)
    assert_allclose(G.num[0][1], np.array([[1., 0.]]))
    assert_allclose(G.poles, np.array([0+0.j]))
    assert_allclose(G.zeros, np.array([0+0.j]))


def test_Transfer_property():
    G = Transfer([1, 1], [1, 1, 1])
    assert G.DiscretizedWith is None

    G.SamplingPeriod = 0.1
    G.DiscretizedWith = 'zoh'
    assert G.DiscretizedWith == 'zoh'

    G = Transfer([1, 1], [1, 1, 1])
    G.num = [1, 2, 1]
    with assert_raises(IndexError):
        G.num = [[1, [1, 2]]]
    G.den = [1, 2, 1]
    with assert_raises(IndexError):
        G.den = [[[1, 2, 3], [1, 2, 5]]]

    with assert_raises(ValueError):
        G.DiscretizedWith = 'zoh'
    with assert_raises(ValueError):
        G.DiscretizationMatrix = 1.
    G = Transfer([0.1, 0.1, -0.5], [1, 1.3, 0.43], 0.1)
    with assert_raises(ValueError):
        G.DiscretizedWith = 'some dummy method'

    G.DiscretizedWith = 'lft'
    G.DiscretizationMatrix = np.array([[1, 2], [1.5, 5.]])  # dummy array
    assert_array_equal(G.DiscretizationMatrix, np.array([[1, 2], [1.5, 5.]]))
    with assert_raises(ValueError):
        G.DiscretizationMatrix = [1., 1.]

    with assert_raises(ValueError):
        G.PrewarpFrequency = 200
    G = Transfer([1, 1], [1, 1, 1], dt=0.1)
    G.DiscretizedWith = 'tustin'
    G.PrewarpFrequency = 0.02
    assert G.PrewarpFrequency == 0.02


def test_Transfer_to_array():
    G = Transfer(1, [1, 1])
    H = Transfer(2, 10)
    with assert_raises(ValueError):
        G.to_array()

    assert_equal(H.to_array(), np.array([[.2]]))
    assert_equal(Transfer(np.arange(9, 90, 9).reshape(3, 3),
                          9*np.ones((3, 3))).to_array(),
                 np.arange(1, 10).reshape(3, 3))


def test_Transfer_algebra_mul_rmul_dt():
    G = Transfer(1, [1, 2], dt=0.1)
    F = Transfer(1, [1, 3])
    with assert_raises(ValueError):
        F*G


def test_Transfer_algebra_truediv_rtruediv():
    G = Transfer(1, [1, 2])
    F = G/0.5
    assert_equal(F.num, np.array([[2.]]))
    assert_equal(F.den, np.array([[1., 2.]]))

    # invert a nonproper system
    with assert_raises(ValueError):
        G/G
    # invert a singular system
    with assert_raises(LinAlgError):
        1 / (np.ones((2, 2))*(1+G))
    with assert_raises(ValueError):
        G/3j

    # invert an invertible system
    J = 1 / (np.eye(2) * G + np.array([[1, 2], [3, 4]]))
    nn, dd = J.polynomials
    nnact = np.array([[x[0].tolist() for x in y] for y in nn])
    ddact = np.array([[x[0].tolist() for x in y] for y in dd])
    nndes = np.array([[[-2., -8.5, -9.], [1., 4., 4.]],
                      [[1.5, 6., 6.], [-0.5, -2.5, -3.]]])
    dddes = np.array([[[1., 1.5, -1.5], [1., 1.5, -1.5]],
                      [[1., 1.5, -1.5], [1., 1.5, -1.5]]])

    assert_array_almost_equal(nnact, nndes)
    assert_array_almost_equal(ddact, dddes)

    G = Transfer(np.eye(3)*0.5)
    assert_array_almost_equal((1 / G).to_array(), np.eye(3)*2)


def test_Transfer_algebra_mul_rmul_scalar_array():
    NUM = [[[12], [-18]], [[6], [-24]]]
    DEN = [[[14, 1.], [21., 1.]], [[7, 1.], [49, 1.]]]
    G = Transfer(NUM, DEN)

    for H in (np.eye(2)*G, G*np.eye(2)):
        assert_equal(H.num[0][1], np.array([[0.]]))
        assert_equal(H.num[1][0], np.array([[0.]]))
        assert_equal(H.den[0][1], np.array([[1.]]))
        assert_equal(H.den[1][0], np.array([[1.]]))
        assert_equal(H.num[0][0], G.num[0][0])
        assert_equal(H.num[1][1], G.num[1][1])
        assert_equal(H.den[0][0], G.den[0][0])
        assert_equal(H.den[1][1], G.den[1][1])

    H = 1/6*G
    assert_equal(float(H.num[0][0]), 2.)
    assert_equal(float(H.num[0][1]), -3.)
    assert_equal(float(H.num[1][0]), 1.)
    assert_equal(float(H.num[1][1]), -4.)

    G = Transfer([[1, 2]], [1, 1])
    H = np.array([2, 1]) * G
    assert_array_equal(H.num[0][0], np.array([[2.]]))
    assert_array_equal(H.num[0][1], np.array([[2.]]))

    H = np.array([2, 0]) * G
    assert_array_equal(H.num[0][1], np.array([[0.]]))
    assert_array_equal(H.den[0][1], np.array([[1.]]))

    H = np.array([[2]]) * G
    assert_array_equal(H.num[0][0], np.array([[2.]]))
    assert_array_equal(H.num[0][1], np.array([[4.]]))

    with assert_raises(ValueError):
        H = np.array([2+1j, 1]) * G

    J = H*0.
    assert_array_equal(J.num[0][0], np.array([[0.]]))
    assert_array_equal(J.num[0][1], np.array([[0.]]))
    assert_array_equal(J.den[0][0], np.array([[1.]]))
    assert_array_equal(J.den[0][1], np.array([[1.]]))

    G = Transfer(1, [1, 1])
    H = G*0.
    assert_array_equal(H.num, np.array([[0.]]))
    assert_array_equal(H.den, np.array([[1.]]))


def test_Transfer_algebra_mul_rmul_siso_mimo():
    F = Transfer(2, [1, 1])
    H = Transfer(np.arange(1, 5).reshape(2, 2), [1, 2])
    K = Transfer([1, 3], [1, 0, 1])

    FK_num, FK_den = (F*K).polynomials
    assert_equal(FK_num, np.array([[2, 6]]))
    assert_equal(FK_den, np.array([[1, 1, 1, 1]]))

    for J in (F*H, H*F):
        for x in range(2):
            for y in range(2):
                assert_equal(J.num[x][y], np.array([[(1+y+2*x)*2]]))
                assert_equal(J.den[x][y], np.array([[1, 3, 2]]))

    H = Transfer([1, 2], [1, 2, 3])*np.arange(1, 5).reshape(2, 2)
    HH = H*H

    for x in range(4):
        assert_equal(sum(HH.den, [])[x], np.array([[1., 4., 10., 12., 9.]]))
        assert_equal(sum(HH.num, [])[x], (x+1)**2 * np.array([[1., 4., 4.]]))

    F = Transfer(1, [1, 1])
    H = State(1, 2, 3, 4, 0.1)
    with assert_raises(ValueError):
        F*H
    with assert_raises(ValueError):
        F*'string'


def test_Transfer_algebra_matmul_rmatmul():

    G = Transfer([[1, [1, 1]]], [[[1, 2, 1], [1, 1]]])
    H = Transfer([[[1, 3]], [1]], [1, 2, 1])
    F = G @ H
    assert_almost_equal(F.num, np.array([[1, 3, 4]]))
    assert_almost_equal(F.den, np.array([[1, 4, 6, 4, 1]]))
    F = H @ G
    assert_almost_equal(F.num[0][0], np.array([[1, 3]]))
    assert_almost_equal(F.num[0][1], np.array([[1, 4, 3]]))
    assert_almost_equal(F.num[1][0], np.array([[1]]))
    assert_almost_equal(F.num[1][1], np.array([[1, 1]]))

    assert_almost_equal(F.den[0][0], np.array([[1, 4, 6, 4, 1]]))
    assert_almost_equal(F.den[0][1], np.array([[1, 3, 3, 1]]))
    assert_almost_equal(F.den[1][0], F.den[0][0])
    assert_almost_equal(F.den[1][1], F.den[0][1])

    F = Transfer(2) @ Transfer(np.eye(2)) @ Transfer(2)
    assert_equal(F.to_array(), 4*np.eye(2))

    G = Transfer([[1, 2]], [1, 1])
    H = np.array([[2], [1]]) @ G
    assert_array_equal(H.num[0][0], np.array([[2.]]))
    assert_array_equal(H.num[0][1], np.array([[4.]]))
    assert_array_equal(H.num[1][0], np.array([[1.]]))
    assert_array_equal(H.num[1][1], np.array([[2.]]))

    G = Transfer([[1, 2]], [1, 1])
    H = G @ np.array([[2], [1]])
    assert H._isSISO
    assert_array_almost_equal(H.num, np.array([[4.]]))
    assert_array_almost_equal(H.den, np.array([[1., 1.]]))

    H = np.array([[2]]) @ G
    assert_array_equal(H.num[0][0], np.array([[2.]]))
    assert_array_equal(H.num[0][1], np.array([[4.]]))

    with assert_raises(ValueError):
        H = np.array([2+1j, 1]) * G

    J = H*0.
    assert_array_equal(J.num[0][0], np.array([[0.]]))
    assert_array_equal(J.num[0][1], np.array([[0.]]))
    assert_array_equal(J.den[0][0], np.array([[1.]]))
    assert_array_equal(J.den[0][1], np.array([[1.]]))

    G = Transfer(1, [1, 1])
    H = G*0.
    assert_array_equal(H.num, np.array([[0.]]))
    assert_array_equal(H.den, np.array([[1.]]))


def test_Transfer_algebra_neg_add_radd():
    G = Transfer(1, [1, 2, 1])
    assert_equal(-(G.num), (-G).num)
    H = Transfer([1, 1], [1, 0.2], 0.1)
    with assert_raises(ValueError):
        G + H
    G, H = Transfer(1), Transfer(2)
    assert_equal((G+H).num, np.array([[3.]]))

    G, H = Transfer(1), State(5)
    assert isinstance(G+H, State)

    G = Transfer(1, [1, 1])
    assert_equal((G+(-G)).num, np.array([[0.]]))
    assert_almost_equal((G + 5).num, np.array([[5, 6]]))

    G = Transfer([[1, 2]], [1, 1])
    H = G + np.array([[3, 4]])
    assert_equal(H.num[0][0], np.array([[3., 4.]]))
    with assert_raises(ValueError):
        G + np.array([3, 4])

    G = Transfer([[1, [1, 1]]], [[[1, 2, 1], [1, 1]]])
    F = - G
    assert_almost_equal(G.num[0][0], -F.num[0][0])
    assert_almost_equal(G.num[0][1], -F.num[0][1])
    H = F + G
    for x in range(2):
        assert_array_equal(H.num[0][x], np.array([[0]]))
        assert_array_equal(H.den[0][x], np.array([[1]]))

    G = Transfer(1, [1, 2, 3])
    F = 5 + G
    assert_almost_equal(F.num, np.array([[5, 10, 16.]]))
    assert_almost_equal(F.den, G.den)
    F = G + 3
    assert_almost_equal(F.num, np.array([[3, 6, 10.]]))
    assert_almost_equal(F.den, G.den)

    F = F * 5
    assert_almost_equal(F.num, np.array([[15, 30, 50]]))
    assert_almost_equal(F.den, G.den)

    F *= 0.4
    assert_almost_equal(F.num, np.array([[6, 12, 20]]))
    assert_almost_equal(F.den, G.den)

    num1 = [[[1., 2.], [0., 3.], [2., -1.]],
            [[1.], [4., 0.], [1., -4., 3.]]]
    den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
            [[3., 0., .0], [2., -1., -1.], [1., 0, 0, 4]]]
    num2 = [[[0, 0, 0, -1], [2.], [-1., -1.]],
            [[1., 2.], [-1., -2.], [4.]]]
    den2 = [[[-1.], [1., 2., 3.], [-1., -1.]],
            [[-4., -3., 2.], [0., 1.], [1., 0.]]]

    G = Transfer(num1, den1)
    assert_raises(ValueError, Transfer, num2, den2)
    den2[1][1] = [2, -1, -1]
    F = Transfer(num2, den2)
    H = G + F
    # Flatten list of lists via sum( , []) trick
    Hnum = [np.array([[3., -3., -6.]]),
            np.array([[5., 6., 9.]]),
            np.array([[-4., -2., 2.]]),
            np.array([[3., 2., -3., 2.]]),
            np.array([[6., -4.]]),
            np.array([[5., -4., 3., 16.]])]

    Hden = [np.array([[3., -2., -4.]]),
            np.array([[1., 2., 3., 0., 0.]]),
            np.array([[-2., -1., 1.]]),
            np.array([[-12., -9., 6., -0., -0.]]),
            np.array([[4., -2., -2.]]),
            np.array([[1., 0., 0., 4., 0.]])]

    Hnum_computed = sum(H.num, [])
    Hden_computed = sum(H.den, [])
    for x in range(np.multiply(*H.shape)):
        assert_almost_equal(Hnum[x], Hnum_computed[x])
        assert_almost_equal(Hden[x], Hden_computed[x])

    # User Reported in #47
    num = np.array([110.0, 0.0])
    den = np.array([85.0, 20.0, 1.0])

    G = Transfer(num, den)
    H = G + 0.25
    F = G + Transfer(0.25)

    for x in [F, H]:
        assert_almost_equal(x.num, np.array([[21.25, 115, 0.25]]))
        assert_almost_equal(x.den, G.den)

    H = G + np.array([[1, 2], [3, 4]])
    Hnumd = [[np.array([[85., 130., 1.]]),
             np.array([[170., 150., 2.]])],
             [np.array([[255., 170., 3.]]),
              np.array([[340., 190., 4.]])]]

    for x, y in ([0, 0], [0, 1], [1, 0], [1, 1]):
        assert_almost_equal(H.num[x][y], Hnumd[x][y])
        assert_almost_equal(H.den[x][y], G.den)

    # Both are static
    with assert_raises(ValueError):
        Transfer([[1, 2], [3, 4]]) + Transfer([[3, 4, 5], [1, 2, 3]])

    with assert_raises(ValueError):
        Transfer([[1, 2], [3, 4]]) + Transfer([[3, 4], [1, 2]], dt=1)


def test_Transfer_slicing():
    Hind = [(1, 5), (4, 1),
            (4, 5), (1, 1),
            (2, 5), (4, 2),
            (2, 5), (4, 2),
            (1, 2), (2, 1),
            (2, 5), (4, 3)]

    np.random.seed(1234)
    H = state_to_transfer(State(np.random.rand(3, 3),
                                np.random.rand(3, 5),
                                np.random.rand(4, 3)))
    F = Transfer(np.random.rand(4, 5))

    for s, sind in ((H, Hind), (F, Hind)):
        for ind, x in enumerate([s[1, :], s[:, 1],
                                s[:, :], s[0, 0],
                                s[1:3, :], s[:, 1:3],
                                s[[1, 2], :], s[:, [1, 2]],
                                s[2, [1, 2]], s[[1, 2], 2],
                                s[::2, :], s[:, ::2]]):
            assert_equal(x.shape, sind[ind])
    assert_raises(ValueError, H.__setitem__)


def test_State_Instantiations():
    assert_raises(TypeError, State)
    G = State(5)
    assert_(G.a.size == 0)
    assert_(G._isSISO)
    assert_(G._isgain)
    assert_equal(G.d, np.array([[5.]]))

    G = State(np.eye(2))
    assert_equal(G.shape, (2, 2))
    assert_(G._isgain)
    assert_(not G._isSISO)

    # Wrong sized A, B, C, D
    assert_raises(ValueError, State, np.ones((3, 2)), [[1], [2]], [1, 2])
    assert_raises(ValueError, State, np.eye(2), [[1], [2], [3]], [1, 2])
    assert_raises(ValueError, State, np.eye(2), [[1], [2]], [1, 2, 3])
    assert_raises(ValueError, State, np.eye(2), [[1], [2]], [1, 2], [0, 0])


def test_State_to_array():
    G = State(1, 1, 1)
    H = State(5)
    with assert_raises(ValueError):
        G.to_array()

    assert_equal(H.to_array(), np.array([[5]]))
    H = State(np.ones((4, 4)))
    assert_equal(H.to_array(), np.ones((4, 4)))


def test_State_algebra_mul_rmul_dt():
    G = State(1, 2, 3, 4, dt=0.1)
    F = State(4, 3, 2, 1)
    with assert_raises(ValueError):
        F*G
    with assert_raises(ValueError):
        G*F


def test_State_algebra_truediv_rtruediv():
    G = State(1, 2, 3, 4)
    F = G/0.5
    assert_equal(F.b, np.array([[4.]]))
    assert_equal(F.d, np.array([[8.]]))
    G.d = 0.
    with assert_raises(LinAlgError):
        G/G
    with assert_raises(ValueError):
        G/3j

    G.d = 4
    # nonminimal but acceptable
    H = G / G
    ha, hb, hc, hd = H.matrices

    assert_array_almost_equal(ha, [[1, -1.5], [0, -0.5]])
    assert_array_almost_equal(hb, [[0.5], [0.5]])
    assert_array_almost_equal(hc, [[3, -3]])
    assert_array_almost_equal(hd, [[1]])

    G = State(np.eye(3)*0.5)
    assert_array_almost_equal((1 / G).to_array(), np.eye(3)*2)


def test_State_algebra_mul_rmul_scalar_array():
    G = State(np.diag([-1, -2]), [[1, 2], [3, 4]], np.eye(2))
    F = G*np.eye(2)
    Fm = G@np.eye(2)
    assert_equal(concatenate_state_matrices(F), concatenate_state_matrices(Fm))
    F = np.eye(2)*G
    Fm = np.eye(2)@G
    assert_equal(concatenate_state_matrices(F), concatenate_state_matrices(Fm))
    H = 1/2*G
    assert_equal(H.c, 0.5*G.c)


def test_State_matmul_rmatmul_ndarray():
    H = State([[-5, -2], [1, 0]], [[2], [0]], [3, 1], 1)
    J1 = np.array([[-5., -2., 0., 0., 0., 0., 2., 4., 6., 8.],
                   [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., -5., -2., 0., 0., 10., 12., 14., 16.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., -5., -2., 18., 20., 22., 24.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [3., 1., 0., 0., 0., 0., 1., 2., 3., 4.],
                   [0., 0., 3., 1., 0., 0., 5., 6., 7., 8.],
                   [0., 0., 0., 0., 3., 1., 9., 10., 11., 12.]])

    J2 = np.array([[-5., -2., 0., 0., 0., 0., 2., 0., 0.],
                   [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., -5., -2., 0., 0., 0., 2., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., -5., -2., 0., 0., 2.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [3., 1., 6., 2., 9., 3., 1., 2., 3.],
                   [12., 4., 15., 5., 18., 6., 4., 5., 6.],
                   [21., 7., 24., 8., 27., 9., 7., 8., 9.],
                   [30., 10., 33., 11., 36., 12., 10., 11., 12.]])

    mat = np.arange(1, 13).reshape(3, 4)
    Fm = concatenate_state_matrices(mat @ H)
    assert_array_almost_equal(J1, Fm)
    Fm = concatenate_state_matrices(H @ mat)
    assert_array_almost_equal(J1, Fm)

    mat = np.arange(1, 13).reshape(4, 3)
    Fm = concatenate_state_matrices(mat @ H)
    assert_array_almost_equal(J2, Fm)
    Fm = concatenate_state_matrices(H @ mat)
    assert_array_almost_equal(J2, Fm)

    G, H = random_state_model(2, 2, 2), random_state_model(2, 3, 3)
    with assert_raises(ValueError):
        G @ H

    # Scalars
    G = random_state_model(1)
    H = 0. @ G
    assert H._isgain
    H = 1. @ G
    assert_almost_equal(concatenate_state_matrices(G),
                        concatenate_state_matrices(H))

    # static gain mults
    G = random_state_model(0, 4, 5)
    H = random_state_model(0, 5, 4)
    assert (G@H)._isgain
    assert_equal((G@H).shape, (4, 4))
    H = random_state_model(0, 3, 3)
    with assert_raises(ValueError):
        G @ H

    G = State(1.)
    H = random_state_model(1, 2, 2)
    assert_almost_equal(concatenate_state_matrices(G @ H),
                        concatenate_state_matrices(H @ G))

    G = random_state_model(1, 4, 5)
    H = random_state_model(1, 4, 5)
    with assert_raises(ValueError):
        G @ H


def test_State_algebra_mul_rmul_mimo_siso():
    sta_siso = State(5)
    sta_mimo = State(2.0*np.eye(3))
    dyn_siso = State(haroldcompanion([1, 3, 3, 1]), e_i(3, -1), e_i(3, 1).T)
    dyn_mimo = State(haroldcompanion([1, 3, 3, 1]), e_i(3, [1, 2]), np.eye(3))
    dyn_mimo_sq = State(haroldcompanion([1, 3, 3, 1]), np.eye(3), np.eye(3))

    G = dyn_siso * dyn_mimo
    J = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                  [-1., -3., -3., 0., 0., 0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., -1., -3., -3., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0., 0., -1., -3., -3., 0., 1.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    assert_array_almost_equal(concatenate_state_matrices(G), J)
    G = dyn_mimo * dyn_siso
    assert_array_almost_equal(concatenate_state_matrices(G), J)

    G = dyn_mimo * sta_siso
    assert_array_almost_equal(G.b, 5*dyn_mimo.b)
    assert_array_almost_equal(G.d, 5*dyn_mimo.d)
    assert_raises(ValueError, sta_mimo.__add__, dyn_mimo)
    F = sta_mimo @ dyn_mimo
    J = np.array([[0., 1., 0., 0., 0.],
                  [0., 0., 1., 1., 0.],
                  [-1., -3., -3., 0., 1.],
                  [2., 0., 0., 0., 0.],
                  [0., 2., 0., 0., 0.],
                  [0., 0., 2., 0., 0.]])
    assert_array_almost_equal(concatenate_state_matrices(F), J)
    assert_almost_equal((dyn_mimo_sq + sta_mimo).d, 2*np.eye(3))


def test_State_algebra_add_radd():
    sta_siso = State(5)
    sta_mimo = State(2.0*np.eye(3))
    dyn_siso = State(haroldcompanion([1, 3, 3, 1]), e_i(3, -1), e_i(3, 1).T)
    dyn_mimo = State(haroldcompanion([1, 3, 3, 1]), e_i(3, [1, 2]), np.eye(3))
    dyn_mimo_sq = State(haroldcompanion([1, 3, 3, 1]), np.eye(3), np.eye(3))

    G = dyn_mimo + sta_siso
    assert_array_almost_equal(G.d, sta_siso.to_array()*np.ones(dyn_mimo.shape))
    assert_raises(ValueError, dyn_mimo.__add__, sta_mimo)
    G = dyn_mimo_sq + sta_mimo
    assert_array_almost_equal(G.d, 2.*np.eye(3))
    G = dyn_mimo + dyn_siso
    J = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0., 1., 0.],
                  [-1., -3., -3., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., -1., -3., -3., 1., 1.],
                  [1., 0., 0., 0., 1., 0., 0., 0.],
                  [0., 1., 0., 0., 1., 0., 0., 0.],
                  [0., 0., 1., 0., 1., 0., 0., 0.]])
    assert_array_almost_equal(concatenate_state_matrices(G), J)
    assert_raises(ValueError, dyn_mimo.__add__, dyn_mimo_sq)
    assert_raises(ValueError, dyn_mimo.__sub__, dyn_mimo_sq)
    assert_raises(ValueError, dyn_mimo.__radd__, dyn_mimo_sq)
    assert_raises(ValueError, dyn_mimo.__rsub__, dyn_mimo_sq)


def test_State_slicing():
    F = State(1, 2, 3, 4)
    F0 = F[0, 0]
    assert_equal(concatenate_state_matrices(F), concatenate_state_matrices(F0))

    F = State(np.random.rand(4, 4))
    H = State(F.d, np.random.rand(4, 3), np.random.rand(5, 4))
    Hind = [(1, 3), (5, 1),
            (5, 3), (1, 1),
            (2, 3), (5, 2),
            (2, 3), (5, 2),
            (1, 2), (2, 1),
            (3, 3), (5, 2)]
    Find = [(1, 4), (4, 1),
            (4, 4), (1, 1),
            (2, 4), (4, 2),
            (2, 4), (4, 2),
            (1, 2), (2, 1),
            (2, 4), (4, 2)]

    for s, sind in ((H, Hind), (F, Find)):
        for ind, x in enumerate([s[1, :], s[:, 1],
                                s[:, :], s[0, 0],
                                s[1:3, :], s[:, 1:3],
                                s[[1, 2], :], s[:, [1, 2]],
                                s[2, [1, 2]], s[[1, 2], 2],
                                s[::2, :], s[:, ::2]]):
            assert_equal(x.shape, sind[ind])

    assert_raises(ValueError, H.__setitem__)


def test_model_zeros():
    # Test example
    A = np.array(
        [[-3.93, -0.00315, 0, 0, 0, 4.03E-5, 0, 0, 0],
         [368, -3.05, 3.03, 0, 0, -3.77E-3, 0, 0, 0],
         [27.4, 0.0787, -5.96E-2, 0, 0, -2.81E-4, 0, 0, 0],
         [-0.0647, -5.2E-5, 0, -0.255, -3.35E-6, 3.6e-7, 6.33E-5, 1.94E-4, 0],
         [3850, 17.3, -12.8, -12600, -2.91, -0.105, 12.7, 43.1, 0],
         [22400, 18, 0, -35.6, -1.04E-4, -0.414, 90, 56.9, 0],
         [0, 0, 2.34E-3, 0, 0, 2.22E-4, -0.203, 0, 0],
         [0, 0, 0, -1.27, -1.00E-3, 7.86E-5, 0, -7.17E-2, 0],
         [-2.2, -177e-5, 0, -8.44, -1.11E-4, 1.38E-5, 1.49E-3, 6.02E-3, -1E-10]
         ])
    B = np.array([[0, 0],
                  [0, 0],
                  [1.56, 0],
                  [0, -5.13E-6],
                  [8.28, -1.55],
                  [0, 1.78],
                  [2.33, 0],
                  [0, -2.45E-2],
                  [0, 2.94E-5]
                  ])
    C = e_i(9, [5, 8], output='r')
    D = np.zeros((2, 2))
    zs = transmission_zeros(A, B, C, D)
    res = np.array([-2.64128629e+01 - 0j,
                    -2.93193619 - 0.419522621j,
                    -9.52183370e-03 + 0j,
                    -2.93193619 + 0.419522621j,
                    1.69789270e-01 - 0j,
                    5.46527700e-01 - 0j])
    # Sort is numerically too sensitive for imaginary parts.
    assert_almost_equal(np.sort(np.imag(zs)), np.sort(np.imag(res)))  # 0.0
    assert_almost_equal(np.sort(np.real(zs)), np.sort(np.real(res)))  # 0.1
    # An example found online (citation lost), please let me know
    A = np.array([[-6.5000, 0.5000, 6.5000, -6.5000],
                  [-0.5000, -5.5000, -5.5000, 5.5000],
                  [-0.5000, 0.5000, 0.5000, -6.5000],
                  [-0.5000, 0.5000, -5.5000, -0.5000]])
    B = np.array([[0., 1, 0],
                  [2., 1, 2],
                  [3., 4, 3],
                  [3., 2, 3]])
    C = np.array([[1, 1, 0, 0]])
    D = np.zeros((1, 3))
    zs = transmission_zeros(A, B, C, D)
    res = np.array([-7, -6])
    assert_almost_equal(np.sort(res), np.sort(zs))  # 1
    # Example from Reinschke, 1988
    A = np.array([[0, 0, 1, 0, 0, 0],
                  [2, 0, 0, 3, 4, 0],
                  [0, 0, 5, 0, 0, 6],
                  [0, 7, 0, 0, 0, 0],
                  [0, 0, 0, 8, 9, 0],
                  [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 0, 11]]).T
    C = np.array([[0, 12, 0, 0, 13, 0],
                  [14, 0, 0, 0, 0, 0],
                  [15, 0, 16, 0, 0, 0]])
    D = np.zeros((3, 2))
    zs = transmission_zeros(A, B, C, D)
    res = np.array([-6.78662791+0.j,  3.09432022+0.j])
    assert_almost_equal(np.sort(zs), np.sort(res))  # 2


def test_static_model_conversion_sampling_period():
    G = State(np.eye(5), dt=0.001)
    H = state_to_transfer(G)
    assert_(H._isgain)  # 0
    assert_(not H._isSISO)  # 1
    assert_equal(H.SamplingPeriod, 0.001)  # 2
    K = transfer_to_state(H)
    assert_equal(K.SamplingPeriod, 0.001)  # 3


def test_random_state_model():
    seed(12345)
    # Simple arguments
    G = random_state_model(0)
    assert G._isgain
    assert G._isSISO
    G = random_state_model(1)
    assert not G._isgain
    assert G._isSISO
    G = random_state_model(1, 1, 2)
    assert not G._isgain
    assert not G._isSISO

    G = random_state_model(5, 2, 4, stable=True)
    assert not (G.poles.real > 0).any()
    G = random_state_model(11, stable=False, prob_dist=[0, 0, 0.5, 0.5])
    assert_array_almost_equal(np.abs(G.poles.real), np.zeros(11))
    assert np.any(G.poles.imag)

    a1 = random_state_model(101, dt=0.1).poles
    assert np.all(np.abs(a1) <= 1.)


def test_basic_pole_properties():
    G = Transfer(0.5, [1, 4, 3]) + 5
    zzz = G.pole_properties()
    assert_array_almost_equal(zzz,
                              np.array([[-1.+0.j, 1.+0.j, 1.+0.j],
                                        [-3.+0.j, 3.+0.j, 1.+0.j]]))


def test_transfer_to_state():
    # Models with static column/row
    num, den = [[1, -1], [[1, -1], 0]], [[[1, 2], 1], [[1, 2], 1]]
    den2, num2 = [list(i) for i in zip(*den)], [list(i) for i in zip(*num)]

    G = Transfer(num, den)
    H = Transfer(num2, den2)

    Gs = transfer_to_state(G)
    Hs = transfer_to_state(H)
    Gm = concatenate_state_matrices(Gs)
    Hm = concatenate_state_matrices(Hs)
    assert_array_almost_equal(Gm, np.array([[-2, 1, 0],
                                            [1, 0, -1],
                                            [-3, 1, 0]]))
    assert_array_almost_equal(Hm, np.array([[-2., 0., 1., 0.],
                                            [0., -2., 0., 1.],
                                            [1., -3., 0., 1.],
                                            [0., 0., -1., 0.]]))

    # Example from Kalman 1963
    num = [[3*np.poly([-3, -5]), [6, 6], [2, 7], [2, 5]],
           [2, 1, [2, 10], [8, 16]],
           [[2, 14, 36], [-2, 0], 1, 2*np.convolve([5, 17], [1, 2])]]
    den = [[np.poly([-1, -2, -4]), [1, 6, 8], [1, 7, 12], [1, 5, 6]],
           [[1, 8, 15], [1, 3], np.poly([-1, -2, -3]), np.poly([-1, -3, -5])],
           [np.poly([-1, -3, -5]), [1, 4, 3], [1, 3], np.poly([-1, -3, -5])]]

    G = Transfer(num, den)
    H = transfer_to_state(G)
    p = H.poles
    p.sort()
    assert_array_almost_equal(p, np.array([-5.+0.j, -5.+0.j, -4.+0.j,
                                           -3.+0.j, -3.+0.j, -3.+0.j,
                                           -2.+0.j, -2.+0.j, -1.+0.j,
                                           -1.+0.j, -1.+0.j]))

    # Reported in gh-#42
    G = Transfer([[[87.8, 8.78], [-103.68, -8.64]],
                  [[129.84, 10.82], [-109.6, -10.96]]],
                 [562.5, 82.5, 1])
    Gss = transfer_to_state(G)
    assert_array_almost_equal(Gss.a, np.kron(np.eye(2), [[0., 1.],
                                                         [-2/1125, -11/75]]))
    assert_array_almost_equal(Gss.b, [[0, 0], [1, 0], [0, 0], [0, 1]])
    des_c = np.array([[0.01560888888888889,
                       0.1560888888888889,
                       -0.015360000000000002,
                       -0.18432],
                      [0.019235555555555558,
                       0.23082666666666668,
                       -0.019484444444444447,
                       -0.19484444444444443]])

    assert_array_almost_equal(Gss.c, des_c)
    assert_array_almost_equal(Gss.d, np.zeros([2, 2]))

    # reported in gh-#50
    num = [[[61.79732492202783, 36.24988430260625, 0.7301196233698941],
            [0.0377840674057878, 0.9974993795127982, 21.763622825733773]]]
    den = [[[84.64, 18.4, 1.0], [1.0, 7.2, 144.0]]]

    TF = transfer_to_state((num, den))
    assert_array_almost_equal([-3.6-1.14472704e+01j,
                               -3.6+1.14472704e+01j,
                               -0.10869565-1.74405237e-07j,
                               -0.10869565+1.74405237e-07j,
                               ],
                              np.sort(TF.poles))
    assert TF.zeros.size == 0

    # rectengular static gain
    gain = np.ones([2, 3])
    Gss = transfer_to_state(Transfer(gain))
    assert_array_almost_equal(gain, Gss.d)


def test_state_to_transfer():
    G = State(-2*np.eye(2), np.eye(2), [[1, -3], [0, 0]], [[0, 1], [-1, 0]])
    H = state_to_transfer(G)
    H11 = H[1, 1]
    assert_array_equal(H11.num, np.array([[0.]]))
    assert_array_equal(H11.den, np.array([[1.]]))


def test_verbosity_State(capsys):
    _ = State.validate_arguments(-2*np.eye(2), np.eye(2),
                                 [[1, -3], [0, 0]], [[0, 1], [-1, 0]],
                                 verbose=True)

    out, err = capsys.readouterr()
    assert not err.strip()
    assert out == ("========================================\n"
                   "Handling A\n"
                   "========================================\n"
                   "Trying to np.array A\n"
                   "========================================\n"
                   "Handling B\n"
                   "========================================\n"
                   "Trying to np.array B\n"
                   "========================================\n"
                   "Handling C\n"
                   "========================================\n"
                   "Trying to np.array C\n"
                   "========================================\n"
                   "Handling D\n"
                   "========================================\n"
                   "Trying to np.array D\n"
                   "All seems OK. Moving to shape mismatch check\n")


def test_verbosity_Transfer(capsys):
    _ = Transfer.validate_arguments(-2*np.eye(2), [1, 2, 3], verbose=True)
    out, err = capsys.readouterr()
    assert not err.strip()
    assert out == ("========================================\n"
                   "Handling numerator\n"
                   "========================================\n"
                   "I found a numpy array\n"
                   "The array has multiple elements\n"
                   "========================================\n"
                   "Handling denominator\n"
                   "========================================\n"
                   "I found a list\n"
                   "I found a list that has only scalars\n"
                   "==================================================\n"
                   "Handling raw entries are done.\n"
                   "Now checking the SISO/MIMO context and regularization.\n"
                   "==================================================\n"
                   "One of the MIMO flags are true\n"
                   "Numerator is MIMO, Denominator is something else\n"
                   "Numerator is MIMO, Denominator is SISO\n")
