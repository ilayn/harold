"""
The MIT License (MIT)

Copyright (c) 2016 Ilhan Polat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from harold import (Transfer, State, e_i, haroldcompanion,
                    transmission_zeros, state_to_transfer, transfer_to_state,
                    concatenate_state_matrices)

from numpy.testing import (assert_,
                           assert_equal,
                           assert_array_equal,
                           assert_raises,
                           assert_almost_equal)


def test_Transfer_Instantiations():
    assert_raises(TypeError, Transfer)
    # Double list is MIMO, num is SISO --> Static!
    G = Transfer(1, [[1, 2, 1]])
    assert_equal(len(G.num), 1)
    assert_equal(len(G.num[0]), 3)
    assert_equal(G.shape, (1, 3))
    assert G._isgain

    G = Transfer([[1]], [1, 2, 1])
    assert G._isSISO
    assert_equal(G.num.shape, (1, 1))
    assert_equal(G.den.shape, (1, 3))

    G = Transfer([[1, 2]], [1, 2, 1])
    assert not G._isSISO
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
    assert G._isSISO
    assert_equal(G.num, np.array([[1]]))
    assert_equal(G.den, np.array([[1]]))

    G = Transfer(None, 1)
    assert G._isSISO
    assert_equal(G.num, np.array([[1]]))
    assert_equal(G.den, np.array([[1]]))

    G = Transfer(np.random.rand(3, 2))
    assert not G._isSISO
    assert_equal(G.shape, (3, 2))
    assert G.poles.size == 0

    assert_raises(IndexError, Transfer, np.ones((3, 2)), [[[1, 2], [1, 1]]])


def test_Transfer_algebra_mul_rmul():
    NUM = [[[12], [-18]], [[6], [-24]]]
    DEN = [[[16, 1.], [21., 1.]], [[10, 1.], [14, 1.]]]
    G = Transfer(NUM, DEN)
    H = np.eye(2)*G
    assert_equal(H.num[0][1], np.array([[0.]]))
    assert_equal(H.num[1][0], np.array([[0.]]))
    assert_equal(H.den[0][1], np.array([[1.]]))
    assert_equal(H.den[1][0], np.array([[1.]]))
    assert_equal(H.num[0][0], G.num[0][0])
    assert_equal(H.num[1][1], G.num[1][1])
    assert_equal(H.den[0][0], G.den[0][0])
    assert_equal(H.den[1][1], G.den[1][1])
    H = G*np.eye(2)
    assert_equal(H.num[0][1], np.array([[0.]]))
    assert_equal(H.num[1][0], np.array([[0.]]))
    assert_equal(H.den[0][1], np.array([[1.]]))
    assert_equal(H.den[1][0], np.array([[1.]]))
    assert_equal(H.num[0][0], G.num[0][0])
    assert_equal(H.num[1][1], G.num[1][1])
    assert_equal(H.den[0][0], G.den[0][0])
    assert_equal(H.den[1][1], G.den[1][1])
    H = 1/6*G
#    assert_equal()


def test_Transfer_algebra_mimo_siso():

    G = Transfer([[1, [1, 1]]], [[[1, 2, 1], [1, 1]]])
    H = Transfer([[[1, 3]], [1]], [1, 2, 1])
    F = G*H
    assert_almost_equal(F.num, np.array([[1, 3, 4]]))
    assert_almost_equal(F.den, np.array([[1, 4, 6, 4, 1]]))
    F = H*G
    assert_almost_equal(F.num[0][0], np.array([[1, 3]]))
    assert_almost_equal(F.num[0][1], np.array([[1, 4, 3]]))
    assert_almost_equal(F.num[1][0], np.array([[1]]))
    assert_almost_equal(F.num[1][1], np.array([[1, 1]]))

    assert_almost_equal(F.den[0][0], np.array([[1, 4, 6, 4, 1]]))
    assert_almost_equal(F.den[0][1], np.array([[1, 3, 3, 1]]))
    assert_almost_equal(F.den[1][0], F.den[0][0])
    assert_almost_equal(F.den[1][1], F.den[0][1])

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
    Hnum = [np.array([[-1., 5/3, 10/3]]),
            np.array([[5., 6., 9.]]),
            np.array([[1., 0.5, -0.5]]),
            np.array([[1., 3., 0.75, -0.5]]),
            np.array([[3., -2.]]),
            np.array([[5., -4., 3., 16.]])
            ]

    Hden = [np.array([[1., -2/3, -4/3]]),
            np.array([[1., 2., 3., 0., 0.]]),
            np.array([[1., 0.5, -0.5]]),
            np.array([[1., 0.75, -0.5, 0., 0.]]),
            np.array([[1., -0.5, -0.5]]),
            np.array([[1., 0., 0., 4., 0.]])
            ]
    Hnum_computed = sum(H.num, [])
    Hden_computed = sum(H.den, [])
    for x in range(np.multiply(*H.shape)):
        assert_almost_equal(Hnum[x], Hnum_computed[x])
        assert_almost_equal(Hden[x], Hden_computed[x])


def test_Transfer_slicing():
    Hind = [(1, 5), (4, 1),
            (4, 5), (1, 1),
            (2, 5), (4, 2),
            (2, 5), (4, 2),
            (1, 2), (2, 1),
            (2, 5), (4, 3)]

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
    assert G.a.size == 0
    assert G._isSISO
    assert G._isgain
    assert_equal(G.d, np.array([[5.]]))

    G = State(np.eye(2))
    assert_equal(G.shape, (2, 2))
    assert G._isgain
    assert not G._isSISO

    assert_raises(ValueError, State,
                  np.eye(2),
                  np.array([[1], [2], [3]]),
                  np.array([1, 2]),
                  0)

    assert_raises(ValueError, State,
                  np.eye(2),
                  np.array([[1], [2]]),
                  np.array([1, 2, 3]),
                  0)
    assert_raises(ValueError, State,
                  np.eye(2),
                  np.array([[1], [2]]),
                  np.array([1, 2]),
                  np.array([0, 0]))


def test_State_algebra_mimo_siso():
    static_siso_state = State(5)
    static_mimo_state = State(2.0*np.eye(3))
    dynamic_siso_state = State(haroldcompanion([1, 3, 3, 1]),
                               e_i(3, -1),
                               e_i(3, 1).T,
                               0)

    dynamic_mimo_state = State(haroldcompanion([1, 3, 3, 1]),
                               e_i(3, [1, 2]),
                               np.eye(3),
                               np.zeros((3, 2)))

    dynamic_square_state = State(haroldcompanion([1, 3, 3, 1]),
                                 np.eye(3),
                                 np.eye(3),
                                 np.zeros((3, 3))
                                 )

    assert_raises(IndexError, dynamic_siso_state.__mul__, static_mimo_state)
    assert_raises(IndexError, dynamic_siso_state.__add__, static_mimo_state)
    assert_raises(IndexError, static_mimo_state.__add__, dynamic_mimo_state)
    assert_raises(IndexError, static_mimo_state.__add__, static_siso_state)
    assert_raises(IndexError, static_mimo_state.__mul__, static_siso_state)
    F = static_mimo_state * dynamic_mimo_state

    assert_almost_equal(F.c, np.eye(3)*2.0)
    assert_almost_equal((dynamic_square_state + static_mimo_state).d,
                        2*np.eye(3))


def test_State_slicing():
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


def test_concatenate_state_matrices():
    G = State(1, 2, 3, 4)
    M = concatenate_state_matrices(G)
    assert_array_equal(M, np.array([[1, 2], [3, 4]]))  # 1
    G = State(np.eye(4))
    assert_array_equal(concatenate_state_matrices(G), np.eye(4))
