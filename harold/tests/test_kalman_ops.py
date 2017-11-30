"""
The MIT License (MIT)

Copyright (c) 2017 Ilhan Polat

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
from scipy.linalg import block_diag
from numpy.testing import assert_array_almost_equal, assert_, assert_raises
from harold import (State, controllability_matrix, observability_matrix,
                    is_kalman_controllable, is_kalman_observable,
                    kalman_decomposition)


# Test data is taken from python-control package tests
def test_controllability_matrix():
    A = np.arange(1, 5).reshape(2, 2)
    B = np.array([[5, 7]]).T
    Wctrue = np.array([[5., 19.], [7., 43.]])
    Wc, _, _ = controllability_matrix((A, B))
    assert_array_almost_equal(Wc, Wctrue)
    assert_(is_kalman_controllable((A, B)))
    assert_(not is_kalman_controllable((block_diag(-1, 0),
                                        np.array([[1], [0]]))))

    B = np.arange(5, 9).reshape(2, 2)
    Wctrue = np.array([[5., 6., 19., 22.], [7., 8., 43., 50.]])
    Wc, _, _ = controllability_matrix((A, B))
    assert_array_almost_equal(Wc, Wctrue)
    assert_(is_kalman_controllable((A, B)))


def test_observability_matrix():
    A = np.arange(1, 5).reshape(2, 2)
    C = np.array([[5, 7]])
    Wotrue = np.array([[5., 7.], [26., 38.]])
    Wo, _, _ = observability_matrix((A, C))
    assert_array_almost_equal(Wo, Wotrue)
    assert_(is_kalman_observable((A, C)))
    assert_(not is_kalman_observable((block_diag(-1, 0), np.array([[1, 0]]))))

    C = np.arange(5, 9).reshape(2, 2)
    Wotrue = np.array([[5., 6.], [7., 8.], [23., 34.], [31., 46.]])
    Wo, _, _ = observability_matrix((A, C))
    assert_array_almost_equal(Wo, Wotrue)


def test_pertransposed_system_dual():
    A = np.array([[1.2, -2.3], [3.4, -4.5]])
    B = np.array([[5.8, 6.9], [8., 9.1]])
    Wc, _, _ = controllability_matrix((A, B))
    Wo, _, _ = observability_matrix((A.T, B.T))
    np.testing.assert_array_almost_equal(Wc, Wo.T)


def test_simple_kalman_decomp():
    assert_raises(ValueError, kalman_decomposition, 1)
    G = State([[2, 1, 1], [5, 3, 6], [-5, -1, -4]], [[1], [0], [0]], [1, 0, 0])
    F = kalman_decomposition(G)
    J = np.array([[2, 0., -1.41421356, -1.],
                  [7.07106781, -3, -7., 0.],
                  [0, 0, 2., 0.],
                  [-1, 0, 0., 0.]])
    assert_array_almost_equal(F.a, J[:3, :3])
    assert_array_almost_equal(F.b, J[:3, [-1]])
    assert_array_almost_equal(F.c, J[[-1], :3])
