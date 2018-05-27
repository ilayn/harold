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

from harold import (staircase, minimal_realization, hessenberg_realization,
                    State, Transfer, matrix_slice, cancellation_distance)

import numpy as np
from numpy import array, poly, zeros, eye, empty, triu_indices_from, zeros_like

from numpy.testing import assert_almost_equal, assert_, assert_raises


def test_staircase():
    M = array([[-6.5, 0.5, 6.5, -6.5, 0., 1., 0.],
               [-0.5, -5.5, -5.5, 5.5, 2., 1., 2.],
               [-0.5, 0.5, 0.5, -6.5, 3., 4., 3.],
               [-0.5, 0.5, -5.5, -0.5, 3., 2., 3.],
               [1., 1., 0., 0., 0., 0., 0.]])
    A, B, C, D = matrix_slice(M, (1, 4), corner='sw')
    a, b, c, T = staircase(A, B, C, form='o', invert=True)
    assert_raises(ValueError, staircase, A, B, C, form='zzz')
    assert_almost_equal(a[2:, :2], zeros((2, 2)))
    assert_almost_equal(T.T @ A @ T, a)
    a, b, c, T = staircase(A, zeros_like(B), C, form='o', invert=True)
    assert_almost_equal(b, zeros_like(B))


def test_cancellation_distance():
    # Shape checks
    assert_raises(ValueError, cancellation_distance, empty((4, 3)), 1)
    f, g = eye(4), eye(3)
    assert_raises(ValueError, cancellation_distance, f, g)


def test_minimal_realization_State():
    M = array([[-6.5, 0.5, 6.5, -6.5, 0., 1., 0.],
               [-0.5, -5.5, -5.5, 5.5, 2., 1., 2.],
               [-0.5, 0.5, 0.5, -6.5, 3., 4., 3.],
               [-0.5, 0.5, -5.5, -0.5, 3., 2., 3.],
               [1., 1., 0., 0., 0., 0., 0.]])
    G = State(*matrix_slice(M, (1, 4), corner='sw'))
    H = minimal_realization(G)
    assert H.a.shape == (2, 2)
    #
    G = State(array([[0., 1., 0., 0., 0.],
                     [-0.1, -0.5, 1., -1., 0.],
                     [0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 1.],
                     [0., 3.5, 1., -2., 2.]]),
              array([[0.], [1.], [0.], [0.], [1.]]),
              array([[0., 3.5, 1., -1., 0.]]),
              array([[1.]]))
    H = minimal_realization(G)
    assert H.a.shape == (4, 4)
    #
    G = State(array([[-2., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., -12., 4., 3.]]),
              array([[1., 0.], [0., 0.], [0., 0.], [0., 1.]]),
              array([[1., -9., 0., 0.], [0., -20., 0., 5.]]),
              array([[0., 0.], [0., 1.]]))
    H = minimal_realization(G)
    assert H.a.shape == (3, 3)


def test_minimal_realization_Transfer():
    G = Transfer([1., -8., 28., -58., 67., -30.],
                 poly([1, 2, 3., 2, 3., 4, 1+(2+1e-6)*1j, 1-(2+1e-6)*1j]))
    H_f = minimal_realization(G)
    assert_almost_equal(H_f.num, array([[1]]))
    H_nf = minimal_realization(G, tol=1e-7)
    assert_almost_equal(H_nf.num, array([[1., -7., 21., -37., 30.]]))
    H = minimal_realization(Transfer(eye(4)))
    assert H._isgain
    assert not H._isSISO
    H = minimal_realization(State(eye(4)))
    assert H._isgain
    assert not H._isSISO


def test_simple_hessenberg_trafo():
    # Made up discrete time TF
    G = Transfer([1., -8., 28., -58., 67., -30.],
                 poly([1, 2, 3., 2, 3., 4, 1 + 1j, 1 - 1j]), dt=0.1)
    H, _ = hessenberg_realization(G, compute_T=1, form='c', invert=1)
    assert_(not np.any(H.a[triu_indices_from(H.a, k=2)]))
    assert_(not np.any(H.b[:-1, 0]))
    H = hessenberg_realization(G, form='o', invert=1)
    assert_(not np.any(H.c[0, :-1]))
    assert_(not np.any(H.a.T[triu_indices_from(H.a, k=2)]))
