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

from harold import (staircase, minimal_realization,
                    State, Transfer, matrix_slice)
from numpy import array, poly, zeros

from numpy.testing import assert_almost_equal, assert_


def test_staircase():
    M = array([[-6.5, 0.5, 6.5, -6.5, 0., 1., 0.],
               [-0.5, -5.5, -5.5, 5.5, 2., 1., 2.],
               [-0.5, 0.5, 0.5, -6.5, 3., 4., 3.],
               [-0.5, 0.5, -5.5, -0.5, 3., 2., 3.],
               [1., 1., 0., 0., 0., 0., 0.]])
    A, B, C, D = matrix_slice(M, (1, 4), corner='sw')
    a, b, c, k = staircase(A, B, C, form='o', invert=True, block_indices=True)
    assert_almost_equal(a[2:, :2], zeros((2, 2)))
    assert_almost_equal(k, array([1, 1]))


def test_minimal_realization_State():
    M = array([[-6.5, 0.5, 6.5, -6.5, 0., 1., 0.],
               [-0.5, -5.5, -5.5, 5.5, 2., 1., 2.],
               [-0.5, 0.5, 0.5, -6.5, 3., 4., 3.],
               [-0.5, 0.5, -5.5, -0.5, 3., 2., 3.],
               [1., 1., 0., 0., 0., 0., 0.]])
    G = State(*matrix_slice(M, (1, 4), corner='sw'))
    H = minimal_realization(G)
    assert_(H.a.shape, (2, 2))


def test_minimal_realization_Transfer():
    G = Transfer([1., -8., 28., -58., 67., -30.],
                 poly([1, 2, 3., 2, 3., 4, 1+(2+1e-6)*1j, 1-(2+1e-6)*1j]))
    H_f = minimal_realization(G)
    assert_almost_equal(H_f.num, array([[1]]))
    H_nf = minimal_realization(G, tol=1e-7)
    assert_almost_equal(H_nf.num, array([[1., -7., 21., -37., 30.]]))
