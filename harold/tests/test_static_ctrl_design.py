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

from numpy import eye, array
from scipy.linalg import block_diag
from numpy.random import rand
from numpy.testing import assert_almost_equal
from pytest import raises as assert_raises
from harold import lqr, State, Transfer


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
