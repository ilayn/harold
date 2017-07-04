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
from harold import (haroldsvd, haroldker, pair_complex_numbers,
                    matrix_slice, e_i)

from scipy.linalg import block_diag, qr, solve
from numpy import fliplr, flipud, array, zeros, s_
from numpy.random import rand, shuffle
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_allclose, assert_raises,
                           )


def test_haroldsvd():
    blkdiag_mat = block_diag(*tuple([10 ** x for x in range(-4, 5)]))
    shuffler = qr(rand(9, 9), mode='full')[0]
    testmat = solve(shuffler, blkdiag_mat).dot(shuffler)
    u, s, v, r = haroldsvd(testmat, also_rank=True)

    assert_allclose(s, flipud(fliplr(blkdiag_mat)))
    assert_equal(r, 9)

    r = haroldsvd(testmat, also_rank=True, rank_tol=1.00001e-1)[-1]
    assert_equal(r, 5)


def test_haroldker():
    testmat = array([[1, 1, 1, 1]])
    testresult_r = haroldker(testmat)
    testresult_l = haroldker(testmat, side='left')
    assert_almost_equal(testmat.dot(testresult_r), zeros((1, 3)))
    assert_equal(testresult_l, array([[0.]]))


def test_cplxpair():
    test_array = [
                  1 + 1j - 0.9e-8,
                  1 + 1j + 0.5e-8,
                  1 - 1j - 1.0e-8j,
                  1 - 1j + 0.3e-8j,
                  2.0,
                  -1.0,
                  -2 - 5j,
                  -2 + 5j,
                  -2 - 4j,
                  -1 - 2j,  # for odd numbered
                  "some text",  # for type check
    ]

    # Type check
    assert_raises(ValueError, pair_complex_numbers, test_array)
    del test_array[-1:]
    # Real part failure
    assert_raises(ValueError, pair_complex_numbers, test_array)
    del test_array[-1:]
    # odd numbered complex check
    assert_raises(ValueError, pair_complex_numbers, test_array)
    del test_array[-1]
    # tolerance check
    assert_raises(ValueError, pair_complex_numbers, test_array)
    shuffle(test_array)
    t1 = pair_complex_numbers(test_array, tol=1e-7)
    tt1 = array([-1.00000000+0.j, 2.00000000+0.j,
                 -2.00000000-5.j, -2.00000000+5.j,
                 0.99999999-1.j, 0.99999999+1.j,
                 1.00000000-1.j, 1.00000000+1.j])

    assert_almost_equal(t1, tt1, decimal=7)


def test_e_i():
    assert_almost_equal(e_i(7, 5, output='r'),
                        array([[0., 0., 0., 0., 0., 1., 0.]])
                        )

    assert_almost_equal(e_i(5, [0, 4, 4, 4, 1]),
                        array([[1., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 1.],
                               [0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0.],
                               [0., 1., 1., 1., 0.]])
                        )

    assert_almost_equal(e_i(5, s_[1:3]),
                        array([[0., 0.],
                               [1., 0.],
                               [0., 1.],
                               [0., 0.],
                               [0., 0.]])
                        )

    assert_almost_equal(e_i(5, slice(1, 5, 2), output='r'),
                        array([[0., 1., 0., 0., 0.],
                               [0., 0., 0., 1., 0.]])
                        )


def test_matrix_slice():
    A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a, b, c, d = matrix_slice(A, (1, 1))
    assert_almost_equal(a, array([[1]]))
    assert_almost_equal(b, array([[2, 3]]))
    assert_almost_equal(c, array([[4], [7]]))
    assert_almost_equal(d, array([[5, 6], [8, 9]]))

    a, b, c, d = matrix_slice(A, (2, 2), 'sw')
    assert_almost_equal(a, array([[1, 2]]))
    assert_almost_equal(b, array([[3]]))
    assert_almost_equal(c, array([[4, 5], [7, 8]]))
    assert_almost_equal(d, array([[6], [9]]))

    a, b, c, d = matrix_slice(A, (0, 0))
    assert_almost_equal(d, array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    for x in (a, b, c):
        assert_equal(x.size, 0)
