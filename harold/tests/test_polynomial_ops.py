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

from harold import haroldgcd, haroldlcm, haroldpoly
from numpy import array, eye

from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises


def test_haroldgcd():
    # Test case 0
    a = array([1, 3, 2])
    b = array([1, 4, 6, 4, 1])
    c = array([0, 1, 1])
    d = array([])
    e = eye(2)
    f = array(1)
    assert_raises(ValueError, haroldgcd, a, e)
    z = haroldgcd(a, b, c, d, f)
    assert_almost_equal(z, array([1]))
    # Test case 1
    a = array([1, 4, 6, 6, 5, 2])
    b = array([1, 14, 71, 154, 120])
    c = array(haroldpoly([-2]*10))
    x = haroldgcd(a, b, c)
    assert_almost_equal(x, array([1, 2]))


def test_haroldlcm():
    assert_raises(ValueError, haroldlcm, [1, 3, 2], [[1, 4, 3], [2, 3, 5]])
    assert_raises(ValueError, haroldlcm, [1, 3, 2], [])

    a, b = haroldlcm([1, 3, 2], [1, 4, 3+5j])
    assert_almost_equal(a, array([1., 6., 11., 6.]))
    assert_almost_equal(b[0], array([1., 3.]))
    assert_almost_equal(b[1], array([1., 2.]))

    # Test the least common multiple
    a, b = haroldlcm([1, 3, 0, -4], [1, -4, -3, 18], [1, -4, 3], [1, -2, -8])
    assert_almost_equal(a, array([1., -7., 3., 59., -68., -132., 144.]))
    # Test the multipliers
    for ind, x in enumerate([array([1., -10., 33., -36.]),
                             array([1., -3., -6., 8.]),
                             array([1., -3., -12., 20.,  48.]),
                             array([1., -5., 1., 21., -18.])]):
        assert_almost_equal(b[ind], x)

    a, b = haroldlcm(1, 1, [1, 3, 0, -4], 1)
    assert_almost_equal(a, array([1, 3, 0, -4]))
    assert_almost_equal(b[2], array([1.]))
    for x in [0, 1, 3]:
        assert_almost_equal(b[x], array([1, 3, 0, -4]))

    a, b = haroldlcm(1, 2, 3, 4)
    for x in range(4):
        assert_almost_equal(b[x], array([1.]))
    assert_almost_equal(a, array([1.]))
