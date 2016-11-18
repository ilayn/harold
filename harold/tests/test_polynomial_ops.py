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
    w = haroldgcd(a, b, c, d, e)
    z = haroldgcd(a, b, c, d, e, f)
    assert_almost_equal(w, array([1, 1]))
    assert_almost_equal(z, array([1]))
    # Test case 1
    a = array([1, 4, 6, 6, 5, 2])
    b = array([1, 14, 71, 154, 120])
    c = array(haroldpoly([-2]*10))
    x = haroldgcd(a, b, c)
    assert_almost_equal(x, array([1, 2]))


def test_haroldlcm():
        # Test the least common multiple
        a, b = haroldlcm(array([1, 3, 0, -4]),
                         array([1, -4, -3, 18]),
                         array([1, -4, 3]),
                         array([1, -2, -8])
                         )
        assert_almost_equal(a,
                            array([1., -7., 3., 59., -68., -132., 144.]))
        # Test the multipliers
        for ind, x in enumerate([array([1., -10., 33., -36.]),
                                 array([1., -3., -6., 8.]),
                                 array([1., -3., -12., 20.,  48.]),
                                 array([1., -5., 1., 21., -18.])]):
            assert_almost_equal(b[ind], x)
