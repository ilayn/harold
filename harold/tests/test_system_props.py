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

from harold import Transfer, system_norm, transfer_to_state

from numpy.testing import assert_almost_equal
from pytest import raises as assert_raises


def test_system_norm_args():
    G = Transfer([100], [1, 10, 100])
    assert_raises(ValueError, system_norm, G, p='a')
    assert_raises(ValueError, system_norm, G, p=2j)
    assert_raises(ValueError, system_norm, G, p=3)


def test_system_norm_simple():
    G = Transfer([100], [1, 10, 100])
    assert_almost_equal(system_norm(G), 1.1547, decimal=5)
    assert_almost_equal(system_norm(G, p=2), 2.2360679)

    F = transfer_to_state(G)
    assert_almost_equal(system_norm(F), 1.1547, decimal=5)
    assert_almost_equal(system_norm(F, p=2), 2.2360679)
