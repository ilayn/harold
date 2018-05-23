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
from numpy import eye, ones, zeros, poly
from harold import State, Transfer, discretize, undiscretize
from numpy.testing import (assert_array_almost_equal,
                           assert_equal,
                           assert_raises)
from warnings import catch_warnings, simplefilter
from harold._arg_utils import _rcond_warn

# Some tests are taken from scipy repository for comparison


def test_discretize_args():
    assert_raises(ValueError, discretize, State(5, dt=1), dt=1)
    assert_raises(ValueError, discretize, State(5), dt=1, method='randomtext')


def test_static_gain_discretization():
    Gd = discretize(State(np.eye(5)), dt=1)
    assert_equal(Gd._dt, 1)


def test_simple_zoh():
    ac = eye(2)
    bc = 0.5 * ones((2, 1))
    cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
    dc = np.array([[0.0], [0.0], [-0.33]])
    G = State(ac, bc, cc, dc)
    ad_truth = 1.648721270700128 * eye(2)
    bd_truth = 0.324360635350064 * np.ones((2, 1))
    # c and d in discrete should be equal to their continuous counterparts
    dt_ = 0.5
    H = discretize(G, dt=dt_, method='zoh')
    ad, bd, cd, dd = H.matrices
    assert_equal(H.SamplingPeriod, dt_)
    assert_array_almost_equal(ad_truth, ad)
    assert_array_almost_equal(bd_truth, bd)
    assert_array_almost_equal(cc, cd)
    assert_array_almost_equal(dc, dd)


def test_simple_foh():
    G = Transfer(1, [1, 0, 0])
    dt_ = 0.1
    H = discretize(G, dt=dt_, method='foh')
    n, d = H.polynomials
    assert_equal(H.SamplingPeriod, dt_)
    assert_array_almost_equal(n, np.array([[5, 20, 5]])/3000)
    assert_array_almost_equal(d, np.array([[1., -2., 1.]]))
    G = Transfer(1, [1, 1])
    dt_ = 0.1
    H = discretize(G, dt=dt_, method='foh')
    n, d = H.polynomials
    assert_equal(H.SamplingPeriod, dt_)
    assert_array_almost_equal(n, np.array([[0.04837418, 0.0467884]]))
    assert_array_almost_equal(d, np.array([[1., -0.90483742]]))


def test_simple_tustin():
    ac = eye(2)
    bc = 0.5 * ones((2, 1))
    cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
    dc = np.array([[0.0], [0.0], [-0.33]])
    dt_ = 0.5
    G = State(ac, bc, cc, dc)
    ad_truth = (5.0 / 3.0) * eye(2)
    bd_truth = (1.0 / 3.0) * ones((2, 1)) / np.sin(np.pi/4)
    cd_truth = np.array([[1.0, 4.0 / 3.0],
                         [4.0 / 3.0, 4.0 / 3.0],
                         [4.0 / 3.0, 1.0 / 3.0]]) * np.sin(np.pi/4)
    dd_truth = np.array([[0.291666666666667],
                         [1.0 / 3.0],
                         [-0.121666666666667]])
    H = discretize(G, dt_, method='bilinear')
    ad, bd, cd, dd = H.matrices

    assert_array_almost_equal(ad_truth, ad)
    assert_array_almost_equal(bd_truth, bd)
    assert_array_almost_equal(cd_truth, cd)
    assert_array_almost_equal(dd_truth, dd)
    assert_equal(H.SamplingPeriod, dt_)

    # Same continuous system again, but change sampling rate

    ad_truth = 1.4 * eye(2)
    bd_truth = 0.2 * ones((2, 1)) / np.sqrt(1/3)
    cd_truth = np.array([[0.9, 1.2], [1.2, 1.2], [1.2, 0.3]]) * np.sqrt(1/3)
    dd_truth = np.array([[0.175], [0.2], [-0.205]])

    dt_ = 1.0 / 3.0

    H = discretize(G, dt_, method='bilinear')
    ad, bd, cd, dd = H.matrices

    assert_array_almost_equal(ad_truth, ad)
    assert_array_almost_equal(bd_truth, bd)
    assert_array_almost_equal(cd_truth, cd)
    assert_array_almost_equal(dd_truth, dd)
    assert_equal(H.SamplingPeriod, dt_)


def test_simple_tustin_prewarp():
    # Example from B. de Moor's slides
    H = Transfer([1, 0.5, 9], [1, 5, 9])
    # prewarp at 3 rad/s
    Hd = discretize(H, dt=0.5, prewarp_at=3/2/np.pi, method='bilinear')
    assert_array_almost_equal(Hd.num, np.array([[0.591468698033,
                                                -0.0772558231247,
                                                0.500683964262]]))
    assert_array_almost_equal(Hd.den, np.array([[1.0,
                                                 -0.0772558231247,
                                                 0.0921526622952]]))
    # while we are at it test the upper Nyquist limit on prewarping
    assert_raises(ValueError, discretize, H, dt=0.5, prewarp_at=2)


def test_simple_lft():
    H = Transfer([1, 0.5, 9], [1, 5, 9])
    assert_raises(ValueError, discretize, H, dt=0.25, method='lft', q=4)
    assert_raises(ValueError, discretize, H, dt=0.25, method='lft', q=eye(3))
    Hd = discretize(H, dt=0.25, method="lft", q=np.array([[1, .5], [.5, 0]]))
    Hdf = discretize(H, dt=0.25, method=">>")
    assert_array_almost_equal(Hd.num, Hdf.num)
    assert_array_almost_equal(Hd.den, Hdf.den)


def test_undiscretize():
    ac = eye(2)
    bc = 0.5 * ones((2, 1))
    cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
    dc = np.array([[0.0], [0.0], [-0.33]])
    G = State(ac, bc, cc, dc)
    for method in ('zoh', 'foh', 'tustin', '>>', '<<'):
        print(method)
        Gd = discretize(G, dt=0.5, method=method)
        Gu = undiscretize(Gd)
        assert_array_almost_equal(Gu.a, ac)
        assert_array_almost_equal(Gu.b, bc)
        assert_array_almost_equal(Gu.c, cc)
        assert_array_almost_equal(Gu.d, dc)

    Gd = State(zeros([2, 2]), ones([2, 1]), ones([1, 2]), dt=1)
    Gd.DiscretizedWith = '<<'
    assert_raises(ValueError, undiscretize, Gd)


def test_undiscretize_zohfoh_zeroeig():
    G = Transfer(1, poly([np.pi, 1e-8, 1e-9]), dt=0.1)
    for method in ('zoh', 'foh'):
        assert_raises(ValueError, undiscretize, G, method=method)


def test_undiscretize_backeuler_nearzero_eig():
    G = Transfer(1, [1, 0, 0], 0.1)
    assert_raises(ValueError, undiscretize, G, method='<<')
    arr = np.arange(9).reshape(3, 3).astype(float)
    arr[0, 0] = 1e-15
    G = State(arr, np.random.rand(3), np.random.rand(3), dt=0.1)
    with catch_warnings():
        simplefilter("error")
        assert_raises(_rcond_warn, undiscretize, G, '<<')
