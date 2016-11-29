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
from numpy.random import rand
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises
from harold import lyapunov_eq_solver


def test_lyapunov_eq_arguments():
    assert_raises(ValueError, lyapunov_eq_solver, [1, 2], np.eye(2))
    assert_raises(ValueError, lyapunov_eq_solver, np.eye(2), [1, 2])
    assert_raises(ValueError, lyapunov_eq_solver, np.eye(2), np.eye(2), [1, 2])
    assert_raises(ValueError, lyapunov_eq_solver,
                  np.eye(2), np.eye(3), np.eye(2))
    assert_raises(ValueError, lyapunov_eq_solver,
                  np.eye(2), np.eye(2), np.eye(3))
    assert_raises(ValueError, lyapunov_eq_solver, 1, 1, 1, form='a')


def test_lyapunov_eq_small_problems():
    A = rand(2, 2)
    Y = rand(2, 2)
    Y = Y + Y.T
    E = rand(2, 2)
    X = lyapunov_eq_solver(A, Y, form='c')
    Res = A.T @ X + X @ A + Y
    assert_almost_equal(Res, np.zeros((2, 2)))  # c, regular
    X = lyapunov_eq_solver(A, Y, E, form='c')
    Res = A.T @ X @ E + E.T @ X @ A + Y
    assert_almost_equal(Res, np.zeros((2, 2)))  # c, generalized
    X = lyapunov_eq_solver(A, Y, form='d')
    Res = A.T @ X @ A - X + Y
    assert_almost_equal(Res, np.zeros((2, 2)))  # d, regular
    X = lyapunov_eq_solver(A, Y, E, form='d')
    Res = A.T @ X @ A - E.T @ X @ E + Y
    assert_almost_equal(Res, np.zeros((2, 2)))  # d, generalized


def test_lyapunov_eq_random_problems():
    n = 10
    A = rand(n, n)
    Y = rand(n, n)
    Y = Y + Y.T
    E = rand(n, n)
    X = lyapunov_eq_solver(A, Y, form='c')
    Res = A.T @ X + X @ A + Y
    assert_almost_equal(Res, np.zeros((n, n)))  # c, regular
    X = lyapunov_eq_solver(A, Y, E, form='c')
    Res = A.T @ X @ E + E.T @ X @ A + Y
    assert_almost_equal(Res, np.zeros((n, n)))  # c, generalized
    X = lyapunov_eq_solver(A, Y, form='d')
    Res = A.T @ X @ A - X + Y
    assert_almost_equal(Res, np.zeros((n, n)))  # d, regular
    X = lyapunov_eq_solver(A, Y, E, form='d')
    Res = A.T @ X @ A - E.T @ X @ E + Y
    assert_almost_equal(Res, np.zeros((n, n)))  # d, generalized
