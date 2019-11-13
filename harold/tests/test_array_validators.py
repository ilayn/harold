import numpy as np
from numpy.linalg import LinAlgError
from pytest import raises as assert_raises

from harold._array_validators import _assert_square, _assert_2d, _assert_finite


def test_assert_square():
    a, b, c = np.array([[1, 2, 3]]), np.eye(2), np.array([[[1, 2], [3, 5]]])
    assert_raises(LinAlgError, _assert_square, a, b, c)


def test_assert_2d():
    a, b, c = np.array([[1, 2, 3]]), np.eye(2), np.array([[[1, 2], [3, 5]]])
    assert_raises(LinAlgError, _assert_2d, a, b, c)


def test_assert_finite():
    a, b = np.array([[1, 2, np.NaN]]), np.eye(2)
    assert_raises(LinAlgError, _assert_finite, a, b)
