"""
This file holds the typical checking utilities mostly copied from NumPy
private functions to accomodate for the occasional NumPy/SciPy renaming issues.
"""
import numpy as np
from numpy.linalg import LinAlgError


def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError('{}-dimensional array given. Array must be '
                              'two-dimensional'.format(a.ndim))


def _assert_square(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinAlgError('Last 2 dimensions of the array must be square'
                              '. Found an array with shape: {}x{}'.format(m, n)
                              )


def _assert_finite(*arrays):
    for a in arrays:
        if not np.isfinite(a).all():
            raise LinAlgError("Array must not contain infs or NaNs")
