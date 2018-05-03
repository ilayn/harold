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

__all__ = ['haroldsvd', 'haroldker', 'pair_complex_numbers', 'e_i',
           'matrix_slice']


def haroldsvd(A, also_rank=False, rank_tol=None):
    """
    This is a wrapper/container function of both the SVD decomposition
    and the rank computation. Since the regular rank computation is
    implemented via SVD it doesn't make too much sense to recompute
    the SVD if we already have the rank information. Thus instead of
    typing two commands back to back for both the SVD and rank, we
    return both. To reduce the clutter, the rank information is supressed
    by default.

    numpy svd is a bit strange because it compresses and looses the
    S matrix structure. From the manual, it is advised to use
    u.dot(np.diag(s).dot(v)) for recovering the original matrix. But
    that won't work for rectangular matrices. Hence it recreates the
    rectangular S matrix of U,S,V triplet.

    Parameters
    ----------

    A : (m,n) array_like
        Matrix to be decomposed
    also_rank : bool, optional
        Whether the rank of the matrix should also be reported or not.
        The returned rank is computed via the definition taken from the
        official numpy.linalg.matrix_rank and appended here.
    rank_tol : {None,float} optional
        The tolerance used for deciding the numerical rank. The default
        is set to None and uses the default definition of matrix_rank()
        from numpy.

    Returns
    -------

    U,S,V : {(m,m),(m,n),(n,n)} array_like
        Decomposed-form matrices
    r : integer
        If the boolean "also_rank" is true, this variable is the numerical
        rank of the matrix D

    """
    try:
        A = np.atleast_2d(np.array(A))
    except TypeError:
        raise TypeError('Incompatible argument, use either list of lists'
                        'or native numpy arrays for svd.')
    except ValueError:
        raise ValueError('The argument cannot be cast as an array.')

    p, m = A.shape
    u, s, v = np.linalg.svd(A, full_matrices=True)
    diags = np.zeros((p, m))  # Reallocate the s matrix of u,s,v
    for index, svalue in enumerate(s):  # Repopulate the diagoanal with svds
        diags[index, index] = svalue

    if also_rank:  # Copy the official rank computation
        if rank_tol is None:
            rank_tol = s.max() * max(p, m) * np.spacing(1.)
        r = sum(s > rank_tol)
        return u, diags, v, r

    return u, diags, v


def haroldker(N, side='right'):
    """
    This function is a straightforward basis computation for the right/left
    nullspace for rank deficient or fat/tall matrices.

    It simply returns the remaining columns of the right factor of the
    singular value decomposition whenever applicable. Otherwise returns
    a zero vector such that it has the same number of rows as the columns
    of the argument, hence the dot product makes sense.

    The basis columns have unity 2-norm except for the trivial zeros.

    Parameters
    ----------
    N : (m,n) array_like
        Matrix for which the nullspace basis to be computed
    side : {'right','left'} string
        The switch for the right or left nullspace computation.

    Returns
    -------
    Nn : (n,dim) array_like
        Basis for the nullspace. dim is the dimension of the nullspace. If
        the nullspace is trivial then dim is 1 for consistent 2D array output

    """
    if side not in ('left', 'right'):
        raise ValueError('side keyword only takes "left,right" as arguments')

    try:
        A = np.atleast_2d(np.array(N))
    except TypeError:
        raise TypeError('Incompatible argument, use either list of lists'
                        'or native numpy arrays for svd.')
    except ValueError:
        raise ValueError('The argument cannot be cast as an array.')

    if side == 'left':
        A = A.conj().T

    m, n = A.shape

    if A.size <= 1:
        # don't bother
        return np.array([[0]])

    V, r = haroldsvd(A, also_rank=True)[2:]

    if r == min(m, n) and m >= n:
        # If full rank and not fat, return trivial zero
        return np.zeros((A.shape[1], 1))
    else:
        return V[:, r:]


def pair_complex_numbers(a, tol=1e-9, realness_tol=1e-9,
                         positives_first=False, reals_first=True):
    """
    Given an array-like somearray, it first tests and clears out small
    imaginary parts via `numpy.real_if_close`. Then pairs complex numbers
    together as consecutive entries. A real array is returned as is.

    Parameters
    ----------

    a : array_like
        Array like object needs to be paired
    tol: float
        The sensitivity threshold for the real and complex parts to be
        assumed as equal.
    realness_tol: float
        The sensitivity threshold for the complex parts to be assumed
        as zero.
    positives_first: bool
        The boolean that defines whether the positive complex part
        should come first for the conjugate pairs
    reals_first: bool
        The boolean that defines whether the real numbers are at the
        beginning or the end of the resulting array.

    Returns
    -------

    paired_array : ndarray
        The resulting paired array

    """
    try:
        array_r_j = np.array(a, dtype='complex').flatten()
    except:
        raise ValueError('Something in the argument array prevents me to '
                         'convert the entries to complex numbers.')

    # is there anything to pair?
    if array_r_j.size == 0:
        return np.array([], dtype='complex')

    # is the array 1D or more?
    if array_r_j.ndim > 1 and np.min(array_r_j.shape) > 1:
        raise ValueError('Currently, I can\'t deal with matrices, so I '
                         'need 1D arrays.')

    # A shortcut for splitting a complex array into real and imag parts
    def return_imre(arr):
        return np.real(arr), np.imag(arr)

    # a close to realness function that operates element-wise
    real_if_close_array = np.vectorize(
            lambda x: np.real_if_close(x, realness_tol), otypes=[np.complex_],
            doc='Elementwise numpy.real_if_close')

    array_r_j = real_if_close_array(array_r_j)
    array_r, array_j = return_imre(array_r_j)

    # are there any complex numbers to begin with or all reals?
    # if not kick the argument back as a real array
    imagness = np.abs(array_j) >= realness_tol

    # perform the imaginary entry separation once
    array_j_ent = array_r_j[imagness]
    num_j_ent = array_j_ent.size

    if num_j_ent == 0:
        # If no complex entries exist sort and return unstable first
        return np.sort(array_r)

    elif num_j_ent % 2 != 0:
        # Check to make sure there are even number of complex numbers
        # Otherwise stop with "odd number --> no pair" error.
        raise ValueError('There are odd number of complex numbers to '
                         'be paired!')
    else:

        # Still doesn't distinguish whether they are pairable or not.
        sorted_array_r_j = np.sort_complex(array_j_ent)
        sorted_array_r, sorted_array_j = return_imre(sorted_array_r_j)

        # Since the entries are now sorted and appear as pairs,
        # when summed with the next element the result should
        # be very small

        if any(np.abs(sorted_array_r[:-1:2] - sorted_array_r[1::2]) > tol):
            # if any difference is bigger than the tolerance
            raise ValueError('Pairing failed for the real parts.')

        # Now we have sorted the real parts and they appear in pairs.
        # Next, we have to get rid of the repeated imaginary, if any,
        # parts in the  --... , ++... pattern due to sorting. Note
        # that the non-repeated imaginary parts now have the pattern
        # -,+,-,+,-,... and so on. So we can check whether sign is
        # not alternating for the existence of the repeatedness.

        def repeat_sign_test(myarr, mylen):
            # Since we separated the zero imaginary parts now any sign
            # info is either -1 or 1. Hence we can test whether -1,1
            # pattern is present. Otherwise we count how many -1s occured
            # double it for the repeated region. Then repeat until we
            # we exhaust the array with a generator.

            x = 0
            myarr_sign = np.sign(myarr).astype(int)
            while x < mylen:
                if np.array_equal(myarr_sign[x:x+2], [-1, 1]):
                    x += 2
                elif np.array_equal(myarr_sign[x:x+2], [1, -1]):
                    myarr[x:x+2] *= -1
                    x += 2
                else:
                    still_neg = True
                    xl = x+2
                    while still_neg:
                        if myarr_sign[xl] == 1:
                            still_neg = False
                        else:
                            xl += 1

                    yield x, xl - x
                    x += 2*(xl - x)

        for ind, l in repeat_sign_test(sorted_array_j, num_j_ent):
            indices = np.dstack(
                        (range(ind, ind+l), range(ind+2*l-1, ind+l-1, -1))
                        )[0].reshape(1, -1)

            sorted_array_j[ind:ind+2*l] = sorted_array_j[indices]

        if any(np.abs(sorted_array_j[:-1:2] + sorted_array_j[1::2]) > tol):
            # if any difference is bigger than the tolerance
            raise ValueError('Pairing failed for the complex parts.')

        # Finally we have a properly sorted pairs of complex numbers
        # We can now combine the real and complex parts depending on
        # the choice of positives_first keyword argument

        # Force entries to be the same for each of the pairs.
        sorted_array_j = np.repeat(sorted_array_j[::2], 2)
        paired_cmplx_part = np.repeat(sorted_array_r[::2], 2).astype(complex)

        if positives_first:
            sorted_array_j[::2] *= -1
        else:
            sorted_array_j[1::2] *= -1

        paired_cmplx_part += sorted_array_j*1j

        if reals_first:
            return np.r_[np.sort(array_r_j[~imagness]), paired_cmplx_part]
        else:
            return np.r_[paired_cmplx_part, np.sort(array_r_j[~imagness])]


def e_i(width, nth=0, output='c'):
    """
    Returns the ``nth`` column(s) of the identity matrix with shape
    ``(width,width)``. Slicing is permitted with the ``nth`` parameter.

    The output is returned without slicing an intermediate identity matrix
    hence can be used without allocating the whole array.

    Parameters
    ----------
    width : int
        The size of the identity matrix from which the columns are taken
    nth : 1D index array
        A sequence/index expression that selects the requested columns/rows
        of the identity matrix. The index starts with zero denoting the first
        column.
    output : str
        This switches the shape of the output; if ``'r'`` is given then
        the rows are returned. The default is ``'c'`` which returns columns
    Returns
    -------
    E : ndarray
      The resulting row/column subset of the identity matrix

    Examples
    --------
    >>> e_i(7, 5, output='r') # The 5th row of 7x7 identity matrix
    array([[ 0.,  0.,  0.,  0.,  0.,  1.,  0.]])

    >>> e_i(5, [0, 4, 4, 4, 1])  # Sequences can also be used
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.]])

    >>> e_i(5,np.s_[1:3])  # or NumPy index expressions
    array([[ 0.,  0.],
           [ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  0.],
           [ 0.,  0.]])

    >>> e_i(5,slice(1,5,2),output='r')  # or Python slice objects
    array([[ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.]])

    """
    col_inds = np.atleast_1d(np.arange(width)[nth])
    m = col_inds.size
    E = np.zeros((width, m)) if output == 'c' else np.zeros((m, width))

    if output == 'c':
        for ind, x in enumerate(col_inds):
            E[x, ind] = 1
    else:
        for ind, x in enumerate(col_inds):
            E[ind, x] = 1

    return E


def matrix_slice(M, corner_shape, corner='nw'):
    """
    Takes a two dimensional array ``M`` and slices into four parts dictated
    by the ``corner_shape`` and the corner string ``corner``. ::

            m   n
        p [ A | B ]
          [-------]
        q [ C | D ]

    If the given corner and the shape is the whole array then the remaining
    arrays are returned as empty arrays, ``numpy.array([])``.

    Parameters
    ----------
    M : ndarray
        2D input matrix
    corner_shape : tuple
        An integer valued 2-tuple for the shape of the corner
    corner : str
        Defines which corner should be used to start slicing. Possible
        options are the compass abbreviations: ``'nw', 'ne', 'sw', 'se'``.
        The default is the north-west corner.

    Returns
    -------
    A : ndarray
        Upper left corner slice
    B : ndarray
        Upper right corner slice
    C : ndarray
        Lower left corner slice
    D : ndarray
        Lower right corner slice

    Examples
    --------
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> matrix_slice(A,(1,1))
    (array([[1]]),
     array([[2, 3]]),
     array([[4],
            [7]]),
     array([[5, 6],
            [8, 9]])
    )
    >>> matrix_slice(A, (2,2), 'sw')
    (array([[1, 2]]),
     array([[3]]),
     array([[4, 5],
            [7, 8]]),
     array([[6],
            [9]])
     )
    >>> matrix_slice(A, (0, 0))  % empty A
    (array([], shape=(0, 0), dtype=int32),
     array([], shape=(0, 3), dtype=int32),
     array([], shape=(3, 0), dtype=int32),
     array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]))
    """
    if corner not in ('ne', 'nw', 'se', 'sw'):
        raise ValueError('The corner string needs to be one of'
                         '"ne, nw, se, sw".')

    x, y = M.shape
    z, w = corner_shape
    if corner == 'nw':
        p, m = z, w
    elif corner == 'ne':
        p, m = x, y - w
    elif corner == 'sw':
        p, m = x - z, w
    else:
        p, m = x - z, y - w

    return M[:p, :m], M[:p, m:], M[p:, :m], M[p:, m:]
