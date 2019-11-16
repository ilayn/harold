import collections
import numpy as np
from numpy.linalg import matrix_rank
import scipy.signal as sig
from scipy.linalg import block_diag, lu, matrix_balance, solve, lstsq
from ._aux_linalg import haroldsvd, e_i

__all__ = ['haroldlcm', 'haroldgcd', 'haroldcompanion', 'haroldpoly',
           'haroldpolyadd', 'haroldpolymul', 'haroldpolydiv']


def haroldlcm(*args, compute_multipliers=True, cleanup_threshold=1e-9):
    """
    Takes n-many 1D numpy arrays and computes the numerical
    least common multiple polynomial. The polynomials are
    assumed to be in decreasing powers, e.g. s^2 + 5 should
    be given as ``[1,0,5]``

    Returns a numpy array holding the polynomial coefficients
    of LCM and a list, of which entries are the polynomial
    multipliers to arrive at the LCM of each input element.

    For the multiplier computation, a variant of [1]_ is used.

    Parameters
    ----------
    args : iterable
        Input arrays. 1-D arrays or array_like sequences of polynomial
        coefficients
    compute_multipliers : bool, optional
        After the computation of the LCM, this switch decides whether the
        multipliers of the given arguments should be computed or skipped.
        A multiplier in this context is ``[1,3]`` for the argument ``[1,2]``
        if the LCM turns out to be ``[1,5,6]``.
    cleanup_threshold : float
        The computed polynomials might contain some numerical noise and after
        finishing everything this value is used to clean up the tiny entries.
        Set this value to zero to turn off this behavior. The default value
        is :math:`10^{-9}`.

    Returns
    --------
    lcmpoly : ndarray
        Resulting 1D polynomial coefficient array for the LCM.
    mults : list
        The multipliers given as a list of 1D arrays, for each given argument.

    Notes
    -----
    If complex-valued arrays are given, only real parts are taken into account.

    Examples
    --------
    >>> a , b = haroldlcm([1,3,0,-4], [1,-4,-3,18], [1,-4,3], [1,-2,-8])
    >>> a
    array([   1.,   -7.,    3.,   59.,  -68., -132.,  144.]
    >>> b
    [array([  1., -10.,  33., -36.]),
     array([  1.,  -3.,  -6.,   8.]),
     array([  1.,  -3., -12.,  20.,  48.]),
     array([  1.,  -5.,   1.,  21., -18.])]
    >>> np.convolve([1, 3, 0, -4], b[0]) # or haroldpolymul() for poly mult
    (array([   1.,   -7.,    3.,   59.,  -68., -132.,  144.]),

    References
    ----------
    .. [1] Karcanias, Mitrouli, "System theoretic based characterisation and
        computation of the least common multiple of a set of polynomials",
        2004, :doi:`10.1016/j.laa.2003.11.009`

    """
    # Regularize the arguments
    args = [np.array(a).squeeze().real for a in args]
    # Add dimension if any scalar arrays such as np.array(1)
    args = [a if a.ndim > 0 else np.atleast_1d(a) for a in args]
    if not all([x.ndim == 1 for x in args]):
        raise ValueError('Input arrays must be 1D.')
    if not all([x.size > 0 for x in args]):
        raise ValueError('Empty arrays are not allowed.')

    # All scalars
    if all([x.size == 1 for x in args]):
        if compute_multipliers:
            return np.array([1.]), [np.array([1.]) for _ in range(len(args))]
        else:
            return np.array([1.])

    # Remove if there are constant polynomials but return their multiplier!
    poppedargs = [x for x in args if x.size > 1]
    # Get the index number of the ones that are popped
    p_ind, l_ind = [], []
    [p_ind.append(ind) if x.size == 1 else l_ind.append(ind)
        for ind, x in enumerate(args)]

    # If there are more than one nonconstant polynomial to consider
    if len(poppedargs) > 1:
        a = block_diag(*(map(haroldcompanion, poppedargs)))
        b = np.concatenate([e_i(x.size-1, -1) for x in poppedargs])
        n = a.shape[0]

        # Balance A
        As, (sca, _) = matrix_balance(a, permute=False, separate=True)
        Bs = b*np.reciprocal(sca)[:, None]

        # Computing full c'bility matrix is redundant we just need to see where
        # the rank drop is (if any!). Due to matrix power, things grow quickly!
        C = Bs
        for _ in range(n-1):
            C = np.hstack([C, As @ C[:, [-1]]])
            if matrix_rank(C) != C.shape[1]:
                break
        else:
            # No break
            C = np.hstack([C, As @ C[:, [-1]]])

        cols = C.shape[1]
        _, s, v = haroldsvd(C)
        temp = s @ v
        lcmpoly = solve(temp[:cols-1, :-1], -temp[:cols-1, -1])
        # Add monic coefficient and flip
        lcmpoly = np.append(lcmpoly, 1)[::-1]
    else:
        lcmpoly = np.trim_zeros(poppedargs[0], 'f')
        lcmpoly = lcmpoly/lcmpoly[0]

    if compute_multipliers:
        n_lcm = lcmpoly.size - 1
        if len(poppedargs) > 1:
            c = block_diag(*[e_i(x.size-1, 0).T for x in poppedargs]) * sca
            b_lcm, _, _, _ = lstsq(C[:c.shape[1], :-1], Bs)
            c_lcm = c @ C[:c.shape[1], :-1]

            # adj(sI-A) formulas with A being a companion matrix. Use a 3D
            # array where x,y,z = adj(sI-A)[x,y] and z is the coefficient array
            adjA = np.zeros([n_lcm, n_lcm, n_lcm])
            # fill in the adjoint
            for x in range(n_lcm):
                # Diagonal terms
                adjA[x, x, :n_lcm-x] = lcmpoly[:n_lcm-x]
                for y in range(n_lcm):
                    if y < x:  # Upper Triangular terms
                        adjA[x, y, x-y:] = adjA[x, x, :n_lcm-(x-y)]
                    elif y > x:  # Lower Triangular terms
                        adjA[x, y, n_lcm-y:n_lcm+1-y+x] = \
                                                    -lcmpoly[-x-1:n_lcm+1]
            # C*adj(sI-A)*B
            mults = c_lcm @ np.sum(adjA * b_lcm, axis=1)
        else:
            mults = np.zeros((1, n_lcm))
            mults[0, -1] = 1.

        if len(p_ind) > 0:
            temp = np.zeros((len(args), lcmpoly.size), dtype=float)
            temp[p_ind] = lcmpoly
            temp[l_ind, 1:] = mults
            mults = temp

        lcmpoly[abs(lcmpoly) < cleanup_threshold] = 0.
        mults[abs(mults) < cleanup_threshold] = 0.
        mults = [np.trim_zeros(z, 'f') for z in mults]
        return lcmpoly, mults
    else:
        return lcmpoly


def haroldgcd(*args):
    """
    Takes 1D numpy arrays and computes the numerical greatest common
    divisor polynomial. The polynomials are assumed to be in decreasing
    powers, e.g. :math:`s^2 + 5` should be given as ``numpy.array([1,0,5])``.

    Returns a numpy array holding the polynomial coefficients
    of GCD. The GCD does not cancel scalars but returns only monic roots.
    In other words, the GCD of polynomials :math:`2` and :math:`2s+4` is
    still computed as :math:`1`.

    Parameters
    ----------
    args : iterable
        A collection of 1D array_likes.

    Returns
    --------
    gcdpoly : ndarray
        Computed GCD of args.

    Examples
    --------
    >>> a = haroldgcd(*map(haroldpoly,([-1,-1,-2,-1j,1j],
                                       [-2,-3,-4,-5],
                                       [-2]*10)))
    >>> a
    array([ 1.,  2.])

    .. warning:: It uses the LU factorization of the Sylvester matrix.
                 Use responsibly. It does not check any certificate of
                 success by any means (maybe it will in the future).
                 I have played around with ERES method but probably due
                 to my implementation, couldn't get satisfactory results.
                 I am still interested in better methods.
    """
    raw_arr_args = [np.atleast_1d(np.squeeze(x)) for x in args]
    arr_args = [np.trim_zeros(x, 'f') for x in raw_arr_args if x.size > 0]
    dimension_list = [x.ndim for x in arr_args]

    # do we have 2d elements?
    if max(dimension_list) > 1:
        raise ValueError('Input arrays must be 1D arrays, rows, or columns')

    degree_list = np.array([x.size-1 for x in arr_args])
    max_degree = np.max(degree_list)
    max_degree_index = np.argmax(degree_list)

    try:
        # There are polynomials of lesser degree
        second_max_degree = np.max(degree_list[degree_list < max_degree])
    except ValueError:
        # all degrees are the same
        second_max_degree = max_degree

    n, p, h = max_degree, second_max_degree, len(arr_args) - 1

    # If a single item is passed then return it back
    if h == 0:
        return arr_args[0]

    if n == 0:
        return np.array([1])

    if n > 0 and p == 0:
        return arr_args.pop(max_degree_index)

    # pop out the max degree polynomial and zero pad
    # such that we have n+m columns
    S = np.array([np.hstack((
            arr_args.pop(max_degree_index),
            np.zeros((1, p-1)).squeeze()
            ))]*p)

    # Shift rows to the left
    for rows in range(S.shape[0]):
        S[rows] = np.roll(S[rows], rows)

    # do the same to the remaining ones inside the regular_args
    for item in arr_args:
        _ = np.array([np.hstack((item, [0]*(n+p-item.size)))]*(
                      n+p-item.size+1))
        for rows in range(_.shape[0]):
            _[rows] = np.roll(_[rows], rows)
        S = np.r_[S, _]

    rank_of_sylmat = np.linalg.matrix_rank(S)

    if rank_of_sylmat == min(S.shape):
        return np.array([1])
    else:
        p, l, u = lu(S)

    u[abs(u) < 1e-8] = 0
    for rows in range(u.shape[0]-1, 0, -1):
        if not any(u[rows, :]):
            u = np.delete(u, rows, 0)
        else:
            break

    gcdpoly = np.real(np.trim_zeros(u[-1, :], 'f'))
    # make it monic
    gcdpoly /= gcdpoly[0]

    return gcdpoly


def haroldcompanion(somearray):
    """
    Takes a 1D numpy array or list and returns the companion matrix
    of the monic polynomial of somearray. Hence ``[0.5,1,2]`` will be first
    converted to ``[1,2,4]``.

    Examples
    --------

    >>> haroldcompanion([2,4,6])
    array([[ 0.,  1.],
           [-3., -2.]])
    >>> haroldcompanion([1,3])
    array([[-3.]])
    >>> haroldcompanion([1])
    array([], dtype=float64)

    """
    if not isinstance(somearray, (list, np.ndarray)):
        raise TypeError('Companion matrices are meant only for '
                        '1D lists or 1D Numpy arrays. I found '
                        'a \"{0}\"'.format(type(somearray).__name__))

    if len(somearray) == 0:
        return np.array([])

    # regularize to flat 1D np.array
    somearray = np.array(somearray).flatten()

    ta = np.trim_zeros(somearray, 'f')
    # convert to monic polynomial.
    # Note: ta *=... syntax doesn't autoconvert to float
    ta = np.array(1/ta[0])*ta
    ta = -ta[-1:0:-1]
    n = ta.size

    if n == 0:  # Constant polynomial
        return np.array([])

    elif n == 1:  # First-order --> companion matrix is a scalar
        return np.atleast_2d(np.array(ta))

    else:  # Other stuff
        return np.vstack((np.hstack((np.zeros((n-1, 1)), np.eye(n-1))), ta))


def haroldpoly(rootlist):
    """
    Takes a 1D array-like numerical elements as roots and forms the polynomial
    """
    if isinstance(rootlist, collections.abc.Iterable):
        r = np.array([x for x in rootlist], dtype=complex)
    else:
        raise TypeError('The argument must be something iterable,\nsuch as '
                        'list, numpy array, tuple etc. I don\'t know\nwhat '
                        'to do with a \"{0}\" object.'
                        ''.format(type(rootlist).__name__))

    n = r.size
    if n == 0:
        return np.ones(1)
    else:
        p = np.array([0.+0j for x in range(n+1)], dtype=complex)
        p[0] = 1  # Monic polynomial
        p[1] = -rootlist[0]
        for x in range(1, n):
            p[x+1] = -p[x]*r[x]
            for y in range(x, 0, -1):
                p[y] -= p[y-1] * r[x]
        return p


def haroldpolyadd(*args, trim_zeros=True):
    """
    A wrapper around NumPy's :func:`numpy.polyadd` but allows for multiple
    args and offers a trimming option.

    Parameters
    ----------
    args : iterable
        An iterable with 1D array-like elements.
    trim_zeros : bool, optional
        If True, the zeros at the front of the input and output arrays are
        truncated. Default is True.

    Returns
    -------
    p : ndarray
        The polynomial coefficients of the sum.

    Examples
    --------
    >>> a = np.array([2, 3, 5, 8])
    >>> b = np.array([1, 3, 4])
    >>> c = np.array([6, 9, 10, -8, 6])
    >>> haroldpolyadd(a, b, c)
    array([ 6., 11., 14.,  0., 18.])
    >>> d = np.array([-2, -4 ,3, -1])
    >>> haroldpolyadd(a, b, d)
    array([ 0.,  0., 11., 11.])
    >>> haroldpolyadd(a, b, d, trim_zeros=False)
    array([ 0.,  0., 11., 11.])

    """
    if trim_zeros:
        trimmedargs = [np.trim_zeros(x, 'f') for x in args]
    else:
        trimmedargs = args

    degs = [len(m) for m in trimmedargs]  # Get the max len of args
    s = np.zeros((1, max(degs)))
    for ind, x in enumerate(trimmedargs):
        s[0, max(degs)-degs[ind]:] += np.real(x)

    return s[0]


def haroldpolymul(*args, trim_zeros=True):
    """
    Simple wrapper around the :func:`numpy.convolve` function for polynomial
    multiplication with multiple args. The arguments are passed through
    the left zero trimming function first.

    See Also
    --------
    haroldpolydiv, :func:`numpy.convolve`, :func:`scipy.signal.convolve`

    Parameters
    ----------
    args : iterable
        An iterable with 1D array-like elements.
    trim_zeros : bool, optional
        If True, the zeros at the front of the arrays are truncated.
        Default is True.

    Returns
    -------
    p : ndarray
        The polynomial coefficients of the product.

    Examples
    --------

    >>> haroldpolymul([0,2,0], [0,0,0,1,3,3,1], [0,0.5,0.5])
    array([ 1.,  4.,  6.,  4.,  1.,  0.])

    """

    if trim_zeros:
        trimmedargs = [np.trim_zeros(x.flatten(), 'f') for x in args]
    else:
        trimmedargs = args

    p = trimmedargs[0]

    for x in trimmedargs[1:]:
        if x.size == 0:  # it was all zeros
            p = []
            break
        p = np.convolve(p, x)

    return p if np.any(p) else np.array([0.])


def haroldpolydiv(dividend, divisor):
    """
    Polynomial division wrapped around :func:`scipy.signal.deconvolve`
    function. Takes two arguments and divides the first
    by the second.

    Parameters
    ----------
    dividend : (n,) array_like
        The polynomial to be divided
    divisor : (m,) array_like
        The polynomial that divides

    Returns
    -------
    factor : ndarray
        The resulting polynomial coeffients of the factor
    remainder : ndarray
        The resulting polynomial coefficients of the remainder

    Examples
    --------

    >>> a = np.array([2, 3, 4 ,6])
    >>> b = np.array([1, 3, 6])
    >>> haroldpolydiv(a, b)
    (array([ 2., -3.]), array([ 1., 24.]))
    >>> c = np.array([1, 3, 3, 1])
    >>> d = np.array([1, 2, 1])
    >>> haroldpolydiv(c, d)
    (array([1., 1.]), array([], dtype=float64))

    """
    h_factor, h_remainder = (np.trim_zeros(x, 'f') for x
                             in sig.deconvolve(dividend, divisor))

    return h_factor, h_remainder
