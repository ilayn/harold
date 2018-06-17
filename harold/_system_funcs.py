from copy import deepcopy
import numpy as np
from numpy.linalg import cond, eig, norm
from numpy import array, flipud, fliplr, eye, zeros
from numpy.random import rand
from scipy.linalg import (solve, svdvals, qr, block_diag,
                          hessenberg, matrix_balance)
from ._aux_linalg import haroldsvd, matrix_slice, e_i
from ._classes import State, Transfer, transfer_to_state
from ._arg_utils import _check_for_state_or_transfer


__all__ = ['staircase', 'cancellation_distance', 'minimal_realization',
           'hessenberg_realization']


def staircase(A, B, C, compute_T=False, form='c', invert=False,
              block_indices=False):
    """
    Given a state model data A, B, C, Returns the so-called staircase form
    State realization to assess the controllability/observability properties.

    If observer-form is requested, internally the system is pertransposed
    and same controllable-form machinery is applied.

    Parameters
    ----------
    A : (n, n) array_like
        State array
    B : (n, m) array_like
        Input array
    C : (p, n) array_like
        Output array
    form : str, optional
        Determines whether the controller- or observer-staircase form
        will be computed via "c" or "o" values.
    invert : bool, optional
        If True, the full rank part of B/C matrix will be compressed to
        lower/right part of the array. Naturally, this also effects the A
        matrix blocks to appear as lower/upper triangular block matrix.

    Returns
    -------
    Ah : (n, n) ndarray
        Resulting State array
    Bh : (n, m) ndarray
        Resulting Input array
    Ch : (p, n) ndarray
        Resulting Output array
    T : (n, n) ndarray
        The transformation matrix such that ::

            [ T⁻¹AT | T⁻¹B ]   [ Ah | Bh ]
            [-------|------] = [----|----]
            [   CT  |      ]   [ Ch |    ]

    Notes
    -----
    For controllability and observability, the existence of zero subdiagonal
    blocks can be checked in a numerically stable fashion, as opposed to
    forming the  Kalman matrices and checking the rank. For certain matrices,
    A^n computations can introduce large errors (for some A that have entries
    with varying order of magnitudes). But it is also prone to numerical
    rank identification.

    """

    if form not in {'c', 'o'}:
        raise ValueError('The "form" key can only take values "c" or "o".')

    _ = State.validate_arguments(A, B, C, zeros((C.shape[0], B.shape[1])))
    if form == 'o':
        A, B, C = A.T, C.T, B.T

    # trivially uncontrollable, quick exit
    if not np.any(B):
        if form == 'o':
            A, B, C = A.T, C.T, B.T
        return A, B, C, eye(A.shape[0])

    n = A.shape[0]
    ub, sb, vb, m = haroldsvd(B, also_rank=True)
    sb[m:, m:] = 0.

    # After these, start the regular case
    A0 = ub.T @ A @ ub
    # Row compress B
    B0 = sb @ vb
    C0 = C @ ub
    T = block_diag(eye(n-ub.shape[1]), ub.T)

    if n == m:
        # Already triangulized B, nothing else to do
        if invert:
            A0, B0, C0 = fliplr(flipud(A0)), flipud(B0), fliplr(C0)
            T = flipud(T)
        return (A0, B0, C0, T.T) if form == 'c' else (A0.T, B0.T, C0.T, T.T)

    next_, size_ = -m, m
    tol_from_A = n*norm(A, 1)*np.spacing(1.)

    for _ in range(A0.shape[0]):
        next_, size_ = next_ + size_, m
        h1, h2, h3, h4 = matrix_slice(A0[next_:, next_:], (size_, size_))

        # If turns out to be zero or empty, We are done quick exit
        if not np.any(h3):
            break

        uh3, sh3, vh3, m = haroldsvd(h3, also_rank=True, rank_tol=tol_from_A)
        sh3[m:, m:] = 0.
        # If the resulting subblock is not full row
        if 0 < m < h3.shape[0]:
            T[-uh3.shape[0]:, :] = uh3.T @ T[-uh3.shape[0]:, :]
            A0[:, next_ + size_:] = A0[:, next_ + size_:] @ uh3
            A0[next_ + size_:, next_:next_ + size_] = sh3 @ vh3
            A0[next_ + size_:, next_ + size_:] = uh3.T @ h4
            p = uh3.shape[0]
            C0[:, -p:] = C0[:, -p:] @ uh3
        else:
            break
    if invert:
        A0, B0, C0, T = fliplr(flipud(A0)), flipud(B0), fliplr(C0), flipud(T)
    return (A0, B0, C0, T.T) if form == 'c' else (A0.T, C0.T, B0.T, T.T)


def hessenberg_realization(G, compute_T=False, form='c', invert=False,
                           output='system'):
    """
    A state transformation is applied in order to get the following form where
    A is a Hessenberg matrix and B (or C if 'form' is set to 'o' ) is row/col
    compressed ::

                                [x x x x x|x x]
                                [x x x x x|0 x]
                                [0 x x x x|0 0]
                                [0 0 x x x|0 0]
                                [0 0 0 x x|0 0]
                                [---------|---]
                                [x x x x x|x x]
                                [x x x x x|x x]

    Parameters
    ----------
    G : State, Transfer, 3-tuple
        A system representation or the A, B, C matrix triplet of a  State
        realization. Static Gain models are returned unchanged.
    compute_T : bool, optional
        If set to True, the array that would bring the system into the desired
        form will also be returned. Default is False.
    form : str, optional
        The switch for selecting between observability and controllability
        Hessenberg forms. Valid entries are ``c`` and ``o``.
    invert : bool, optional
        Select which side the B or C matrix will be compressed. For example,
        the default case returns the B matrix with (if any) zero rows at the
        bottom. invert option flips this choice either in B or C matrices
        depending on the "form" switch.
    output : str, optional
        In case only the resulting matrices and not a system representation is
        needed, this keyword can be used with the value ``'matrices'``. The
        default is ``'system'``. If 3-tuple is given and 'output' is still
        'system' then the feedthrough matrix is taken to be 0.

    Returns
    -------
    Gh : State, tuple
        A realization or the matrices are returned depending on the 'output'
        keyword.
    T : ndarray
        If 'compute_T' is set to True, the array for the state transformation
        is returned.

    """
    if isinstance(G, tuple):
        a, b, c = G
        a, b, c, *_ = State.validate_arguments(a, b, c, zeros((c.shape[0],
                                                               b.shape[1])))
        in_is_sys = False
    else:
        _check_for_state_or_transfer(G)
        in_is_sys = True
        if isinstance(G, Transfer):
            a, b, c, _ = transfer_to_state(G, output='matrices')
        else:
            a, b, c = G.a, G.b, G.c

    if form == 'o':
        a, b, c = a.T, c.T, b.T

    qq, bh = qr(b)
    ab, cb = qq.T @ a @ qq, c @ qq
    ah, qh = hessenberg(ab, calc_q=True)
    ch = cb @ qh

    if compute_T:
        T = qq @ qh

    if invert:
        ah, bh, ch = fliplr(flipud(ah)), flipud(bh), fliplr(ch)
        if compute_T:
                T = flipud(T)

    if form == 'o':
        ah, bh, ch = ah.T, ch.T, bh.T

    if output == 'system':
        if in_is_sys:
            Gh = State(ah, bh, ch, dt=G.SamplingPeriod)
        else:
            Gh = State(ah, bh, ch)
        if compute_T:
            return Gh, T
        else:
            return Gh
    else:
        if compute_T:
            return ah, bh, ch, T
        else:
            return ah, bh, ch


def cancellation_distance(F, G):
    """
    Computes the upper and lower bounds of the perturbation needed to render
    the pencil :math:`[F-pI | G]` rank deficient. It is used for assessing
    the controllability/observability degeneracy distance and hence for
    minimality assessment.

    Parameters
    ----------
    A : array_like
        Square input array (n x n)
    B : array_like
        Input array (n x m)

    Returns
    -------
    upper2 : float
        Upper bound on the norm of the perturbation ``[dF | dG]`` such
        that ``[F+dF-pI | G+dG]` is rank deficient for some ``p``.
    upper1 : float
        A theoretically softer upper bound than ``upper2`` for the same
        norm.
    lower0 : float
        Lower bound on the norm given in ``upper2``
    e_f : complex
        Indicates the eigenvalue that renders ``[F+dF-pI | G+dG ]`` rank
        deficient
    radius : float
        The perturbation with the norm bound ``upper2`` is located within
        a disk in the complex plane whose center is on ``e_f`` and whose
        radius is bounded by this output.

    Notes
    -----
    Implements the upper bounds given in [1]_

    References
    ----------

    .. [1] D. Boley, Estimating the Sensitivity of the Algebraic Structure
        of Pencils with Simple Eigenvalue Estimates, :doi:`10.1137/0611046`

    """
    if not np.equal(*F.shape):
        raise ValueError('F input must be a square array.')
    if F.shape[0] != G.shape[0]:
        raise ValueError('F and G inputs must have the same number of rows.')

    A = np.c_[F, G].T
    n, m = A.shape
    B = e_i(n, np.s_[:m])
    D = e_i(n, np.s_[m:])
    C, _ = qr(2*rand(n, n-m) - 1, mode='economic')
    evals, V = eig(np.c_[A, C])
    K = cond(V)
    X = V[:m, :]
    Y = V[m:, :]

    upp0 = [0]*n
    for x in range(n):
        upp0[x] = norm((C-evals[x]*D).dot(Y[:, x])) / norm(X[:, x])

    f = np.argsort(upp0)[0]
    e_f = evals[f]
    upper1 = upp0[f]
    upper2 = svdvals(A - e_f*B)[-1]
    lower0 = upper2/(K+1)
    radius = upper2*K

    return upper2, upper1, lower0, e_f, radius


def minimal_realization(G, tol=1e-6):
    """
    Given system realization G, this computes minimal realization such that
    if a State representation is given then the returned representation is
    controllable and observable within the given tolerance ``tol``. If
    a Transfer representation is given, then the fractions are simplified
    in the representation entries.

    Parameters
    ----------
    G : State, Transfer
        System representation to be checked for minimality
    tol: float
        The sensitivity threshold for the cancellation.

    Returns
    -------
    G_min : State, Transfer
        Minimal realization of the input `G`

    Notes
    -----
    For State() inputs the alogrithm uses ``cancellation_distance()`` and
    ``staircase()`` for the tests. A basic two pass algorithm performs:

        1- First distance to mode cancellation is computed then also
        the Hessenberg form is obtained with the identified o'ble/c'ble
        block numbers.

        2- If staircase form reports that there are no cancellations but the
        distance is less than the tolerance, distance wins and the
        corresponding mode is removed.

    For Transfer() inputs, every entry of the representation is checked for
    pole/zero cancellations and ``tol`` is used to decide for the decision
    precision.
    """
    _check_for_state_or_transfer(G)

    if isinstance(G, State):
        if G._isgain:
            return State(G.to_array())
        else:
            A, B, C, D = G.matrices
            Am, Bm, Cm = _minimal_realization_state(A, B, C, tol=tol)
            if Am.size > 0:
                return State(Am, Bm, Cm, D, dt=G.SamplingPeriod)
            else:
                return State(D, dt=G.SamplingPeriod)
    else:
        if G._isgain:
            return Transfer(G.to_array())
        else:
            num, den = G.polynomials
            numm, denm = _minimal_realization_transfer(num, den, tol=tol)
            return Transfer(numm, denm, dt=G.SamplingPeriod)


def _minimal_realization_state(A, B, C, tol=1e-6):
    """
    Low-level function to perform the state removel if any for minimal
    realizations. No consistency check is performed.
    """

    # Empty matrices, don't bother
    if A.size == 0:
        return A, B, C

    # scale the system matrix with possible permutations
    A, T = matrix_balance(A)
    # T always has powers of 2 nonzero elements
    B, C = solve(T, B), C @ T

    n = A.shape[0]
    # Make sure that we still have states left, otherwise done
    if n == 0:
        return A, B, C

    # Now obtain the c'ble and o'ble staircase forms
    Ac, Bc, Cc, _ = staircase(A, B, C)
    Ao, Bo, Co, _ = staircase(A, B, C, form='o', invert=True)
    # And compute the distance to rank deficiency.
    kc, *_ = cancellation_distance(Ac, Bc)
    ko, *_ = cancellation_distance(Ao.T, Co.T)

    # If both distances are above tol then we have already minimality
    if min(kc, ko) > tol:
        return A, B, C
    else:
        # Here, we have improved the cancellation distance computations by
        # first scaling the system and then forming the staircase forms.

        # If unctrblity distance is smaller, let it first (no reason)
        if kc <= tol:
            # Start removing and check if the distance gets bigger
            # Observability form removes from top left
            # controllability form removes from bottom right
            while kc <= tol:
                Ac, Bc, Cc = (Ac[:-1, :-1], Bc[:-1, :], Cc[:, :-1])
                if Ac.size == 0:
                    A, B, C = [array([], dtype=float)]*3
                    break
                else:
                    kc, *_ = cancellation_distance(Ac, Bc)
            # Return the resulting matrices
            A, B, C = Ac, Bc, Cc
            # Same with the o'ble modes, but now kc might have removed
            # unobservable mode already so get the distance again
            ko, *_ = cancellation_distance(A.T, C.T)

        # Still unobservables ?
        if ko <= tol:
            Ao, Bo, Co, To = staircase(A, B, C, form='o', invert=True)
            while ko <= tol:  # Until cancel dist gets big
                Ao, Bo, Co = Ao[1:, 1:], Bo[1:, :], Co[:, 1:]
                if Ao.size == 0:
                    A, B, C = [array([], dtype=float)]*3
                else:
                    ko, *_ = cancellation_distance(Ao, Bo)

            # Return the resulting matrices
            A, B, C = Ao, Bo, Co

    return A, B, C


def _minimal_realization_transfer(num, den, tol=1e-6):
    '''
    A helper function for obtaining a minimal representation of the
    Transfer() models.
    The method is pretty straightforward; going over the pole/zero pairs
    and removing them if they are either exactly the same or within their
    neigbourhood in 2-norm sense with threshold `tol`.
    '''
    # MIMO or not?
    if isinstance(num, list):
        # Don't touch the original data
        num = deepcopy(num)
        den = deepcopy(den)

        # Walk over entries for pole/zero cancellations
        m, p = len(num[0]), len(num)
        for row in range(p):
            for col in range(m):
                (num[row][col],
                 den[row][col]) = _minimal_realization_simplify(num[row][col],
                                                                den[row][col],
                                                                tol)
    # It's SISO search directly
    else:
        num, den = _minimal_realization_simplify(num, den, tol)

    return num, den


def _minimal_realization_simplify(num, den, tol):
    '''
    This is a simple distance checker between the each root of num and all
    roots of den to see whether there are any pairs that are sufficiently
    close to each other defined by `tol`.
    '''
    # Early exit if numerator is a scalar
    if num.size == 1:
        m = den[0, 0]
        return num/m, den/m

    # Get the gain from leading coefficients to work with monic polynomials
    k_gain = num[0, 0]/den[0, 0]
    plz = np.roots(den[0])
    zrz = np.roots(num[0])

    # Root finding algorithms are inherently ill-conditioned. Hence it might
    # happen that real multiplicities can turn out as complex pairs, e.g.,
    # np.roots(np.poly([1,2,3,2,3,4]))

    # This is a simple walk over zeros checking if there is something close
    # enough to it in the pole list with the slight extra check:
    # If we encounter 3.0 zero vs. 3.0+1e-9i pole, we look for another real
    # 3.0 in the zeros and for the conjugate of the pole and vice versa.

    # Zeros (both reals and one element of each complex pairs)
    zrz = np.r_[zrz[np.imag(zrz) == 0.], zrz[np.imag(zrz) > 0.]]

    safe_z = []

    for z in zrz:
        dist = np.abs(plz-z)
        bool_cz = np.imag(z) > 0
        # Do we have a match ?
        if np.min(dist) < tol + tol*np.abs(z):
            # Get the index and check the complex part
            match_index = np.argmin(dist)
            pz = plz[match_index]
            bool_cp = np.imag(pz) > 0

            if bool_cz and bool_cp:
                plz = np.delete(plz, match_index)
                # remove also the conjugate
                del_index, = np.where(plz == np.conj(pz))
                plz = np.delete(plz, del_index[0])

            elif bool_cz and not bool_cp:
                # We have a complex pair of zeros and a real pole
                # If there is another entry of this pole then we
                # cancel both of them otherwise we assume a real/real
                # cancellation and convert the other complex zero to real.

                # First get rid of the real pole
                plz = np.delete(plz, match_index)
                # Now search another real pole that is also close
                dist = np.abs(plz-z)
                if np.min(dist) < tol + tol*np.abs(z):
                    match_index = np.argmin(dist)
                    plz = np.delete(plz, match_index)
                else:
                    # It was a real/complex cancellation, make the zero real
                    safe_z += np.real(z)

            elif not bool_cz and bool_cp:
                # Same with above but this time we convert the other pole
                # to catch in the next iteration if another zero exists
                plz = np.delete(plz, match_index)
                conj_index, = np.where(plz == np.conj(pz))
                plz[conj_index[0]] = np.real(plz[conj_index[0]])

            else:
                plz = np.delete(plz, match_index)

        else:
            safe_z += [z]

            if bool_cz:
                safe_z += [np.conj(z)]

    return np.atleast_2d(k_gain*np.poly(safe_z)), np.atleast_2d(np.poly(plz))
