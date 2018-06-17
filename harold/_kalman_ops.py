import numpy as np
from scipy.linalg import block_diag, solve
from ._classes import State, _state_or_abcd
from ._aux_linalg import haroldsvd
from ._arg_utils import _check_for_state

__all__ = ['controllability_matrix', 'observability_matrix',
           'kalman_decomposition', 'is_kalman_controllable',
           'is_kalman_observable']


def controllability_matrix(G, compress=False):
    """
    Computes the Kalman controllability and the transformation matrix. The
    algorithm is the literal computation of the controllability matrix with
    increasing powers of A.

    Numerically, this test is not robust and prone to errors.

    Parameters
    ----------
    G : State, tuple
        System or a tuple of (n,n), (n,m) arraylike
    compress : bool, optional
        If set to True, then the returned controllability matrix is row
        compressed, and in case of uncontrollable modes, has that many
        zero rows.

    Returns
    -------
    Cc : (n,nm) ndarray
        Kalman Controllability Matrix
    T : (n,n) ndarray, tuple
        The transformation matrix such that T^T * Cc is row compressed
        and the number of zero rows at the bottom corresponds to the number
        of uncontrollable modes.
    r: int
        Numerical rank of the controllability matrix

    """
    sys_flag, mats = _state_or_abcd(G, 2)
    if sys_flag:
        A = G.a
        B = G.b
    else:
        A, B = mats

    n = A.shape[0]
    Cc = B.copy()

    for i in range(1, n):
        Cc = np.hstack((Cc, np.linalg.matrix_power(A, i).dot(B)))

    if compress:
        T, S, V, r = haroldsvd(Cc, also_rank=True)
        return S.dot(V.T), T, r

    T, *_, r = haroldsvd(Cc, also_rank=True)
    return Cc, T, r


def observability_matrix(G, compress=False):
    """
    Computes the Kalman controllability and the transformation matrix. The
    algorithm is the literal computation of the observability matrix with
    increasing powers of A.

    Numerically, this test is not robust and prone to errors.

    Parameters
    ----------
    G : State, tuple
        System or a tuple of (n,n), (n,m) arraylike
    compress : bool, optional
        If set to True, then the returned observability matrix is row
        compressed, and in case of unobservability modes, has that many
        zero rows.

    Returns
    -------
    Co : ndarray
        Kalman observability matrix with shape (nm x n)
    T : (n,n) ndarray
        The transformation matrix such that `T.T @ Cc` is row compressed
        and the number of zero rows on the right corresponds to the number
        of unobservable modes.
    r: int
        Numerical rank of the observability matrix

    """
    sys_flag, mats = _state_or_abcd(G, -1)

    if sys_flag:
        A = G.a
        C = G.c
    else:
        A, C = mats

    n = A.shape[0]
    Co = C.copy()

    for i in range(1, n):
        Co = np.vstack((Co, C.dot(np.linalg.matrix_power(A, i))))

    if compress:
        T, S, V, r = haroldsvd(Co, also_rank=True)
        return T.dot(S), V.T, r

    *_, T, r = haroldsvd(Co, also_rank=True)
    return Co, T, r


def kalman_decomposition(G, compute_T=False, output='system',
                         cleanup_threshold=1e-9):
    r"""
    By performing a sequence of similarity transformations the State
    representation is transformed into a special structure such that
    if the system has uncontrollable/unobservable modes, the corresponding
    rows/columns of the B/C matrices have zero blocks and the modes
    are isolated in the A matrix. That is to say, there is no contribution
    of the controllable/observable states on the dynamics of these modes.


    Note that, Kalman operations are numerically not robust. Hence the
    resulting decomposition might miss some 'almost' pole-zero cancellations.
    Hence, this should be used as a rough assesment tool but not as
    actual minimality check or maybe to demonstrate the concepts academic
    purposes to show the modal decomposition. Use cancellation_distance() and
    minimal_realization() functions instead with better numerical properties.

    Parameters
    ----------

    G : State
        The state representation that is to be converted into the block
        triangular form such that unobservable/uncontrollable modes
        corresponds to zero blocks in B/C matrices
    compute_T : boolean
        Selects whether the similarity transformation matrix will be
        returned.
    output : {'system','matrices'}
        Selects whether a State object or individual state matrices will
        be returned.
    cleanup_threshold : float
        After the similarity transformation, the matrix entries smaller than
        this threshold in absolute value would be zeroed. Setting this value
        to zero turns this behavior off.

    Returns
    -------
    Gk : State, tuple
        Returns a state representation or its matrices as a tuple if
        ``output = 'matrices'``
    T  : ndarray
        If ``compute_T`` is ``True``, returns the similarity transform matrix
        that brings the state representation in the resulting decomposed form.

    Examples
    --------
    >>> G = State([[2, 1, 1],
    ...            [5, 3, 6],
    ...            [-5, -1, -4]],
    ...            [[1], [0], [0]],  # B array
    ...            [1, 0, 0])
    >>> is_kalman_controllable(G)
    False
    >>> is_kalman_observable(G)
    False
    >>> F = kalman_decomposition(G)
    >>> print(F.a, F.b, F.c, sep='\n')
    [[ 2.          0.         -1.41421356]
     [ 7.07106781 -3.         -7.        ]
     [ 0.          0.          2.        ]]
    [[-1.]
     [ 0.]
     [ 0.]]
    [[-1.  0.  0.]]
    >>> H = minimal_realization(F)
    >>> H.matrices
    (array([[2.]]), array([[1.]]), array([[1.]]), array([[0.]]))

    """
    _check_for_state(G)

    # If a static gain, then skip and return the argument
    if G._isgain:
        if output == 'matrices':
            return G.matrices

        return G

    # TODO: This is an unreliable test anyways but at least check
    # which rank drop of Cc, Co is higher and start from that
    # to get a tiny improvement

    # First check if controllable
    if not is_kalman_controllable(G):
        _, Tc, r = controllability_matrix(G)
    else:
        Tc = np.eye(G.a.shape[0])
        r = G.a.shape[0]

    ac = solve(Tc, G.a) @ Tc
    bc = solve(Tc, G.b)
    cc = G.c @ Tc
    ac[abs(ac) < cleanup_threshold] = 0.
    bc[abs(bc) < cleanup_threshold] = 0.
    cc[abs(cc) < cleanup_threshold] = 0.

    if r == 0:
        raise ValueError('The system is trivially uncontrollable.'
                         'Probably B matrix is numerically all zeros.')
    elif r != G.a.shape[0]:
        aco, auco = ac[:r, :r], ac[r:, r:]
        # bco = bc[:r, :]
        cco, cuco = cc[:, :r], cc[:, r:]
        do_separate_obsv = True
    else:
        aco, _, cco = ac, bc, cc
        auco, cuco = None, None
        do_separate_obsv = False

    if do_separate_obsv:
        _, To_co, _ = observability_matrix((aco, cco))
        _, To_uco, _ = observability_matrix((auco, cuco))
        To = block_diag(To_co, To_uco)
    else:
        if not is_kalman_observable((ac, cc)):
            _, To, r = observability_matrix((ac, cc))
        else:
            To = np.eye(ac.shape[0])

    A = solve(To, ac) @ To
    B = solve(To, bc)
    C = cc @ To

    # Clean up the mess, if any, for the should-be-zero entries
    A[abs(A) < cleanup_threshold] = 0.
    B[abs(B) < cleanup_threshold] = 0.
    C[abs(C) < cleanup_threshold] = 0.
    D = G.d.copy()

    if output == 'matrices':
        if compute_T:
            return (A, B, C, D), Tc @ To

        return (A, B, C, D)

    if compute_T:
        return State(A, B, C, D, G.SamplingPeriod), Tc.dot(To)

    return State(A, B, C, D, G.SamplingPeriod)


def is_kalman_controllable(G):
    """
    Tests the rank of the Kalman controllability matrix and compares it
    with the A matrix size, returns a boolean depending on the outcome.

    Parameters
    ----------
    G : State, tuple
        The system or the (A,B) matrix tuple

    Returns
    -------
    test_bool : Boolean
        Returns True if the input is Kalman controllable
    """
    sys_flag, mats = _state_or_abcd(G, 2)
    if sys_flag:
        A = G.a
        B = G.b
    else:
        A, B = mats

    _, _, r = controllability_matrix((A, B))

    if A.shape[0] > r:
        return False

    return True


def is_kalman_observable(G):
    """
    Tests the rank of the Kalman observability matrix and compares it
    with the A matrix size, returns a boolean depending on the outcome.

    Parameters
    ----------

    G : State, tuple
        The system or the (A,C) matrix tuple

    Returns
    -------
    test_bool : Boolean
        Returns True if the input is Kalman observable
    """
    sys_flag, mats = _state_or_abcd(G, -1)

    if sys_flag:
        A = G.a
        C = G.c
    else:
        A, C = mats

    _, _, r = observability_matrix((A, C))

    if A.shape[0] > r:
        return False

    return True
