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
from scipy.linalg import block_diag
from ._classes import State, _state_or_abcd
from ._aux_linalg import haroldsvd


def kalman_controllability(G, compress=False):
    """
    Computes the Kalman controllability related quantities. The algorithm
    is the literal computation of the controllability matrix with increasing
    powers of A. Numerically, this test is not robust and prone to errors if
    the A matrix is not well-conditioned or its entries have varying order
    of magnitude as at each additional power of A the entries blow up or
    converge to zero rapidly.

    Parameters
    ----------
    G : State() or tuple of {(n,n),(n,m)} array_like matrices
        System or matrices to be tested

    compress : Boolean
        If set to True, then the returned controllability matrix is row
        compressed, and in case of uncontrollable modes, has that many
        zero rows.

    Returns
    -------

    Cc : {(n,nxm)} 2D numpy array
        Kalman Controllability Matrix
    T : (n,n) 2D numpy arrays
        The transformation matrix such that T^T * Cc is row compressed
        and the number of zero rows at the bottom corresponds to the number
        of uncontrollable modes.
    r: integer
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


def kalman_observability(G, compress=False):
    """
    Computes the Kalman observability related objects. The algorithm
    is the literal computation of the observability matrix with increasing
    powers of A. Numerically, this test is not robust and prone to errors if
    the A matrix is not well-conditioned or too big as at each additional
    power of A the entries blow up or converge to zero rapidly.

    Parameters
    ----------
    G : State() or {(n,n),(n,m)} array_like matrices
        System or matrices to be tested
    compress : Boolean
        If set to True, then the returned observability matrix is row
        compressed, and in case of unobservability modes, has that many
        zero rows.

    Returns
    -------
    Co : {(n,nxm)} 2D numpy array
        Kalman observability matrix
    T : (n,n) 2D numpy arrays
        The transformation matrix such that T^T * Cc is row compressed
        and the number of zero rows on the right corresponds to the number
        of unobservable modes.
    r: integer
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
    """
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

    Example usage and verification : ::

        G = State([[2,1,1],[5,3,6],[-5,-1,-4]],[[1],[0],[0]],[[1,0,0]],0)
        print('Is it Kalman Cont\'ble ? ',is_kalman_controllable(G))
        print('Is it Kalman Obsv\'ble ? ',is_kalman_observable(G))
        F = kalman_decomposition(G)
        print(F.a,F.b,F.c)
        H = minimal_realization(F.a,F.b,F.c)
        print('The minimal system matrices are:',*H)


    Expected output : ::

        Is it Kalman Cont'ble ?  False
        Is it Kalman Obsv'ble ?  False
        [[ 2.          0.         -1.41421356]
         [ 7.07106781 -3.         -7.        ]
         [ 0.          0.          2.        ]]

        [[-1.]
         [ 0.]
         [ 0.]]

        [[-1.  0.  0.]]

        The minimal system matrices are:
         [[ 2.]] [[ 1.]] [[ 1.]]

    .. warning:: Kalman decomposition is often described in an ambigous fashion
                 in the literature. I would like to thank Joaquin Carrasco for
                 his generous help on this matter for his lucid argument as to
                 why this is probably happening. This is going to be
                 reimplemented with better tests on pathological models.

    Parameters
    ----------

    G : State()
        The state representation that is to be converted into the block
        triangular form such that unobservable/uncontrollable modes
        corresponds to zero blocks in B/C matrices

    compute_T : boolean
        Selects whether the similarity transformation matrix will be
        returned.

    output : {'system','matrices'}
        Selects whether a State() object or individual state matrices
        will be returned.

    cleanup_threshold : float
        After the similarity transformation, the matrix entries smaller
        than this threshold in absolute value would be zeroed. Setting
        this value to zero turns this behavior off.

    Returns
    -------
    Gk : State() or if output = 'matrices' is selected (A,B,C,D) tuple
        Returns a state representation or its matrices as a tuple

    T  : (nxn) 2D-numpy array
        If compute_T is True, returns the similarity transform matrix
        that brings the state representation in the resulting decomposed
        form such that

            Gk.a = inv(T)*G.a*T
            Gk.b = inv(T)*G.b
            Gk.c = G.c*T
            Gk.d = G.d

    """
    if not isinstance(G, State):
        raise TypeError('The argument must be a State() object')

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
        Tc, r = kalman_controllability(G)[1:]
    else:
        Tc = np.eye(G.a.shape[0])
        r = G.a.shape[0]

    ac = np.linalg.solve(Tc, G.a).dot(Tc)
    bc = np.linalg.solve(Tc, G.b)
    cc = G.c.dot(Tc)
    ac[abs(ac) < cleanup_threshold] = 0.
    bc[abs(bc) < cleanup_threshold] = 0.
    cc[abs(cc) < cleanup_threshold] = 0.

    if r == 0:
        raise ValueError('The system is trivially uncontrollable.'
                         'Probably B matrix is numerically all zeros.')
    elif r != G.a.shape[0]:
        aco, auco = ac[:r, :r], ac[r:, r:]
        bco = bc[:r, :]
        cco, cuco = cc[:, :r], cc[:, r:]
        do_separate_obsv = True
    else:
        aco, bco, cco = ac, bc, cc
        auco, cuco = None, None
        do_separate_obsv = False

    if do_separate_obsv:
        To_co = kalman_observability((aco, cco))[1]
        To_uco = kalman_observability((auco, cuco))[1]
        To = block_diag(To_co, To_uco)
    else:
        if not is_kalman_observable((ac, cc)):
            To, r = kalman_observability((ac, cc))[1:]
        else:
            To = np.eye(ac.shape[0])

    A = np.linalg.solve(To, ac).dot(To)
    B = np.linalg.solve(To, bc)
    C = cc.dot(To)

    # Clean up the mess, if any, for the should-be-zero entries
    A[abs(A) < cleanup_threshold] = 0.
    B[abs(B) < cleanup_threshold] = 0.
    C[abs(C) < cleanup_threshold] = 0.
    D = G.d.copy()

    if output == 'matrices':
        if compute_T:
            return (A, B, C, D), Tc.dot(To)

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

    G : State() or tuple of {(nxn),(nxm)} array_like matrices
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

    r = kalman_controllability((A, B))[-1]

    if A.shape[0] > r:
        return False

    return True


def is_kalman_observable(G):
    """
    Tests the rank of the Kalman observability matrix and compares it
    with the A matrix size, returns a boolean depending on the outcome.

    Parameters
    ----------

    G : State() or tuple of {(nxn),(pxn)} array_like matrices
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

    r = kalman_observability((A, C))[-1]

    if A.shape[0] > r:
        return False

    return True
