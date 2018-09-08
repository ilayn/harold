import numpy as np
from numpy import eye, zeros, block, array, poly
from numpy.linalg import matrix_power
from scipy.linalg import (solve, eigvals, block_diag,
                          solve_continuous_are as care,
                          solve_discrete_are as dare)

from ._arg_utils import _check_for_state_or_transfer
from ._classes import Transfer, transfer_to_state, _state_or_abcd
from ._kalman_ops import controllability_matrix
from ._aux_linalg import matrix_slice

__all__ = ['lqr', 'ackermann', 'pole_placement']


def lqr(G, Q, R=None, S=None, weight_on='state'):
    """
    A full-state static feedback control design solver for which the following
    quadratic cost function is integrated (summed) over all positive time
    axis ::

                   [x]' [ Q | S ] [x]
        J(x, u) =  [-]  [---|---] [-]
                   [u]  [ S | R ] [u]

    for a continuous-time (discrete-time) system. If the system is given as a
    transfer-function then first a conversion to a state model is performed.
    If the "weight_on" is set to "output" then the cost function is assumed
    to be in the following form ::

                   [x]' [ C | D ]' [ Q | S ] [ C | D ] [x]
        J(x, u) =  [-]  [---|---]  [---|---] [---|---] [-] := J(y, u)
                   [u]  [ 0 | I ]  [ S | R ] [ 0 | I ] [u]

    Based on the provided system, the solution type is automatically selected
    between continuous-time and discrete-time solutions.


    Parameters
    ----------
    G : State, Transfer
        The regulated dynamic system representation
    Q : array_like
        Square, state or output weighting matrix
    R : array_like
        Square input weighting matrix
    S: array_like
        Crossweighting matrix
    weight_on : str, optional
        Depending on its value of "state" or "output", the weights are applied
        on either states or the output.

    Returns
    -------
    K : ndarray
        The regulator K such that A-BK is stable
    X : ndarray
        The stabilizing solution to the corresponding Riccati equation
    eigs : ndarray
        The array of closed loop A-BK eigenvalues.

    Notes
    -----
    For the conditions that weight matrices should satisfy, see SciPy
    documentation over :func:`scipy.linalg.solve_continuous_are` and
    :func:`scipy.linalg.solve_discrete_are`

    Moreover, for the output weighted case, the returned solution is not
    always guaranteed to be the stabilizing solution.

    If a Transfer is given, a the mismatch between the ``Q`` shape and
    the resulting number of states after the conversion can happen. It is
    recommended to work with State realizations directly.

    """
    _check_for_state_or_transfer(G)

    if G._isgain:
        raise ValueError('State feedback design requires dynamic system'
                         ' representations. The argument is a static gain.')

    if weight_on.lower() not in ('state, output'):
        raise ValueError('"weight_on" keyword can either be "state" or'
                         ' "output" but received "{}"'.format(weight_on))

    if isinstance(G, Transfer):
        T = transfer_to_state(G)
    else:
        T = G

    n, m, p = T.NumberOfStates, T.NumberOfInputs, T.NumberOfOutputs

    if R is None:
        R = eye(m)

    func = care if T.SamplingSet == 'R' else dare
    # Arrays, shape mismatches and sign definiteness are handled by
    # scipy riccati solver. So pass directly if not "output".

    if weight_on == 'output':
        Q = array(Q, dtype=float, ndmin=2)
        R = array(R, dtype=float, ndmin=2)
        for ind, (arg, rc, st) in enumerate(
                                zip([Q, R], [p, m], ['input', 'output'])):
            if arg.shape[0] != rc or arg.shape[0] != arg.shape[1]:
                raise ValueError('{0} array should be both square and have '
                                 ' {1} rows/cols as the number of {2}s for '
                                 '{2} weighting. The argument has the shape '
                                 '{3}.'.format('QR'[ind], rc, st, arg.shape))

        mat = block([[T.c, T.d], [zeros((m, n)), eye(m)]])
        if S is None:
            fact = block_diag(Q, R)
        else:
            S = array(S, dtype=float, ndmin=2)
            if S.shape != (p, m):
                raise ValueError('S array is expected to have the shape {}'
                                 ' but has {}'.format((p, m), S.shape))
            fact = block([[Q, S], [S.T, R]])

        fact = mat.T @ fact @ mat
        Q, S, _, R = matrix_slice(fact, (m, m), corner='se')
        X = func(a=T.a, b=T.b, q=Q, r=R, s=S)
    else:
        X = func(a=T.a, b=T.b, q=Q, r=R, s=S)

    if T.SamplingSet == 'R':
        K = solve(R, T.b.T @ X) if S is None else solve(R, T.b.T @ X + S.T)
    else:
        K = solve(T.b.T @ X @ T.b + R, T.b.T @ X @ T.a) if S is None else \
                            solve(T.b.T @ X @ T.b + R, T.b.T @ X @ T.a + S.T)

    return K, X, eigvals(T.a - T.b @ K)


def ackermann(G, loc):
    """
    Pole placement using Ackermann's polynomial method.

    Parameters
    ----------
    G : State, tuple
        The model or state and input arrays of the model as a tuple
    loc: arraylike
        Desired eigenvalue locations given as a 1D arraylike

    Returns
    -------
    K: ndarray
        Resulting static full state feedback gain such that the array ``A-B@K``
        has the eigenvalues prescribed by ``loc``

    Notes
    -----
    This is a naive implementation method of pole placement. Numerically it is
    quite fragile and instable. Hence it might only give meaningful results for
    small and well-controllable systems.

    """
    # Check input arguments.
    is_sys, arrs = _state_or_abcd(G, n=2)
    A, B = (G.a, G.b) if is_sys else arrs
    n, m = A.shape[0], B.shape[1]

    if m != 1:
        raise ValueError('Ackermann method is only applicable to single input'
                         ' systems.')

    loc = array(loc, dtype=float)
    if loc.ndim > 1:
        raise ValueError('Pole location array must be 1D.')

    Cc, _, r = controllability_matrix((A, B), compress=False)
    # if not controllable
    if r < Cc.shape[0]:
        raise ValueError('The system is numerically uncontrollable. Pole '
                         'placement is not possible.')
    # Desired characteristic polynomial
    p = poly(loc).real[::-1]
    # Get the state-matrix-to-be via matrix evaluation of the polynomial
    # TODO: Maybe implement Horner's scheme for evaluation?
    pmat = np.zeros((n, n), dtype=float)
    for pow_a in range(n+1):
        pmat += p[pow_a] * matrix_power(A, pow_a)

    return solve(Cc, pmat)[[-1], :]


def pole_placement(sys_or_ab, target_poles):
    pass
