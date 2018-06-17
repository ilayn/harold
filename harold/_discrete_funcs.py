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
from numpy import block, zeros, eye
from numpy.linalg import LinAlgError
from scipy.linalg import (expm, logm, block_diag, matrix_balance, eigvals,
                          norm, solve)
from warnings import warn, simplefilter, catch_warnings
from ._classes import Transfer, State, transfer_to_state, state_to_transfer
from ._global_constants import _KnownDiscretizationMethods as KDM
from ._arg_utils import _check_for_state_or_transfer
from ._arg_utils import _rcond_warn

__all__ = ['discretize', 'undiscretize']


def discretize(G, dt, method='tustin', prewarp_at=0., q=None):
    """
    Continuous- to discrete-time model conversion.

    The default discretization method is 'tustin'.

    Parameters
    ----------
    G : State, Transfer
        The to-be-discretized system representation
    dt : float
        The positive scalar for the sampling period in seconds
    method : str
        The method to be used for discretization. The valid method inputs
        can be listed via harold._global_constants._KnownDiscretizationMethods
        variable.
    prewarp_at : float
        If the discretization method is 'tustin' or one of its aliases, this
        positive scalar modifies the response such that the frequency warp
        happens elsewhere to match the dynamics at this frequency.
    q : 2D ndarray
        If given, represents the custom discretization matrix such that the
        following star-product is computed::

                       ┌───────┐      ─┐
                       │  1    │       │
                    ┌──┤ ─── I │<─┐    │
                    │  │  z    │  │    │
                    │  └───────┘  │    │    1
                    │             │    ├─  ─── I
                    │   ┌─────┐   │    │    s
                    └──>│     ├───┘    │
                        │  Q  │        │
                    ┌──>│     ├───┐    │
                    │   └─────┘   │   ─┘
                    │             │
                    │   ┌─────┐   │
                    └───┤     │<──┘
                        │  G  │
                    <───┤     │<──
                        └─────┘

        where Q is the kronecker product, I_n ⊗ q and n being the number of
        states.

    Returns
    -------
    Gd : State, Transfer
        The resulting discrete model representation

    Notes
    -----
    Apart from the zero-order hold, first-order hold methods, the remaining
    methods are special cases of a particular Q and computed as as such.

    """

    _check_for_state_or_transfer(G)

    if method not in KDM:
        raise ValueError('I don\'t know the discretization method "{0}". But '
                         'I know:\n {1}'.format(method, ',\n'.join(KDM)))

    if G.SamplingSet == 'Z':
        raise ValueError('The argument is already modeled as a '
                         'discrete-time system.')

    if isinstance(G, Transfer):
        T = transfer_to_state(G)
    else:
        T = G

    if G._isgain:
        # Method doesn't matter
        Gd = Transfer(G.to_array(), dt=dt) if isinstance(G, Transfer) else \
                                                    State(G.to_array(), dt=dt)
        Gd.SamplingPeriod = dt
    else:
        discretized_args = _discretize(T, dt, method, prewarp_at, q)

        if isinstance(G, State):
            Gd = State(*discretized_args)
            Gd.DiscretizedWith = method
        else:
            Gss = State(*discretized_args)
            Gd = state_to_transfer(Gss)
            Gd.DiscretizedWith = method

        if method == 'lft':
            Gd.DiscretizationMatrix = q
        elif method in ('tustin', 'bilinear', 'trapezoidal'):
            Gd.PrewarpFrequency = prewarp_at

        Gd.SamplingPeriod = dt

    return Gd


def _discretize(T, dt, method, prewarp_at, q):

    m, n = T.shape[1], T.NumberOfStates

    if method == 'zoh':
        """

                 [A | B]   [ exp(A) | ∫exp(A)dt * B ]   [ Ad | Bd ]
            expm [- - -] = [------------------------] = [---------]
                 [0 | 0]   [   0    |       I       ]   [ C  | D  ]

        """

        M = block([[T.a, T.b], [zeros((m, m+n))]])
        # Don't permute, destroys the zero structure
        Ms, (sca, _) = matrix_balance(M, permute=0, separate=1)
        # inverted scale after exponentiation via broadcast+elwise mult.
        eM = expm(Ms*dt) * (sca[:, None] * np.reciprocal(sca))
        Ad, Bd, Cd, Dd = eM[:n, :n], eM[:n, n:], T.c, T.d

    elif method == 'foh':
        """
        This conversion is done via the expm() identity

                  [ A*t   B*t   0 ]   [ Φ  Γ₁  Γ₂]
             expm [  0     0    I ] = [ 0  I   I ]
                  [  0     0    0 ]   [ 0  0   I ]

        where Φ, Γ₁, Γ₂ satisfies

            Φ(B*t) = (A*t)Γ₁ + B*t = (A*t)²Γ₂ + (A*t + I)B*t.

        See Franklin, Powell "Digital Control of Dynamic Systems" 3rd ed.
        section 6.3.2
        """
        M = block([[block_diag(block([T.a, T.b])*dt, eye(m))],
                   [zeros((m, n+2*m))]])
        # Don't permute, destroys the zeros/identity structure
        Ms, (sca, _) = matrix_balance(M, permute=0, separate=1)
        # inverted scale after exponentiation via broadcast+elwise mult.
        eM = expm(Ms) * (sca[:, None] * np.reciprocal(sca))

        Ad = eM[:n, :n]
        Bd0, Bd1 = eM[:n, n:n+m], eM[:n, n+m:]
        Bd = Bd0 + Ad @ Bd1 - Bd1
        Cd = T.c
        Dd = T.d + T.c @ Bd1

    elif method in ('bilinear', 'tustin', 'trapezoidal'):
        if prewarp_at == 0.:
            q = np.array([[1, np.sqrt(dt)], [np.sqrt(dt), dt/2]])
        else:
            if 1/(2*dt) < prewarp_at:
                raise ValueError('Prewarping frequency is beyond the Nyquist'
                                 ' rate. It has to satisfy 0 < w < 1/(2*Δt)'
                                 ' and Δt being the sampling period in '
                                 'seconds. Δt={0} is given, hence the maximum'
                                 ' allowed is {1} Hz.'.format(dt, 1/(2*dt))
                                 )
            prew_rps = 2 * np.pi * prewarp_at
            sq2tan = np.sqrt(2*np.tan(prew_rps * dt / 2)/prew_rps)
            q = np.array([[1, sq2tan], [sq2tan, sq2tan**2/2]])

        Ad, Bd, Cd, Dd = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    elif method in ('forward euler', 'forward difference',
                    'forward rectangular', '>>'):
        q = np.array([[1, np.sqrt(dt)], [np.sqrt(dt), 0]])
        Ad, Bd, Cd, Dd = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    elif method in ('backward euler', 'backward difference',
                    'backward rectangular', '<<'):
        q = np.array([[1, np.sqrt(dt)], [np.sqrt(dt), dt]])
        Ad, Bd, Cd, Dd = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    elif method == 'lft':
        if q is None:
            raise ValueError('"lft" method requires a 2x2 interconnection '
                             'matrix "q" between s and z indeterminates.')
        Ad, Bd, Cd, Dd = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    return Ad, Bd, Cd, Dd, dt


def undiscretize(G, method=None, prewarp_at=0., q=None):
    """
    Discrete- to continuous-time model conversion.

    If the model has the Discretization Method set and no method is given,
    then uses that discretization method to reach back to the continous
    system model.

    Parameters
    ----------
    G : State, Transfer
        Discrete-time system to be undiscretized
    method : str
        The method to use for converting discrete model to continuous.
    prewarp_at : float
        If method is "tustin" or its aliases then this is the prewarping
        frequency that discretization was corrected for.
    q : (2, 2) array_like
        The LFT interconnection matrix if the method is "lft".

    Returns
    -------
    Gc : State, Transfer
        Undiscretized continuous-time system
    """
    _check_for_state_or_transfer(G)

    if G.SamplingSet == 'R':
        raise ValueError('The argument is already modeled as a '
                         'continuous-time system.')

    dt = G.SamplingPeriod

    if method is None:
        if G.DiscretizedWith is None:
            method = 'tustin'
        else:
            method = G.DiscretizedWith

    if method == 'lft':
        if G.DiscretizationMatrix is None and q is None:
            raise ValueError('"lft" method requires also the '
                             'DiscretizationMatrix property set.')
        # At least one of them exists
        else:
            # Allow for custom q for lft different than the original
            if q is None:
                q = G.DiscretizationMatrix

    if isinstance(G, Transfer):
        if G._isgain:
            return Transfer(G.to_array)
        T = transfer_to_state(G)
        undiscretized_args = _undiscretize(T, dt, method, prewarp_at, q)
        return state_to_transfer(State(*undiscretized_args))
    else:
        if G._isgain:
            return State(G.to_array)
        undiscretized_args = _undiscretize(G, dt, method, prewarp_at, q)
        return State(*undiscretized_args)


def _undiscretize(T, dt, method, prewarp_at, q):

    m, n = T.NumberOfInputs, T.NumberOfStates

    # Error message for log based zoh, foh
    logmsg = ('The matrix logarithm returned a complex array, probably due'
              ' to poles on the negative real axis, and a continous-time '
              'model cannot be obtained via this method without '
              'perturbations.')

    if method == 'zoh':

        if np.any(np.abs(eigvals(T.a)) < np.sqrt(np.spacing(norm(T.a, 1)))):
            raise ValueError('The system has poles near 0, "zoh" method'
                             ' cannot be applied.')
        M = block([[T.a, T.b], [zeros((m, n)), eye(m)]])
        Ms, (sca, _) = matrix_balance(M, permute=0, separate=1)

        eM = logm(M) * (sca[:, None] * np.reciprocal(sca)) * (1/dt)
        if np.any(eM.imag):
            raise ValueError(logmsg)

        Ac, Bc, Cc, Dc = eM[:n, :n], eM[:n, n:], T.c, T.d

    elif method == 'foh':
        """
        We use the explicit formulas for Φ, Γ₁, Γ₂

                      [ Φ  Γ₁  Γ₂]
                 logm [ 0  I   I ]
                      [ 0  0   I ]

        Here a direct logarithm won't work since we don't have Γ terms. However
        discrete time matrices are given by

                Ad = exp(Ac*t) = Φ
                Bd = Ac⁻²(Φ - I)²Bc/t                      (*)
                Cd = Cc
                Dc = Dc + Cc[Ac⁻²(Φ - I) - Ac⁻¹]Bc/t

        since Ac⁻¹(Φ-I) = ∫exp(Ac)dt and Ac⁻¹ commutes with Φ, Bd*t = Φ²*Bc,
        the solution follows.
        """
        if np.any(np.abs(eigvals(T.a)) < np.sqrt(np.spacing(norm(T.a, 1)))):
            raise ValueError('The system has poles near 0, "foh" method'
                             ' cannot be applied.')

        M = block([[T.a, dt*T.b, zeros((n, m))],  # Notice dt factor from (*)
                   [zeros((m, n)), eye(m), eye(m)],
                   [zeros((m, n+m)), eye(m)]])
        Ms, (sca, _) = matrix_balance(M, permute=0, separate=1)
        # Look out for the initial dt factor of T.b
        eM = logm(M)/dt * (sca[:, None] * np.reciprocal(sca))
        if np.any(eM.imag):
            raise ValueError(logmsg)

        Ac = eM[:n, :n]
        Bc0, Bc1 = eM[:n, n:n+m], -eM[:n, n+m:]

        # Now we have Bc0 = Ac⁻(Φ - I)Bc, logm once again to get Bc
        M = block([[T.a, Bc0],
                   [zeros((m, n)), eye(m)]])
        Ms, (sca, _) = matrix_balance(M, permute=0, separate=1)
        eM = logm(M)/dt * (sca[:, None] * np.reciprocal(sca))
        if np.any(eM.imag):
            raise ValueError(logmsg)

        # Now back-substitute
        Bc, Cc, Dc = eM[:n, n:], T.c, T.d - T.c @ Bc1

    elif method in ('bilinear', 'tustin', 'trapezoidal'):
        if prewarp_at == 0.:
            q = np.array([[-2/dt, 2/np.sqrt(dt)], [2/np.sqrt(dt), -1]])
        else:
            if 1/(2*dt) <= prewarp_at:
                raise ValueError('Prewarping frequency is beyond the Nyquist'
                                 ' rate. It has to satisfy 0 < w < 1/(2*Δt)'
                                 ' and Δt being the sampling period in '
                                 'seconds. Δt={0} is given, hence the maximum'
                                 ' allowed is {1} Hz.'.format(dt, 1/(2*dt)))
            prew_rps = 2 * np.pi * prewarp_at
            sq2tan = np.sqrt(2*np.tan(prew_rps * dt / 2)/prew_rps)
            q = np.array([[-2/sq2tan**2, 1/sq2tan], [1/sq2tan, -1]])

        Ac, Bc, Cc, Dc = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    elif method in ('forward euler', 'forward difference',
                    'forward rectangular', '>>'):
        q = np.array([[-1/dt, 1/np.sqrt(dt)], [1/np.sqrt(dt), 0]])
        Ac, Bc, Cc, Dc = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    elif method in ('backward euler', 'backward difference',
                    'backward rectangular', '<<'):
        # nonproper via lft, compute explicitly.
        with catch_warnings(record=True) as war:
            simplefilter("always")
            try:
                iAd = solve(T.a, eye(n))
            except LinAlgError:
                raise ValueError('The state matrix has eigenvalues at zero '
                                 'and this conversion method can\'t be used.')

        if len(war) > 0:
            warn('The state matrix has eigenvalues too close to imaginary'
                 ' axis. This conversion method might give inaccurate '
                 'results', _rcond_warn, stacklevel=2)

        Ac = np.eye(n) - iAd
        Ac /= dt
        Bc = 1/np.sqrt(dt) * (iAd @ T.b)
        Cc = 1/np.sqrt(dt) * (T.c @ iAd)
        Dc = T.d - T.c @ iAd @ T.b

    elif method == 'lft':
        if q is None:
            raise ValueError('"lft" method requires a 2x2 interconnection '
                             'matrix "q" between s and z indeterminates.')
        Ac, Bc, Cc, Dc = _simple_lft_connect(q, T.a, T.b, T.c, T.d)

    return Ac, Bc, Cc, Dc


def rediscretize(G, dt, method='tustin', alpha=0.5):
    """
    .. todo:: Not implemented yet!
    """
    pass


def _simple_lft_connect(q, A, B, C, D):
    """
    A helper function for simple upper LFT connection with well-posedness
    check for discrete/continuous conversion purposes.

    Here we form the following star product

                       ┌───────┐      ─┐
                       │  1    │       │
                    ┌──┤ ─── I │<─┐    │
                    │  │  z    │  │    │
                    │  └───────┘  │    │    1
                    │             │    ├─  ─── I
                    │   ┌─────┐   │    │    s
                    └──>│     ├───┘    │
                        │  q  │        │
                    ┌──>│     ├───┐    │
                    │   └─────┘   │   ─┘
                    │             │
                    │   ┌─────┐   │
                    └───┤     │<──┘
                        │  T  │
                    <───┤     │<──
                        └─────┘

    Here q is whatever the rational mapping that links s to z in
    the following sense:

        1         1                    1        1
       ─── = F_u(───,q) = q_22 + q_21 ─── (I - ─── q_11)⁻¹ q12
        s         z                    z        z

    where F_u denotes the upper linear fractional representation. As an
    example, for the usual discretization cases, the map is

                      ┌         │         ┐
                      │    1    │    √T   │
                  Q = │─────────┼─────────│
                      │   √T    │   T*α   │
                      └         │         ┘

    with α defined as

    α = 1   --> backward difference, (backward euler)
    α = 0.5 --> Tustin, (bilinear)
    α = 0   --> forward difference (forward euler)

    """
    q = np.asarray(q)
    if q.ndim != 2 or q.shape != (2, 2):
        raise ValueError('q must be exactly a 2x2 array')

    n = A.shape[0]
    ij = np.eye(n)
    q11, q12, q21, q22 = q.ravel()
    if q22 == 0.:
        NAinv = ij
    else:
        with catch_warnings(record=True) as war:
            simplefilter('always')
            try:
                NAinv = solve(ij - q22*A, eye(n))
            except LinAlgError:
                raise LinAlgError('The resulting state matrix during this '
                                  'operation leads to a singular matrix '
                                  'inversion and hence cannot be computed.')

        if len(war) > 0:
            warn('The resulting state matrix during this operation '
                 'has an eigenstructure very close to unity. Hence '
                 'the final model might be inaccurate', _rcond_warn,
                 stacklevel=2)

    # Compute the star product
    Ad = q11*ij + q12*A @ NAinv*q21
    Bd = q12*(A @ NAinv*q22 + ij) @ B if q22 != 0. else q12*B
    Cd = C @ NAinv*q21
    Dd = D + C @ NAinv*q22 @ B if q22 != 0. else D

    return Ad, Bd, Cd, Dd
