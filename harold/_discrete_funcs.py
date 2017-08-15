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
from numpy.linalg import cond
from scipy.linalg import expm, logm, kron, solve

from ._classes import Transfer, State, transfer_to_state, state_to_transfer
from ._global_constants import _KnownDiscretizationMethods
from ._aux_linalg import matrix_slice

__all__ = ['discretize', 'undiscretize']


def discretize(G, dt, method='tustin', PrewarpAt=0., q=None):
    if not isinstance(G, (Transfer, State)):
        raise TypeError('I can only convert State or Transfer objects but I '
                        'found a \"{0}\" object.'.format(type(G).__name__)
                        )
    if G.SamplingSet == 'Z':
        raise TypeError('The argument is already modeled as a '
                        'discrete-time system.')

    if isinstance(G, Transfer):
        T = transfer_to_state(G)
    else:
        T = G

    if G._isgain:
        Gd = G
        Gd.SamplingPeriod = dt

    else:

        discretized_args = __discretize(T, dt, method, PrewarpAt, q)

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
            Gd.PrewarpFrequency = PrewarpAt

    return Gd


def __discretize(T, dt, method, PrewarpAt, q):
    """
    Actually, I think that presenting this topic as a numerical
    integration problem is confusing more than it explains. Most
    items here can be presented as conformal mappings and nobody
    needs to be limited to riemann sums of particular shape. As
    I found that scipy version of this, adopts Zhang SICON 2007
    parametrization which surprisingly fresh!

    Here I "generalized" to any rational function representation
    if you are into that mathematician lingo (see the 'fancy'
    ASCII art below). I used LFTs instead, for all real rational
    approx. mappings (for whoever wants to follow that rabbit).
    """

    m, n = T.shape[1], T.NumberOfStates

    if method == 'zoh':
        """
        Zero-order hold is not much useful for linear systems and
        in fact it should be discouraged since control problems
        don't have boundary conditions as in stongly nonlinear
        FEM simulations of CFDs so on. Most importantly it is not
        stability-invariant which defeats its purpose. But whatever



        This conversion is usually done via the expm() identity

            [A | B]   [ exp(A) | int(exp(A))*B ]   [ Ad | Bd ]
        expm[- - -] = [------------------------] = [---------]
            [0 | 0]   [   0    |       I       ]   [ C  | D  ]
        """

        M = np.r_[np.c_[T.a, T.b], np.zeros((m, m+n))]
        eM = expm(M*dt)
        Ad, Bd, Cd, Dd = eM[:n, :n], eM[:n, n:], T.c, T.d

    elif method == 'lft':
        """
        Here we form the following star product
                                      _
                       ---------       |
                       |  1    |       |
                    ---| --- I |<--    |
                    |  |  z    |  |    |
                    |  ---------  |    |
                    |             |    |> this is the lft of (1/s)*I
                    |   -------   |    |
                    --->|     |----    |
                        |  q  |        |
                    --->|     |----    |
                    |   -------   |   _|
                    |             |
                    |   -------   |
                    ----|     |<--|
                        |  T  |
                    <---|     |<---
                        -------

        Here q is whatever the rational mapping that links s to z in
        the following sense:

            1         1                    1        1
           --- = F_u(---,q) = q_22 + q_21 --- (I - --- q_11)⁻¹ q12
            s         z                    z        z

        where F_u denotes the upper linear fractional representation.

        As an example, for the usual discretization cases, the map is

                  [     I     |  sqrt(T)*I ]
              Q = [-----------|------------]
                  [ sqrt(T)*I |    T*α*I   ]

        with α defined as in Zhang 2007 SICON.

        α = 0   --> backward diff, (backward euler)
        α = 0.5 --> Tustin,
        α = 1   --> forward difference (forward euler)

        """

        # TODO: Check if interconnection is well-posed !!!!

        if q is None:
            raise ValueError('\"lft\" method requires an interconnection '
                             'matrix \"q" between s and z variables.')

        # Copy n times for n integrators
        q11, q12, q21, q22 = (kron(np.eye(n), x) for x in matrix_slice(
                              q, (-1, -1)))

        # Compute the star product
        ZAinv = solve(np.eye(n) - q22.dot(T.a), q21)
        AZinv = solve(np.eye(n) - T.a.dot(q22), T.b)

        Ad = q11 + q12.dot(T.a.dot(ZAinv))
        Bd = q12.dot(AZinv)
        Cd = T.c.dot(ZAinv)
        Dd = T.d + T.c.dot(q22.dot(AZinv))

    elif method in ('bilinear', 'tustin', 'trapezoidal'):
        if not PrewarpAt == 0.:
            if 1/(2*dt) < PrewarpAt:
                raise ValueError('Prewarping Frequency is beyond the Nyquist'
                                 ' rate. It has to satisfy 0 < w < 1/(2*Δt)'
                                 ' and Δt being the sampling period in '
                                 'seconds. Δt={0} is given, hence the max.'
                                 ' allowed is {1} Hz.'.format(dt, 1/(2*dt))
                                 )

            TwoTanw_Over_w = np.tan(2*np.pi*PrewarpAt*dt/2)/(2*np.pi*PrewarpAt)
            q = np.array([[1, np.sqrt(2*TwoTanw_Over_w)],
                          [np.sqrt(2*TwoTanw_Over_w), TwoTanw_Over_w]])
        else:
            q = np.array([[1, np.sqrt(dt)], [np.sqrt(dt), dt/2]])

        return __discretize(T, dt, "lft", 0., q)

    elif method in ('forward euler', 'forward difference',
                    'forward rectangular', '>>'):
        return __discretize(T, dt, "lft", 0, q=np.array([[1, np.sqrt(dt)],
                                                         [np.sqrt(dt), 0]]))

    elif method in ('backward euler', 'backward difference',
                    'backward rectangular', '<<'):
        return __discretize(T, dt, "lft", 0, q=np.array([[1, np.sqrt(dt)],
                                                        [np.sqrt(dt), dt]]))

    else:
        raise ValueError('I don\'t know that discretization method. But '
                         'I know {0} methods.'
                         ''.format(_KnownDiscretizationMethods)
                         )

    return Ad, Bd, Cd, Dd, dt


def undiscretize(G, use_method=None):
    """
    Converts a discrete time system model continuous system model.
    If the model has the Discretization Method set, then uses that
    discretization method to reach back to the continous system model.

    Parameters
    ----------
    G : State()
        System to be undiscretized
    use_method : str
        The method to use for converting discrete model to continous.

    """
    if not isinstance(G, (Transfer, State)):
        raise TypeError('The argument is not transfer '
                        'function or a state\nspace model.'
                        )

    if G.SamplingSet == 'R':
        raise TypeError('The argument is already modeled as a '
                        'continuous time system.')

    if G._isgain:
        Gc = G
        Gc.SamplingPeriod = None
    else:
        args = __undiscretize(G, use_method)
        if isinstance(G, State):
            Gc = State(*args)
        else:
            Gss = State(*args)
            Gc = state_to_transfer(Gss)

    return Gc


def __undiscretize(G, method_to_use):

    if isinstance(G, Transfer):
        T = transfer_to_state(G)
    else:
        T = G

    m, n = T.NumberOfInputs, T.NumberOfStates
    dt = G.SamplingPeriod

    if method_to_use is None:
        if 'with it' in G.DiscretizedWith:  # Check if warning comes back
            missing_method = True
        else:
            method_to_use = G.DiscretizedWith

    if method_to_use == 'zoh':
        M = np.r_[
                   np.c_[T.a, T.b],
                   np.c_[np.zeros((m, n)), np.eye(m)]
                  ]
        eM = logm(M)*(1/T.SamplingPeriod)
        Ac, Bc, Cc, Dc = eM[:n, :n], eM[:n, n:], T.c, T.d

    elif (method_to_use in ('bilinear', 'tustin',
                            'trapezoidal') or missing_method):  # Manually
        X = np.eye(n) + T.a
        if 1/np.linalg.cond(X) < 1e-8:  # TODO: Totally psychological limit
            raise ValueError('The discrete A matrix has eigenvalue(s) '
                             'very close to -1 (rcond of I+Ad is {0})'
                             ''.format(1/np.linalg.cond(X)))

        iX = dt/2*(np.eye(n) + T.a)
        Ac = solve(-iX, (np.eye(n)-T.a))
        Bc = solve(iX, T.b)
        Cc = T.c.dot(np.eye(n) + 0.5*dt*solve(iX, (np.eye(n) - T.a)))
        Dc = T.d - 0.5*dt*T.c.dot(solve(iX, T.b))

    elif (method_to_use in ('forward euler', 'forward difference',
                            'forward rectangular', '>>')):
        Ac = -1/dt * (np.eye(n)-T.a)
        Bc = 1/dt * (T.b)
        Cc = T.c
        Dc = T.d

    elif (method_to_use in ('backward euler', 'backward difference',
                            'backward rectangular', '<<')):
        X = T.a
        if 1/cond(X) < 1e-8:  # TODO: Totally psychological limit
            raise ValueError('The discrete A matrix has eigenvalue(s) '
                             'very close to 0 (rcond of I+Ad is {0})'
                             ''.format(1/np.linalg.cond(X)))

        iX = dt*T.a
        Ac = solve(-iX, (np.eye(n) - T.a))
        Bc = solve(iX, T.b)
        Cc = T.c.dot(np.eye(n) + dt*solve(iX, (np.eye(n) - T.a)))
        Dc = T.d - dt*T.c.dot(solve(iX, T.b))

    return Ac, Bc, Cc, Dc


def rediscretize(G, dt, method='tustin', alpha=0.5):
    """
    .. todo:: Not implemented yet!
    """
    pass
