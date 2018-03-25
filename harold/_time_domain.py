"""
The MIT License (MIT)

Copyright (c) 2018 Ilhan Polat

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
from harold._classes import Transfer, transfer_to_state
from harold._discrete_funcs import discretize
from harold._arg_utils import _check_for_state, _check_for_state_or_transfer

__all__ = ['simulate_linear_system', 'simulate_step_response',
           'simulate_impulse_response']


def simulate_linear_system(sys, u, t=None, x0=None, per_channel=False):
    """
    Compute the linear model response to an input array sampled at given time
    instances.

    Parameters
    ----------
    sys : {State, Transfer}
        The system model to be simulated
    u : array_like
        The real-valued input sequence to force the model. 1D arrays for single
        input models and 2D arrays that has as many columns as the number of
        inputs are valid inputs.
    t : array_like, optional
        The real-valued sequence to be used for the evolution of the system.
        The values should be equally spaced otherwise an error is raised. For
        discrete time models increments different than the sampling period also
        raises an error. On the other hand for discrete models this can be
        omitted and a time sequence will be generated automatically.
    x0 : array_like, optional
        The initial condition array. If omitted an array of zeros is assumed.
        Note that Transfer models by definition assume zero initial conditions
        and will raise an error.
    per_channel : bool, optional
        If this is set to True and if the system has multiple inputs, the
        response of each input is returned individually. For example, if a
        system has 4 inputs and 3 outputs then the response shape becomes
        (num, p, m) instead of (num, p) where k-th slice (:, :, k) is the
        response from the k-th input channel. For single input systems, this
        keyword has no effect.

    Returns
    -------
    yout : ndarray
        The resulting response array. The array is 1D if sys is SISO and
        has p columns if sys has p outputs.
    tout : ndarray
        The time sequence used in the simulation. If the parameter t is not
        None then a copy of t is given.

    Notes
    -----
    For Transfer models, first conversion to a state model is performed and
    then the resulting model is used for computations.

    """
    _check_for_state_or_transfer(sys)

    # Quick initial condition checks
    if x0 is not None:
        if sys._isgain:
            raise ValueError('Static system models can\'t have initial '
                             'conditions set.')
        if isinstance(sys, Transfer):
            raise ValueError('Transfer models can\'t have initial conditions '
                             'set.')
        x0 = np.asarray(x0, dtype=float).squeeze()
        if x0.ndim > 1:
            raise ValueError('Initial condition can only be a 1D array.')
        else:
            x0 = x0[:, None]

        if sys.NumberOfStates != x0.size:
            raise ValueError('The initial condition size does not match the '
                             'number of states of the model.')

    # Always works with State Models
    try:
        _check_for_state(sys)
    except ValueError:
        sys = transfer_to_state(sys)

    n, m = sys.NumberOfStates, sys.shape[1]

    is_discrete = sys.SamplingSet == 'Z'
    u = np.asarray(u, dtype=float).squeeze()
    if u.ndim == 1:
        u = u[:, None]

    t = _check_u_and_t_for_simulation(m, sys._dt, u, t, is_discrete)
    # input and time arrays are regular move on

    # Static gains are simple matrix multiplications with no x0
    if sys._isgain:
        if sys._isSISO:
            yout = u * sys.d.squeeze()
        else:
            # don't bother for single inputs
            if m == 1:
                per_channel = False

            if per_channel:
                yout = np.einsum('ij,jk->ikj', u, sys.d.T)
            else:
                yout = u @ sys.d.T

    # Dynamic model
    else:
        # TODO: Add FOH discretization for funky input
        # ZOH discretize the continuous system based on the time increment
        if not is_discrete:
            sys = discretize(sys, t[1]-t[0], method='zoh')

        sample_num = len(u)
        a, b, c, d = sys.matrices
        # Bu and Du are constant matrices so get them ready (transposed)
        M_u = np.block([b.T, d.T])
        at = a.T

        # Explicitly skip single inputs for per_channel
        if m == 1:
            per_channel = False

        # Shape the response as a 3D array
        if per_channel:
            xout = np.empty([sample_num, n, m], dtype=float)

            for col in range(m):
                xout[0, :, col] = 0. if x0 is None else x0.T
                Bu = u[:, [col]] @ b.T[[col], :]

                # Main loop for xdot eq.
                for row in range(1, sample_num):
                    xout[row, :, col] = xout[row-1, :, col] @ at + Bu[row-1]

            # Get the output equation for each slice of inputs
            # Cx + Du
            yout = np.einsum('ijk,jl->ilk', xout, c.T) + \
                np.einsum('ij,jk->ikj', u, d.T)
        # Combined output
        else:
            BDu = u @ M_u
            xout = np.empty([sample_num, n], dtype=float)
            xout[0] = 0. if x0 is None else x0.T
            # Main loop for xdot eq.
            for row in range(1, sample_num):
                xout[row] = (xout[row-1] @ at) + BDu[row-1, :n]

            # Now we have all the state evolution get the output equation
            yout = xout @ c.T + BDu[:, n:]

    return yout, t


def simulate_step_response(sys, t=None):
    """
    Compute the linear model response to an Heaviside function (or all-ones
    array) sampled at given time instances.

    If the time array is omitted then a time sequence is generated based on
    the slowest pole of sys based on the straightforward settling time
    estimate

        ts = 4/(ζω)

    and its discrete equivalent. The sampling period is also approximated by
    the fastest stable pole. If the system has all nonnegative poles, -1. is
    used for the computations as the threshold for the settling times.

    Parameters
    ----------
    sys : {State, Transfer}
        The system model to be simulated
    t : array_like
        The real-valued sequence to be used for the evolution of the system.
        The values should be equally spaced otherwise an error is raised. For
        discrete time models increments different than the sampling period also
        raises an error. On the other hand for discrete models this can be
        omitted and a time sequence will be generated automatically.
    """
    _check_for_state_or_transfer(sys)
    if t is None:
        tf, ts = _compute_tfinal_and_dt(sys)

        # Massage tf such that it is an integer multiple of ts
        mult = int(tf // ts)
        if sys._isdiscrete:
            t = np.linspace(0, sys._dt*(mult-1), num=mult)
        else:
            t = np.linspace(0, ts*(mult-1), num=mult)

    m = sys.shape[1]
    u = np.ones([len(t), m])

    return simulate_linear_system(sys, u=u, t=t, per_channel=1)


def simulate_impulse_response(sys, t=None):
    """
    Compute the linear model response to an Dirac delta pulse (or all-zeros
    array except the first sample being 1. at each channel) sampled at given
    time instances.

    If the time array is omitted then a time sequence is generated based on
    the slowest pole of sys based on the straightforward settling time
    estimate

        ts = 4/(ζω)

    and its discrete equivalent. The sampling period is also approximated by
    the fastest stable pole. If the system has all nonnegative poles, -1. is
    used for the computations as the threshold for the settling times.

    Parameters
    ----------
    sys : {State, Transfer}
        The system model to be simulated
    t : array_like
        The real-valued sequence to be used for the evolution of the system.
        The values should be equally spaced otherwise an error is raised. For
        discrete time models increments different than the sampling period also
        raises an error. On the other hand for discrete models this can be
        omitted and a time sequence will be generated automatically.
    """
    _check_for_state_or_transfer(sys)
    if t is None:
        tf, ts = _compute_tfinal_and_dt(sys)
        # Massage tf such that it is an integer multiple of ts
        mult = int(tf // ts)
        t = np.linspace(0, ts*(mult-1), num=mult)

    m = sys.shape[1]
    u = np.zeros([len(t), m])
    u[0] = 1.

    return simulate_linear_system(sys, u=u, t=t, per_channel=1)


def _compute_tfinal_and_dt(sys):
    """
    Helper function to estimate a final time and a sampling period for
    time domain simulations.
    """
    # if a static model is given, don't bother with checks
    if sys._isgain:
        return (50*sys._dt, sys._dt) if sys._isdiscrete else (5, 0.1)
    # Make a rough prediction about the time required based on the slowest mode
    p = sys.poles
    if sys._isdiscrete:

        pmag = np.abs(p)
        stab_p = pmag[pmag < 1.]

        # If no stable pole quick exit
        if stab_p.size == 0.:
            return sys._dt*100, sys._dt
        # s-domain equivalents
        ps = np.log(stab_p)/sys._dt

    else:
        ps = p[p.real < 0]
        # If no stable pole quick exit
        if ps.size == 0.:
            return 20, 1/np.max([np.abs(p).max(), .1])

    wns = np.abs(ps)
    zetas = -np.cos(np.angle(ps))

    # Threshold omega 0.001
    slow_wn = np.max([np.min(wns), 0.001])
    # Threshold zeta at 0.1 to avoid too large tfinal
    slow_zeta = np.max([np.min([zetas[np.argmin(wns)], 0.1]), 0.1])

    return 4/(slow_wn*slow_zeta), sys._dt if sys._isdiscrete else 1/wns.max()


def _check_u_and_t_for_simulation(m, dt, u, t, isdiscrete):
    """
    Helper function to validate the input arguments for simulate_linear_system
    """
    # Discrete models can omit t array, make one here for convenience
    if t is None:
        if not isdiscrete:
            raise ValueError('Continuous time models need an evenly spaced '
                             'time sequence from which the sampling period '
                             'will be obtained.')
        else:
            u_samples = len(u)
            t = np.linspace(0, (u_samples-1)*dt, num=u_samples)
    else:
        t = np.asarray(t, dtype=float).squeeze()
        if t.ndim != 1:
            raise ValueError('Time array needs to be a 1D array.')
        t_diff = np.diff(t)
        if not np.allclose(t_diff, t_diff[0]) or not t_diff[0] > 0.:
            raise ValueError('Time array should be equally spaced and '
                             'increasing.')

        if isdiscrete and not np.isclose(dt, t_diff[0]):
            raise ValueError('Time array increment {} is not equal to the'
                             ' model sampling period {}.'.format(t_diff[0],
                                                                 dt))

    if u.size < 1:
        raise ValueError('The input array should at least have one point.')

    # First dimension is always # of samples
    if len(u) != len(t):
        raise ValueError('The input and time arrays should have the same'
                         ' length. t: {} vs. u: {}'.format(t.shape,
                                                           u.shape))

    if u.shape[1] != m:
        raise ValueError('Number of input columns ({}) don\'t match the number'
                         ' of inputs ({}) of the given model.'
                         ''.format(u.shape[1], m))
    return t
