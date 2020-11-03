import numpy as np
from numpy import (reciprocal, einsum, maximum, minimum, zeros_like,
                   atleast_1d, squeeze)
from scipy.linalg import eig, eigvals, matrix_balance, norm
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
    the poles of the model.

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

    Returns
    -------
    yout : ndarray
        The resulting response array. The array is 1D if sys is SISO and
        has p columns if sys has p outputs. If there are also m inputs the
        array is 3D array with the shape (<num of samples>, p, m)
    tout : ndarray
        The time sequence used in the simulation. If the parameter t is not
        None then a copy of t is given.

    """
    _check_for_state_or_transfer(sys)
    # Always works with State Models
    try:
        _check_for_state(sys)
    except ValueError:
        sys = transfer_to_state(sys)

    if t is None:
        tf, ts = _compute_tfinal_and_dt(sys)
        t = np.arange(0, tf+ts, ts, dtype=float)
    else:
        t, ts = _check_custom_time_input(t)

    m = sys.shape[1]
    u = np.ones([len(t), m], dtype=float)

    return simulate_linear_system(sys, u=u, t=t, per_channel=1)


def simulate_impulse_response(sys, t=None):
    """
    Compute the linear model response to an Dirac delta pulse (or all-zeros
    array except the first sample being 1/dt at each channel) sampled at given
    time instances.

    If the time array is omitted then a time sequence is generated based on
    the poles of the model.

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

    Returns
    -------
    yout : ndarray
        The resulting response array. The array is 1D if sys is SISO and
        has p columns if sys has p outputs. If there are also m inputs the
        array is 3D array with the shape (<num of samples>, p, m)
    tout : ndarray
        The time sequence used in the simulation. If the parameter t is not
        None then a copy of t is given.

    """
    _check_for_state_or_transfer(sys)
    # Always works with State Models
    try:
        _check_for_state(sys)
    except ValueError:
        sys = transfer_to_state(sys)

    if t is None:
        tf, ts = _compute_tfinal_and_dt(sys, is_step=False)
        t = np.arange(0, tf+ts, ts, dtype=float)
    else:
        t, ts = _check_custom_time_input(t)

    m = sys.shape[1]
    u = np.zeros([len(t), m], dtype=float)
    u[0] = 1./ts

    return simulate_linear_system(sys, u=u, t=t, per_channel=1)


def _compute_tfinal_and_dt(sys, is_step=True):
    """
    Helper function to estimate a final time and a sampling period for
    time domain simulations. It is essentially geared towards impulse response
    but is also used for step responses.

    For discrete-time models, obviously dt is inherent and only tfinal is
    computed.

    Parameters
    ----------
    sys : {State, Transfer}
        The system to be investigated
    is_step : bool
        Scales the dc value by the magnitude of the nonzero mode since
        integrating the impulse response gives ∫exp(-λt) = -exp(-λt)/λ.
        Default is True.

    Returns
    -------
    tfinal : float
        The final time instance for which the simulation will be performed.
    dt : float
        The estimated sampling period for the simulation.

    Notes
    -----
    Just by evaluating the fastest mode for dt and slowest for tfinal often
    leads to unnecessary, bloated sampling (e.g., Transfer(1,[1,1001,1000]))
    since dt will be very small and tfinal will be too large though the fast
    mode hardly ever contributes. Similarly, change the numerator to [1, 2, 0]
    and the simulation would be unnecessarily long and the plot is virtually
    an L shape since the decay is so fast.

    Instead, a modal decomposition in time domain hence a truncated ZIR and ZSR
    can be used such that only the modes that have significant effect on the
    time response are taken. But the sensitivity of the eigenvalues complicate
    the matter since dλ = <w, dA*v> with <w,v> = 1. Hence we can only work
    with simple poles with this formulation. See Golub, Van Loan Section 7.2.2
    for simple eigenvalue sensitivity about the nonunity of <w,v>. The size of
    the response is dependent on the size of the eigenshapes rather than the
    eigenvalues themselves.

    """
    sqrt_eps = np.sqrt(np.spacing(1.))
    min_points = 100  # min number of points
    min_points_z = 20  # min number of points
    max_points = 10000  # max number of points
    max_points_z = 75000  # max number of points for discrete models
    default_tfinal = 5  # Default simulation horizon
    total_cycles = 5  # number of cycles for oscillating modes
    pts_per_cycle = 25  # Number of points divide a period of oscillation
    log_decay_percent = np.log(100)  # Factor of reduction for real pole decays

    # if a static model is given, don't bother with checks
    if sys._isgain:
        if sys._isdiscrete:
            return sys._dt*min_points_z, sys._dt
        else:
            return default_tfinal, default_tfinal / min_points

    if sys._isdiscrete:
        # System already has sampling fixed  hence we can't fall into the same
        # trap mentioned above. Just get nonintegrating slow modes together
        # with the damping.
        dt = sys._dt
        tfinal = default_tfinal
        p = eigvals(sys.a)
        # Array Masks
        # unstable
        m_u = (np.abs(p) >= 1 + sqrt_eps)
        p_u, p = p[m_u], p[~m_u]
        if p_u.size > 0:
            m_u = (p_u.real < 0) & (np.abs(p_u.imag) < sqrt_eps)
            t_emp = np.max(log_decay_percent / np.abs(np.log(p_u[~m_u])/dt))
            tfinal = max(tfinal, t_emp)

        # zero - negligible effect on tfinal
        m_z = np.abs(p) < sqrt_eps
        p = p[~m_z]
        # Negative reals- treated as oscillary mode
        m_nr = (p.real < 0) & (np.abs(p.imag) < sqrt_eps)
        p_nr, p = p[m_nr], p[~m_nr]
        if p_nr.size > 0:
            t_emp = np.max(log_decay_percent / np.abs((np.log(p_nr)/dt).real))
            tfinal = max(tfinal, t_emp)
        # discrete integrators
        m_int = (p.real - 1 < sqrt_eps) & (np.abs(p.imag) < sqrt_eps)
        p_int, p = p[m_int], p[~m_int]
        # pure oscillatory modes
        m_w = (np.abs(np.abs(p) - 1) < sqrt_eps)
        p_w, p = p[m_w], p[~m_w]
        if p_w.size > 0:
            t_emp = total_cycles * 2 * np.pi / np.abs(np.log(p_w)/dt).min()
            tfinal = max(tfinal, t_emp)

        if p.size > 0:
            t_emp = log_decay_percent / np.abs((np.log(p)/dt).real).min()
            tfinal = max(tfinal, t_emp)

        if p_int.size > 0:
            tfinal = tfinal * 5

        # Make tfinal an integer multiple of dt
        num_samples = tfinal // dt
        if num_samples > max_points_z:
            tfinal = dt * max_points_z
        else:
            tfinal = dt * num_samples

        return tfinal, dt

    # Improve conditioning via balancing and zeroing tiny entries
    # See <w,v> for [[1,2,0], [9,1,0.01], [1,2,10*np.pi]] before/after balance
    b, (sca, perm) = matrix_balance(sys.a, separate=True)
    p, l, r = eig(b, left=True, right=True)
    # Reciprocal of inner product <w,v> for each λ, (bound the ~infs by 1e12)
    # G = Transfer([1], [1,0,1]) gives zero sensitivity (bound by 1e-12)
    eig_sens = reciprocal(maximum(1e-12, einsum('ij,ij->j', l, r).real))
    eig_sens = minimum(1e12, eig_sens)
    # Tolerances
    p[np.abs(p) < np.spacing(eig_sens * norm(b, 1))] = 0.
    # Incorporate balancing to outer factors
    l[perm, :] *= reciprocal(sca)[:, None]
    r[perm, :] *= sca[:, None]
    w, v = sys.c @ r, l.T.conj() @ sys.b

    origin = False
    # Computing the "size" of the response of each simple mode
    wn = np.abs(p)
    if np.any(wn == 0.):
        origin = True

    dc = zeros_like(p, dtype=float)
    # well-conditioned nonzero poles, np.abs just in case
    ok = np.abs(eig_sens) <= 1/sqrt_eps
    # the averaged t→∞ response of each simple λ on each i/o channel
    # See, A = [[-1, k], [0, -2]], response sizes are k-dependent (that is
    # R/L eigenvector dependent)
    dc[ok] = norm(v[ok, :], axis=1)*norm(w[:, ok], axis=0)*eig_sens[ok]
    dc[wn != 0.] /= wn[wn != 0] if is_step else 1.
    dc[wn == 0.] = 0.
    # double the oscillating mode magnitude for the conjugate
    dc[p.imag != 0.] *= 2

    # Now get rid of noncontributing integrators and simple modes if any
    relevance = (dc > 0.1*dc.max()) | ~ok
    psub = p[relevance]
    wnsub = wn[relevance]

    tfinal, dt = [], []
    ints = wnsub == 0.
    iw = (psub.imag != 0.) & (np.abs(psub.real) <= sqrt_eps)

    # Pure imaginary?
    if np.any(iw):
        tfinal += (total_cycles * 2 * np.pi / wnsub[iw]).tolist()
        dt += (2 * np.pi / pts_per_cycle / wnsub[iw]).tolist()
    # The rest ~ts = log(%ss value) / exp(Re(λ)t)
    texp_mode = log_decay_percent / np.abs(psub[~iw & ~ints].real)
    tfinal += texp_mode.tolist()
    dt += minimum(texp_mode / 50,
                  (2 * np.pi / pts_per_cycle / wnsub[~iw & ~ints])).tolist()

    # All integrators?
    if len(tfinal) == 0:
        return default_tfinal*5, default_tfinal*5/min_points

    tfinal = np.max(tfinal)*(5 if origin else 1)
    dt = np.min(dt)

    dt = tfinal / max_points if tfinal // dt > max_points else dt
    tfinal = dt * min_points if tfinal // dt < min_points else tfinal

    return tfinal, dt


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


def _check_custom_time_input(t):
    """
    Helper function for simple and rather expensive checks for sanity
    """
    t = atleast_1d(t)
    if t.ndim > 1:
        t = squeeze(t)
        if t.ndim > 1:
            raise ValueError('Time array should be a 1D array but has '
                             '{} nontrivial dimensions'.format(t.ndim))
    if t.size < 2:
        raise ValueError('Time array should have at least two data points.')
    dt = t[1] - t[0]
    if dt <= 0.:
        raise ValueError('The time increment dt cannot be negative; '
                         'Difference of the first two samples t1 - t0 = {}'
                         ''.format(dt))
    # np.diff is somewhat slower than the diff of the views
    if not np.allclose(t[1:] - t[:-1], dt):
        raise ValueError('Supplied time array is not numerically equally '
                         'spaced (checked via numpy.allclose).')

    return t, dt
