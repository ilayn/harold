import numpy as np
from numpy import (empty, empty_like, rollaxis, block, diag_indices, logspace,
                   geomspace, polyval, zeros, floor, ceil, unique, log10,
                   array)
from ._classes import State, Transfer, transfer_to_state, transmission_zeros
from ._system_funcs import (_minimal_realization_state, hessenberg_realization)
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['frequency_response']


def frequency_response(G, w=None, samples=None, w_unit='Hz', output_unit='Hz',
                       use_minreal=False):
    """
    Compute the frequency response of a State() or Transfer() representation.

    Parameters
    ----------
    G : State of Transfer
        The realization for which the frequency response is computed
    w : array_like, optional
        If not None, then this array will be used as the frequency points. If
        it is None then the routine will examine the pole/zero structure and
        create a frequency grid automatically.
    samples: int, optional
        Lower bound on the number of samples to be evaluated if ``w`` is None
    w_unit : str, optional
        ``Hz`` or ``rad/s``
    output_unit : str, optional
        ``Hz`` or ``rad/s``
    use_minreal : bool, optional
        If set to True, ``G`` will be passed through a minimal realization
        check and its uncontrollable/unobservable modes, if any, will be
        removed.

    Returns
    -------
    fr_arr : ndarray
        The frequency response of the system G with the shape ``(p, m, #freq)``
        For SISO systems, the response is squeezed and a 1D array is returned.
    w : ndarray
        Frequency grid that is used to evaluate the frequency response. The
        input/output units are taken into account.
    """
    # TODO: Investigate if *args,*kwargs is a better syntax here for
    # better argument parsing.

    _check_for_state_or_transfer(G)

    for x in (w_unit, output_unit):
        if x.lower() not in ('hz', 'rad/s'):
            raise ValueError('Frequency unit must be "Hz" or "rad/s". "{0}"'
                             ' is not recognized.'.format(x))

    w = _get_freq_grid(G, w, samples, w_unit, 'rad/s')

    if G._isgain:
        if G._isSISO:
            if isinstance(G, Transfer):
                fr_arr = array([1]*2)*G.num[0, 0]
            else:
                fr_arr = array([1]*2)*G.d[0, 0]
        else:
            if isinstance(G, Transfer):
                fr_arr = zeros((2,)+G.shape) + G.to_array()
            else:
                fr_arr = zeros((2,)+G.shape) + G.d

            fr_arr = rollaxis(fr_arr, 0, 3)
    else:
        p, m = G.shape
        fr_arr = empty((len(w), m, p), dtype='complex')

        if isinstance(G, State):
            for row in range(p):
                aa, bb, cc = hessenberg_realization((G.a, G.b, G.c[[row], :]),
                                                    form='o', invert=True,
                                                    output='matrices')
                if use_minreal:
                    aaa, bbb, ccc = _minimal_realization_state(aa, bb, cc)
                else:
                    aaa, bbb, ccc = aa, bb, cc
                dt = G.SamplingPeriod if G.SamplingSet == 'Z' else None
                fr_arr[:, :, row] = _State_freq_resp(aaa, bbb, ccc[0, -1],
                                                     w, dt)

            if np.any(G.d):
                fr_arr += G.d.T[None, :, :]

            if G._isSISO:
                # Get rid of singleton dimensions
                fr_arr = fr_arr.ravel()
            else:
                # Currently the shape is (freqs,cols,rows) for broadcasting.
                # roll axes to have (row, col, freq) shape
                fr_arr = rollaxis(rollaxis(fr_arr, 0, 3), 1)
        else:
            iw = w*1j
            if G.SamplingSet == 'Z':
                iw = np.exp(iw*G.SamplingPeriod, out=iw)

            if G._isSISO:
                fr_arr = (polyval(G.num[0], iw)/polyval(G.den[0], iw))
            else:
                fr_arr = empty((len(w), m, p), dtype='complex')
                for rows in range(p):
                    for cols in range(m):
                        fr_arr[:, cols, rows] = (
                                polyval(G.num[rows][cols].flatten(), iw) /
                                polyval(G.den[rows][cols].flatten(), iw)
                                )
                fr_arr = rollaxis(rollaxis(fr_arr, 0, 3), 1)

    return fr_arr, w if output_unit == 'rad/s' else w/2/np.pi


def _State_freq_resp(mA, mb, sc, f, dt=None):
    """Generate the frequency response values for a state space representation.

    The realization must be strictly in the observable Hessenberg form.

    Implements the inner loop of Misra, Patel SIMAX 1988 Algo. 3.1 in
    batches of B matrices instead of looping over every column of B.

    Parameters
    ----------
    mA : array_like {n x n}
        The A matrix of the realization in the upper Hessenberg form
    mb : array_like {n x m}
        The B vector of the realization
    sc : float
        The only nonzero coefficient of the o'ble-Hessenberg form
    f : array_like
        The frequency grid
    d : bool, optional
        Evaluate on the imaginary axis or unit circle. Default is imaginary
        axis.

    Returns
    -------
    r  : complex-valued numpy array

    """
    nn, m = mA.shape[0], mb.shape[1]
    r = empty((f.size, m), dtype=complex)
    Ab = block([-mA, mb]).astype(complex)
    U = empty_like(Ab)

    imag_indices = diag_indices(nn)
    # Triangularization of a Hessenberg matrix
    for ind, val in enumerate(f):
        U[:, :] = Ab  # Working copy
        U[imag_indices] += val*1j if dt is None else np.exp(dt*val*1j)
        for x in range(1, nn):
            U[x, x:] -= (U[x, x-1] / U[x-1, x-1]) * U[x-1, x:]

        r[ind, :] = U[-1, -m:] / U[-1, -1-m]

    return r*sc


def _get_freq_grid(G, w, samples, iu, ou):
    """
    Compute a frequency grid of points for the interesting parts.

    Parameters
    ----------
    G : {State, Transfer}
        The system for which the grid will be generated
    w : array_like
        The custom grid given by the user
    samples : int
        Number of samples to be generated
    iu : str
        'Hz' or 'rad/s'
    ou : str
        'Hz' or 'rad/s'

    Returns
    -------
    wout : ndarray
        Resulting grid of frequencies.

    """
    eps = np.spacing(1.)
    sqeps = np.sqrt(eps)

    # internally always work with rad/s to comply with conventions(!).
    # Reconvert at the output if needed
    isDiscrete = G.SamplingSet == 'Z'
    if isDiscrete:
        dt = G.SamplingPeriod
        nyq_freq = np.pi / dt

    # Check the properties of the user-grid and regularize
    if w is not None:
        w = array(w)
        # TODO: Currently this branch always returns 'rad/s'
        w_u = np.array(w.squeeze(), ndmin=1, dtype=float)
        # needs to be a 1D array
        if w_u.ndim > 1:
            raise ValueError('The frequency array should be a 1D float array')

        # convert the internal array to rad/s
        if iu == 'Hz':
            w_u *= 2*np.pi

        # Discrete time behavior doesn't make sense beyond Nyquist freq.
        if isDiscrete:
            w_out = w_u[w_u <= nyq_freq] if nyq_freq < np.max(w_u) else w_u
            if w_out.size < 1:
                raise ValueError('There are no frequency points below the '
                                 'Nyquist frequency: {} Hz.'.format(0.5/dt))
        else:
            w_out = w_u
    # w is None, auto grid creation
    else:
        # We first check whether we need to bother if G is static gain which
        # needs just two samples
        if G._isgain:
            if samples is None:
                samples = 2
            ud = np.log10(nyq_freq) if isDiscrete else 2.
            ld = ud-2 if isDiscrete else -2.
            return logspace(ld, ud, samples)
        else:
            # We acquire the individual SISO system zeros to get a better
            # frequency grid resolution at each channel of a MIMO response.
            # Poles are common.
            pz_list = G.poles.tolist()
            p, m = G.shape
            for row in range(p):
                for col in range(m):
                    # Get zeros of the subsystems without creating State models
                    abcd = transfer_to_state(G[row, col], output='matrices')
                    if abcd[0] is None:
                        abcd = (np.array([]),)*3 + (abcd[-1],)
                    # abc = _minimal_realization_state(*abcd[:-1])
                    tzs = transmission_zeros(*abcd)
                    if tzs.size > 0:
                        pz_list += tzs.tolist()

            # Take a single element from complex pair and remove integrators
            int_pole = 1. if isDiscrete else 0.
            pz_list = np.array(pz_list)
            pz_list = pz_list[~(pz_list == int_pole) & ~(pz_list.imag < 0)]
            # ignore multiplicities
            pz_list = unique(pz_list)

            if pz_list.size == 0:
                # oops all integrators. Add dummy modes for range creation
                if isDiscrete:
                    pz_list = np.array([0.01+0j, -0.8+0j])  # 0j needed for log
                else:
                    pz_list = np.array([0.001, 100])

            if isDiscrete:
                # Poles at the origin are causing warnings
                # Map them to discrete inf
                # J = Transfer(np.poly([-1,1,2]),np.poly([1,0,2]), 0.5)
                # bode_plot(J)

                nz_pz = np.abs(pz_list) > np.spacing(1000.)
                pz_list[~nz_pz] = np.pi/dt
                pz_list[nz_pz] = np.log(pz_list[nz_pz]) / dt

            nat_freq = np.abs(pz_list)
            sorting_ind = nat_freq.argsort()
            nat_freq = nat_freq[sorting_ind]
            damp_fact = np.abs(pz_list.real[sorting_ind])/nat_freq

            smallest_pz = max(nat_freq[0], np.spacing(100.))
            largest_pz = min(nat_freq[-1], 1e16)
            # Add one more decade padding for modes too close to the bounds
            ud, ld = ceil(log10(largest_pz))+1, floor(log10(smallest_pz))-1
            if isDiscrete:
                ud = min(ud, np.log10(nyq_freq))
                # place at least 2 decades if ud and ld too close
                if ud - ld < 1.:
                    ld = floor(ud-2)
            nd = int(ceil(ud - ld))

        # points per decade
        ppd = 15

        if samples is None:
            samples = ppd * nd

        # Sprinkle more points around modes to get continuous bumps and dents.
        # If the plot is going to be between [1e-6, 1e6] Hz, it needs more
        # points than 'ppd' due to the squashed range around a mode. So get a
        # better number of points depending on the range 'nd'.

        # Peak frequency of an underdamped response is ωp = ωc √̅1̅-̅2̅̅ζ̅²
        # the threshold for underdamped modes:
        ind = damp_fact < 0.5
        wp = nat_freq.copy()
        wp[ind] = nat_freq[ind] * np.sqrt(1-2*damp_fact[ind]**2)
        underdamp = np.full_like(wp, np.inf)
        underdamp[ind] = damp_fact[ind]
        w_extra = []

        for idx, fr in enumerate(wp):
            # don't include singularity frequency for undamped modes
            # but miss a little
            if underdamp[idx] < sqeps:
                fr += 100*np.sqrt(np.spacing(fr))

            # Spread is [85%, 115%]
            if not np.isinf(underdamp[idx]):
                num = int(max(5, 5 - ceil(log10(max(underdamp[idx], sqeps)))))
            else:
                num = 3
            w_extra += _loglog_points_around(fr,
                                             w_extra[-1] if w_extra else None,
                                             spread=0.15,
                                             num=num)

        # sprinkle more around the nyquist frequency
        if isDiscrete:
            w_extra += logspace(ud+log10(0.75), 0.995*ud, num=ppd,
                                endpoint=True).tolist()

        # Basis grid
        w = logspace(ld, ud, samples).tolist()
        w = np.sort(w + w_extra)

        # remove the extras if any beyond nyq_freq
        if isDiscrete and w[-1] > 0.995*nyq_freq:
            w = w[w <= 0.995*nyq_freq]

        # Remove accidental exact undamped mode hits from the tails of others
        for p in nat_freq[damp_fact < sqeps]:
            w = w[~(np.abs(w-p) < 100*eps)]

        w_out = w

    if ou == 'Hz':
        w_out /= 2*np.pi

    return w_out


def _loglog_points_around(x, w, spread=0.15, num=10):
    """Place symmetriccally doubly logarithmic points around a given point x.

          ------------------------x------------------------
          o---------o------o---o-o-o-o---o------o---------o

    Skips some points if the points are already covered by the previous call

    Parameters
    ----------
    x: float
        The center point
    w: float, None
        The threshold that the new points should be greater than.
    spread: float
        The number that defines the interval for the total range
    num: int
        2*num is returned around x

    Returns
    -------
    g: ndarray
        Resulting point array

    """
    s, e = x*(1-spread), x*(1+spread)
    sl, el, xl = np.log10([s, e, x])
    # Right shift
    shr = abs(np.ceil(xl)) + 1  # Remove the 0 possibility
    # Left shift
    shl = abs(np.ceil(xl)) + 1  # Remove the 0 possibility
    g = (10**(geomspace(sl-shr, xl-shr, num, endpoint=True) + shr)).tolist()
    # If not filtered as the following large models explode in number of freqs
    # cf. "beam.mat" from SLICOT benchmark suite
    g = g if w is None else [x for x in g if x > w*1.17876863]
    g += (10**(geomspace(xl+shl, el+shl, num, endpoint=False) - shl)).tolist()
    return g
