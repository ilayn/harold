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
from numpy import (empty, empty_like, rollaxis, block, diag_indices, logspace,
                   polyval, zeros, floor, ceil, unique, log10, real, array)
from ._classes import State, Transfer, transfer_to_state, transmission_zeros
from ._system_funcs import (_minimal_realization_state, hessenberg_realization)
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['frequency_response']


def frequency_response(G, w=None, samples=None, w_unit='Hz', output_unit='Hz',
                       use_minreal=False):
    """
    Computes the frequency response of a State() or Transfer() representation.


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

#    if w is None:
    w = _get_freq_grid(G, w, samples, w_unit, 'rad/s')
#    else:
#        w = array(w, dtype=float, ndmin=1)
#        # Convert to rad/s if necessary
#        if w_unit == 'Hz':
#            w *= 2*np.pi

    if G._isgain:
        if G._isSISO:
            if isinstance(G, Transfer):
                fr_arr = array([1]*2)*G.num[0, 0]
            else:
                fr_arr = array([1]*2)*G.d[0, 0]
        else:
            if isinstance(G, Transfer):
                fr_arr = zeros((2,)+G.shape) + array(G.num)
            else:
                fr_arr = zeros((2,)+G.shape) + array(G.d)

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
    """
    This is the low level function to generate the frequency response
    values for a state space representation. The realization must be
    strictly in the observable Hessenberg form.

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
    # internally always work with rad/s to comply with conventions(!).
    # Reconvert at the output if needed
    isDiscrete = G.SamplingSet == 'Z'
    if isDiscrete:
        dt = G.SamplingPeriod
        nyq_freq = 1/(2*dt)*2*np.pi

    # Check the properties of the user-grid and regularize
    if w is not None:
        # TODO: Currently this branch always returns 'rad/s'
        w_u = np.array(np.squeeze(w), ndmin=1, dtype=float).copy()
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
                                 'Nyquist frequency: {} Hz.'.format(nyq_freq))
        else:
            w_out = w_u
    # w is None, auto grid creation
    else:
        # We first check whether we need to bother if G is static gain which
        # needs just two samples
        if G._isgain:
            if samples is None:
                samples = 2
            ud = nyq_freq if isDiscrete else 2.
            ld = np.floor(np.log10(nyq_freq)) if isDiscrete else -2.
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
                    abc = _minimal_realization_state(*abcd[:-1])
                    tzs = transmission_zeros(*abc, abcd[-1])
                    if tzs.size > 0:
                        pz_list += tzs.tolist()

            # ignore multiplicities
            pz_list = unique(pz_list)
            # Remove modes beyond Nyquist Frequency
            if isDiscrete:
                pz_list = pz_list[pz_list <= nyq_freq]
            # Take a single element from complex pair and remove integrators
            int_pole = 1. if isDiscrete else 0.
            pz_list = pz_list[(np.imag(pz_list) >= 0.) & (pz_list != int_pole)]

            if isDiscrete:
                nat_freq = np.abs(np.log(pz_list / dt))
            else:
                nat_freq = np.abs(pz_list)

            damp_fact = np.abs(real(pz_list))/nat_freq

            # np.sqrt(np.spacing(100.)) ~ 1.2e-7
            smallest_pz = max(np.min(nat_freq), 1.3e-7)
            largest_pz = max(np.max(nat_freq), smallest_pz+10)
            # Add one more decade padding for modes too close to the bounds
            ud, ld = ceil(log10(largest_pz))+1, floor(log10(smallest_pz))-1
            if isDiscrete:
                ud = min(ud, np.log10(nyq_freq))
            nd = ud - ld

        # points per decade
        ppd = 15

        if samples is None:
            samples = ppd * nd

        sqeps = np.sqrt(np.spacing(1.))
        # Sprinkle more points around modes to get continuous bumps and dents.
        # If the plot is going to be between [1e-6, 1e6] Hz, it needs more
        # points than 'ppd' due to the squashed range around a mode. So get a
        # better number of points depending on the range 'nd'.

        # Peak frequency of an underdamped response is ωp = ωc √̅1̅-̅2̅̅ζ̅²
        # If solved for 0.5 factor, ζ ~0.61 is the threshold for underdamped
        # modes.
        ind = damp_fact < np.sqrt(0.375)
        wp = nat_freq[ind] * np.sqrt(1-2*damp_fact[ind]**2)
        underdamp = damp_fact[ind]
        spread = log10(0.75)
        w_extra = []
        for idx, fr in enumerate(wp):
            # don't include singularity frequency for undamped modes
            # but miss a little
            if underdamp[idx] < sqeps:
                fr += 100*np.sqrt(np.spacing(fr))

            # TODO: insert points in doubly log-spaced fashion
            # Spread is [75%, 125%]
            frl = log10(fr)
            num = ceil(min(40, 10 - round(5*log10(
                                        max(underdamp[idx], sqeps))))/2)
            w_extra += logspace(frl+spread, frl, num=num,
                                endpoint=True).tolist()
            w_extra += logspace(frl, frl-spread, num=num,
                                endpoint=False).tolist()

        if isDiscrete:
            w_extra += logspace(ud+spread, ud, num=ppd,
                                endpoint=False).tolist()
        # Basis grid
        w = logspace(ld, ud, samples).tolist()
        w = np.sort(w + w_extra)
        # Remove accidental exact undamped mode hits from the tails of others
        w_out = w[np.in1d(w, nat_freq[damp_fact < sqeps], invert=True)]
        if ou == 'Hz':
            w_out /= 2*np.pi

    return w_out
