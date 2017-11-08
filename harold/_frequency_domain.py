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
                   polyval, zeros, zeros_like, floor, ceil, unique, log10,
                   real, array)
from ._classes import State, Transfer, transfer_to_state, transmission_zeros
from ._system_funcs import staircase, _minimal_realization_state
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['frequency_response']


def frequency_response(G, w=None, samples=None, w_unit='Hz', output_unit='Hz'):
    """
    Computes the frequency response matrix of a State() or Transfer()
    object.

    The State representations are always checked for minimality and,
    if any, unobservable/uncontrollable modes are removed.

    Parameters
    ----------
    G: State of Transfer
        The realization for which the frequency response is computed
    custom_grid : array_like
        An array of sorted positive numbers denoting the frequencies
    high : float
        Power of 10 denoting the maximum frequency. If a discrete
        realization is given this is overridden by the Nyquist frequency.
    low : float
        Power of 10 denoting the minimum frequency.
    samples: int
        Number of samples to be created between `high` and `low`

    Returns
    -------
    freq_resp_array : Complex_valued numpy array
        The frequency response of the system G
    w : 1D numpy array
        Frequency grid that is used to evaluate the frequency response
    """
    ############################################################
    # TODO: Investigate if *args,*kwargs is a better syntax here for
    # better argument parsing.
    ############################################################

    _check_for_state_or_transfer(G)

    for x in (w_unit, output_unit):
        if x.lower() not in ('hz', 'rad/s'):
            raise ValueError('Frequency unit must be "Hz" or "rad/s". "{0}"'
                             ' is not recognized.'.format(x))

    if w is None:
        w = _get_freq_grid(G, w, samples, w_unit, output_unit)
    else:
        w = array(w, dtype=float, ndmin=1)

    # Convert to Hz if necessary
    if not w_unit == 'Hz':
        w = np.rad2deg(w)

    if G._isgain:
        if G._isSISO:
            if isinstance(G, Transfer):
                freq_resp_array = array([1]*2)*G.num[0, 0]
            else:
                freq_resp_array = array([1]*2)*G.d[0, 0]
        else:
            if isinstance(G, Transfer):
                freq_resp_array = zeros((2,)+G.shape) + array(G.num)
            else:
                freq_resp_array = zeros((2,)+G.shape) + array(G.d)

            freq_resp_array = rollaxis(freq_resp_array, 0, 3)

    elif G._isSISO:
        freq_resp_array = zeros_like(w, dtype='complex')

        if isinstance(G, State):
            abc = _minimal_realization_state(*G.matrices[:-1])
            aa, bb, cc = staircase(*abc, form='o', invert=True)
            freq_resp_array = _State_freq_resp(aa, bb, cc[0, -1], w)

            if np.any(G.d):
                freq_resp_array += G.d[0, 0]

        else:
            iw = w.flatten()*1j
            freq_resp_array = (polyval(G.num[0], iw) / polyval(G.den[0], iw))
    else:
        p, m = G.shape
        freq_resp_array = empty((len(w), m, p), dtype='complex')

        if isinstance(G, State):
            aa, bb, cc = _minimal_realization_state(*G.matrices[:-1])

            for rows in range(p):
                aaa, bbb, ccc = staircase(aa, bb, cc[[rows], :], form='o',
                                          invert=True)
                freq_resp_array[:, :, rows] = _State_freq_resp(aaa, bbb,
                                                               ccc[0, -1], w)

            if np.any(G.d):
                freq_resp_array += G.d[0, 0].T

            # Currently the shape is (freqs,cols,rows) for broadcasting.
            # roll axes in the correct places to have (row, col, freq) shape
            freq_resp_array = rollaxis(rollaxis(freq_resp_array, 0, 3), 1)

        else:
            iw = w.flatten()*1j
            freq_resp_array = empty((len(w), m, p), dtype='complex')
            for rows in range(p):
                for cols in range(m):
                    freq_resp_array[:, cols, rows] = (
                            polyval(G.num[rows][cols].flatten(), iw) /
                            polyval(G.den[rows][cols].flatten(), iw)
                            )
            freq_resp_array = rollaxis(rollaxis(freq_resp_array, 0, 3), 1)

    return freq_resp_array, w


def _State_freq_resp(mA, mb, sc, f):
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
    f  : array_like
        The frequency grid

    Returns
    -------
    r  : complex-valued numpy array

    """

    nn, m = mA.shape[0], mb.shape[1]
    r = empty((f.size, m), dtype=complex)
    Ab = block([-mA, mb]).astype(complex)
    X = empty_like(Ab)

    imag_indices = diag_indices(nn)

    for ind, val in enumerate(f):
        X[:, :] = Ab  # Working copy
        X[imag_indices] += val*1j
        for x in range(1, nn):
            X[x] -= (X[x, x-1] / X[x-1, x-1]) * X[x-1]

        r[ind, :] = X[-1, -m:]/X[-1, -1-m]

    return r*sc


def _get_freq_grid(G, w, samples, iu, ou):
    # internally always work with rad/s to comply with conventions(!).
    # Reconvert at the output if needed

    isDiscrete = G.SamplingSet == 'Z'
    dt = G.SamplingPeriod
    # Check the properties of the user-grid and regularize
    if w is not None:
        w_u = np.array(w, ndmin=1, dtype=float).ravel()
        # needs to be a 1D array
        if w_u.ndim > 1:
            raise ValueError('The frequency array should be a 1D float array')

        # Discrete time behavior doesn't make sense beyond Nyquist freq.
        if isDiscrete:
            nyq_freq = 1/(2*dt)
            if nyq_freq > np.max(w):
                w_out = w_u[w_u < nyq_freq]

            if w_out.size < 1:
                raise ValueError('There are no frequency points below the '
                                 'Nyquist frequency: {} Hz.'.format(nyq_freq))

        # convert the internal array to rad/s
        if iu == 'Hz':
            w_out *= 2*np.pi

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

        # Basis grid
        w = logspace(ld, ud, samples).tolist()
        w = np.sort(w + w_extra)
        # Remove accidental exact undamped mode hits from the tails of others
        w_out = w[~np.in1d(w, nat_freq[damp_fact < sqeps])]
        if ou == 'Hz':
            w_out /= 2*np.pi

    return w_out
