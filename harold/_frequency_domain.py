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
import matplotlib.pyplot as plt

from ._classes import State, Transfer
from ._system_funcs import staircase, minimal_realization

__all__ = ['frequency_response', 'bode_plot', 'nyquist_plot']


def _State_frequency_response_generator(mA, mb, sc, f):
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
    r = np.empty((f.size, m), dtype=complex)
    Ab = np.hstack((-mA, mb)).astype(complex)
    X = np.empty_like(Ab)

    imag_indices = np.diag_indices(nn)

    for ind, val in enumerate(f):
        X[:, :] = Ab  # Working copy
        X[imag_indices] += val*1j
        for x in range(1, nn):
            X[x] -= (X[x, x-1] / X[x-1, x-1]) * X[x-1]

        r[ind, :] = X[-1, -m:]/X[-1, -1-m]

    return r*sc


def frequency_response(G, custom_grid=None, high=None, low=None, samples=None,
                       custom_logspace=None,
                       input_freq_unit='Hz', output_freq_unit='Hz'):
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
    custom_logspace: 3-tuple


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

    if not isinstance(G, (State, Transfer)):
        raise ValueError('The argument should either be a State() or '
                         'Transfer() object. I have found a {0}'
                         ''.format(type(G).__qualname__))

    for x in (input_freq_unit, output_freq_unit):
        if x not in ('Hz', 'rad/s'):
            raise ValueError('I can only handle "Hz" and "rad/s" as '
                             'frequency units. "{0}" is not recognized.'
                             ''.format(x))

    _is_discrete = G.SamplingSet == 'Z'

    if _is_discrete:
        nyq_freq = 1/(2*G.SamplingPeriod)

    # We first check whether we need to bother if G is a gain
    # which overrides the user input for the grid in Hz. except
    # the output_freq_unit

    if G._isgain:
        samples = 2
        if _is_discrete:
            high = nyq_freq
            low = np.floor(np.log10(nyq_freq))
        else:
            high = 2
            low = -2
    else:
        pz_list = np.append(G.poles, G.zeros)

        if _is_discrete:
            nat_freq = np.abs(np.log(pz_list / G.SamplingPeriod))
        else:
            nat_freq = np.abs(pz_list)

        smallest_pz = np.max([np.min(nat_freq), 1e-7])
        largest_pz = np.max([np.max(nat_freq), smallest_pz+10])

    # The order of hierarchy is as follows:
    #  - We first check if a custom frequency grid is supplied
    #  - If None, then we check if a logspace-like option is given
    #  - If that's also None we check whether custom logspace
    #       limits are supplied with defaults for missing
    #           .. high    --> +2 decade from the fastest pole/zero
    #           .. low     --> -3 decade from the slowest pole/zero
    #           .. samples --> 1000 points

    # TODO: Implement a better/nonuniform algo for discovering new points
    # around  poles and zeros. Right now there is a chance to hit a pole
    # or a zero head on. matlab coarseness in practice leads to some
    # weirdness even when granularity = 1.

    if G._isgain:
        w = np.logspace(low, high, samples)
    elif custom_grid is None:
        if custom_logspace is None:
            high = np.ceil(np.log10(largest_pz)) + 1 if high is None else high
            low = np.floor(np.log10(smallest_pz)) - 1 if low is None else low
            samples = 1000 if samples is None else samples
        else:
            high, low, samples = custom_logspace
        w = np.logspace(low, high, samples)
    else:
        w = np.asarray(custom_grid, dtype='float')

    # Convert to Hz if necessary
    if not input_freq_unit == 'Hz':
        w = np.rad2deg(w)

    if G._isgain:
        if G._isSISO:
            if isinstance(G, Transfer):
                freq_resp_array = np.array([1]*2)*G.num[0, 0]
            else:
                freq_resp_array = np.array([1]*2)*G.d[0, 0]
        else:
            if isinstance(G, Transfer):
                freq_resp_array = np.zeros((2,)+G.shape) + np.array(G.num)
            else:
                freq_resp_array = np.zeros((2,)+G.shape) + np.array(G.d)

            freq_resp_array = np.rollaxis(freq_resp_array, 0, 3)

    elif G._isSISO:
        freq_resp_array = np.zeros_like(w, dtype='complex')

        if isinstance(G, State):
            aa, bb, cc = staircase(
                            *minimal_realization(*G.matrices[:-1]),
                            form='o',
                            invert=True
                            )
            freq_resp_array = _State_frequency_response_generator(
                                                        aa, bb, cc[0, -1], w)

            if np.any(G.d):
                freq_resp_array += G.d[0, 0]

        else:
            iw = w.flatten()*1j
            freq_resp_array = (np.polyval(G.num[0], iw) /
                               np.polyval(G.den[0], iw)
                               )
    else:
        p, m = G.shape
        freq_resp_array = np.empty((len(w), m, p), dtype='complex')

        if isinstance(G, State):
            aa, bb, cc = minimal_realization(*G.matrices[:-1])

            for rows in range(p):
                aaa, bbb, ccc = staircase(
                                          aa, bb, cc[[rows], :],
                                          form='o', invert=True
                                          )
                freq_resp_array[:,
                                :,
                                rows] = _State_frequency_response_generator(
                                                            aaa,
                                                            bbb,
                                                            ccc[0, -1],
                                                            w
                                                            )

            if np.any(G.d):
                freq_resp_array += G.d[0, 0].T

            # Currently the shape is (freqs,cols,rows) for broadcasting.
            # roll axes in the correct places to have (row, col, freq) shape
            freq_resp_array = np.rollaxis(
                                          np.rollaxis(freq_resp_array, 0, 3),
                                          1)

        else:
            iw = w.flatten()*1j
            freq_resp_array = np.empty((len(w), m, p), dtype='complex')
            for rows in range(p):
                for cols in range(m):
                    freq_resp_array[:, cols, rows] = (
                            np.polyval(G.num[rows][cols].flatten(), iw) /
                            np.polyval(G.den[rows][cols].flatten(), iw)
                            )
            freq_resp_array = np.rollaxis(
                                          np.rollaxis(freq_resp_array, 0, 3),
                                          1)

    return freq_resp_array, w


def bode_plot(G, w=None, use_db=False, use_radian=False):
    """
    Draws the Bode plot of the system G. As the name implies, this only
    creates a plot and for the data that is used `frequency_response()`
    should be used.

    Parameters
    ----------
    G : {State,Transfer}
        The system for which the Bode plot will be drawn
    w : array_like
        Range of frequencies
    use_db : bool
        Uses the deciBell unit for the magnitude plots.
    use_radian : bool
        Uses radians per second for the frequencies.

    Returns
    -------
    plot : matplotlib.figure.Figure

    """
    if not isinstance(G, (State, Transfer)):
        raise TypeError('The first argument should be a system'
                        ' representation.')
    db_scale = 20 if use_db else 1
    if w is not None:
        fre, ww = frequency_response(G, w)
    else:
        fre, ww = frequency_response(G)

    mag = db_scale * np.log10(np.abs(fre))
    pha = np.unwrap(np.angle(fre, deg=True if use_radian else False))

    if G._isSISO:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].semilogx(ww, mag)
        axs[1].semilogx(ww, pha)
        axs[1].set_xlabel(r'Frequency ({})'
                          ''.format('rad/s' if use_radian else 'Hz'))
        axs[0].set_ylabel(r'Magnitude{}'.format(' (dB)' if use_db else ''))
        axs[1].set_ylabel(r'Phase (deg)')
        for x in range(2):
            axs[x].grid(True, which='both')
        return fig

    p, m = G.shape
    fig, axs = plt.subplots(2*p, m, sharex=True)
    # For SIMO systems axs returns a 1D array hence double indices
    # lead to errors.
    if axs.ndim == 1:
        axs = axs[:, None]

    for col in range(m):
        for row in range(p):
            axs[2*row, col].semilogx(ww, mag[row, col, :])
            axs[2*row+1, col].semilogx(ww, pha[row, col, :])
            axs[2*row, col].grid(True, which='both')
            axs[2*row+1, col].grid(True, which='both')
            # MIMO Labels and gridding
            if col == 0:
                axs[2*row, col].set_ylabel(r'Magnitude'.format(
                                                ' (dB)' if use_db else ''))
                axs[2*row+1, col].set_ylabel(r'Phase (deg)')
            if row == p - 1:
                axs[2*row+1, col].set_xlabel(r'Frequency ({})'.format(
                                        'rad/s' if use_radian else 'Hz'))
    return fig


def nyquist_plot(G, w=None):
    """
    Draws the Nyquist plot of the system G.

    Parameters
    ----------
    G : {State,Transfer}
        The system for which the Nyquist plot will be drawn
    w : array_like
        Range of frequencies

    Returns
    -------
    plot : matplotlib.figure.Figure

    """
    if not isinstance(G, (State, Transfer)):
        raise TypeError('The first argument should be a system'
                        ' representation.')

    if w is not None:
        fre, ww = frequency_response(G, w)
    else:
        fre, ww = frequency_response(G)
        rr = np.real(fre)
        ii = np.imag(fre)

    if G._isSISO:
        plt.plot(rr, ii, '-')
        plt.plot(rr, -ii, '-.')
        # (-1,0) point
        plt.plot([-1], [0], 'b+')
        plt.xlabel(r'Real Part')
        plt.ylabel(r'Imaginary Part')
        plt.grid(which='both', axis='both')
        fig = plt.gcf()
        return fig

    p, m = G.shape
    fig, axs = plt.subplots(p, m)
    # For SIMO systems axs returns a 1D array hence double indices
    # lead to errors.
    if axs.ndim == 1:
            axs = axs[:, None] if m == 1 else axs[None, :]

    for col in range(m):
        for row in range(p):
            axs[row, col].plot(rr[row, col, :], ii[row, col, :], '-')
            axs[row, col].plot(rr[row, col, :], -ii[row, col, :], '-.')
            axs[row, col].plot([-1], [0], 'b+')
            axs[row, col].grid(True, which='both')
            # MIMO Labels and gridding
            if col == 0:
                axs[row, col].set_ylabel('Imaginary Part')
            if row == p - 1:
                axs[row, col].set_xlabel('Real Part')
    return fig
