"""
The MIT License (MIT)

Copyright (c) 2017 Ilhan Polat

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

from ._frequency_domain import frequency_response
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['bode_plot', 'nyquist_plot']


def bode_plot(G, w=None, use_db=False, use_hz=True, use_degree=True):
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
    use_db : bool, optional
        Uses the deciBell unit for the magnitude plots.
    use_hz : bool, optional
        Uses Hz unit for the output frequencies. This also assumes the input
        frequencies are in Hz.
    use_degree : bool, optional
        The phase angle is shown in degrees or in radians.
    Returns
    -------
    plot : matplotlib.figure.Figure

    """
    _check_for_state_or_transfer(G)
    f_unit = 'Hz' if use_hz else 'rad/s'
    db_scale = 20 if use_db else 1
    isDiscrete = True if G.SamplingSet == 'Z' else False
    if isDiscrete:
        dt = G.SamplingPeriod
        nyq = 1/(2*dt) if use_hz else np.pi/dt

    if w is not None:
        fre, ww = frequency_response(G, w, w_unit=f_unit, output_unit=f_unit)
    else:
        fre, ww = frequency_response(G, output_unit=f_unit)

    fre[np.abs(fre) == 0.] = np.nan
    mag = db_scale * np.log10(np.abs(fre))

    # Mask NaN values if any
    if np.isnan(fre).any():
        pha = np.empty_like(fre, dtype=float)
        pha[~np.isnan(fre)] = np.unwrap(np.angle(fre[~np.isnan(fre)]))
    else:
        pha = np.unwrap(np.angle(fre))
    if use_degree:
        pha = np.rad2deg(pha)

    if G._isSISO:
        fig, axs = plt.subplots(2, 1, sharex=True, squeeze=False)
        axs[0, 0].semilogx(ww, mag)
        axs[1, 0].semilogx(ww, pha)
        if isDiscrete:
            axs[0, 0].axvline(nyq, linestyle='dashed', linewidth=2)
            axs[1, 0].axvline(nyq, linestyle='dashed', linewidth=2)
        axs[1, 0].set_xlabel(r'Frequency ({})'.format(f_unit))
        axs[0, 0].set_ylabel(r'Magnitude {}'.format('(dB)' if use_db else
                             r'($\mathregular{10^x}$)'))
        axs[1, 0].set_ylabel(r'Phase ({})'
                             ''.format('deg' if use_degree else 'rad'))
        for x in range(2):
            axs[x, 0].grid(True, which='both')

        fig.align_ylabels()
        return axs

    p, m = G.shape
    fig, axs = plt.subplots(2*p, m, sharex=True, squeeze=False)

    for col in range(m):
        for row in range(p):
            axs[2*row, col].semilogx(ww, mag[row, col, :])
            axs[2*row+1, col].semilogx(ww, pha[row, col, :])
            if isDiscrete:
                axs[2*row, col].axvline(nyq, linestyle='dashed', linewidth=2)
                axs[2*row+1, col].axvline(nyq, linestyle='dashed', linewidth=2)
            axs[2*row, col].grid(True, which='both')
            axs[2*row+1, col].grid(True, which='both')
            # MIMO Labels and gridding
            if col == 0:
                axs[2*row, col].set_ylabel(r'Magnitude {}'.format('(dB)'
                                           if use_db else
                                           r'($\mathregular{10^x}$)'))
                axs[2*row+1, col].set_ylabel(r'Phase (deg)')
            if row == p - 1:
                axs[2*row+1, col].set_xlabel(r'Frequency ({})'.format(f_unit))

    fig.align_ylabels()

    return axs


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
    _check_for_state_or_transfer(G)

    if w is not None:
        fre, ww = frequency_response(G, w)
    else:
        fre, ww = frequency_response(G)

    rr = fre.real
    ii = fre.imag

    p, m = G.shape
    fig, axs = plt.subplots(p, m, squeeze=False)

    for col in range(m):
        for row in range(p):
            if G._isSISO:
                rdata = rr
                idata = ii
            else:
                rdata = rr[row, col, :]
                idata = ii[row, col, :]

            axs[row, col].plot(rdata, idata, '-')
            axs[row, col].plot(rdata, -idata, '-.')
            axs[row, col].plot([-1], [0], 'b+')
            axs[row, col].grid(True, which='both')
            # MIMO Labels and gridding
            if col == 0:
                axs[row, col].set_ylabel('Imaginary Part')
            if row == p - 1:
                axs[row, col].set_xlabel('Real Part')
    return axs
