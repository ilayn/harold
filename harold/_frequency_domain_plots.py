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


from ._classes import State, Transfer
from ._frequency_domain import frequency_response


__all__ = ['bode_plot', 'nyquist_plot']


def bode_plot(G, w=None, use_db=False, use_radians=False):
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
    use_radians : bool
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
    pha = np.unwrap(np.angle(fre, deg=False if use_radians else True))

    if G._isSISO:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].semilogx(ww, mag)
        axs[1].semilogx(ww, pha)
        axs[1].set_xlabel(r'Frequency ({})'
                          ''.format('rad/s' if use_radians else 'Hz'))
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
                                        'rad/s' if use_radians else 'Hz'))
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
