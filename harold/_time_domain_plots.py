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
import matplotlib.pyplot as plt

from ._arg_utils import _check_for_state_or_transfer
from ._time_domain import simulate_step_response, simulate_impulse_response

__all__ = ['step_response_plot', 'impulse_response_plot']


def step_response_plot(sys, t=None):
    """
    Plots the step response of a model. If the system is MIMO then the
    response is plotted as a subplot from input m to output p on a (p x m)
    plot matrix.

    Parameters
    ----------
    sys : {State, Transfer}
        The system to be simulated
    t : array_like, optional
        The 1D array that represents the time sequence

    Returns
    -------
    fig : matplotlib.figure.Figure
        Returns the figure handle of the step response

    """
    _check_for_state_or_transfer(sys)
    yout, tout = simulate_step_response(sys, t=t)

    if sys._isSISO:
        fig, axs = plt.subplots(1, 1)
        if sys._isdiscrete:
            axs.step(tout, yout)
        else:
            axs.plot(tout, yout)

        axs.grid(b=True)
    else:
        nrows, ncols = (yout.shape[1], 1) if yout.ndim == 2 else yout.shape[1:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                                sharey=True, squeeze=0)

        # Get the appropriate plotter line plot or a step plot
        ptype = 'step' if sys._isdiscrete else 'plot'
        for row in range(nrows):
            for col in range(ncols):
                getattr(axs[row, col], ptype)(tout, yout[:, row, col]
                                              if yout.ndim == 3
                                              else yout[:, row])
                axs[row, col].grid(b=True)

    fig.text(0, .5, 'Amplitude', ha='center', va='center', rotation='vertical')
    fig.text(.5, 0, 'Time', ha='center', va='center')
    fig.suptitle('Step response')

    return fig


def impulse_response_plot(sys, t=None):
    """
    Plots the impulse response of a model. If the system is MIMO then the
    response is plotted as a subplot from input m to output p on a (p x m)
    plot matrix.

    Parameters
    ----------
    sys : {State, Transfer}
        The system to be simulated
    t : array_like, optional
        The 1D array that represents the time sequence

    Returns
    -------
    fig : matplotlib.figure.Figure
        Returns the figure handle of the impulse response

    """
    _check_for_state_or_transfer(sys)
    yout, tout = simulate_impulse_response(sys, t=t)

    if sys._isSISO:
        fig, axs = plt.subplots(1, 1)
        if sys._isdiscrete:
            axs.step(tout, yout)
        else:
            axs.plot(tout, yout)

        axs.grid(b=True)
    else:
        nrows, ncols = (yout.shape[1], 1) if yout.ndim == 2 else yout.shape[1:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                                sharey=True, squeeze=False)

        # Get the appropriate plotter line plot or a step plot
        ptype = 'step' if sys._isdiscrete else 'plot'
        for row in range(nrows):
            for col in range(ncols):
                getattr(axs[row, col], ptype)(tout, yout[:, row, col]
                                              if yout.ndim == 3
                                              else yout[:, row])
                axs[row, col].grid(b=True)

    fig.text(0, .5, 'Amplitude', ha='center', va='center', rotation='vertical')
    fig.text(.5, 0, 'Time', ha='center', va='center')
    fig.suptitle('Impulse response')

    return fig
