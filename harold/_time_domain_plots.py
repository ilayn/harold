import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ._arg_utils import _check_for_state_or_transfer
from ._time_domain import simulate_step_response, simulate_impulse_response

__all__ = ['step_response_plot', 'impulse_response_plot']


def _check_sys_args(sys):
    """Check the arguments of time domain plotting funcs.

    Parameters
    ----------
    sys : State, Transfer or iterable containing State or Transfer objects.

    Returns
    -------
    syslist : list
        Resulting system list including singletons.

    """
    try:
        _check_for_state_or_transfer(sys)
        syslist = [sys]
    except ValueError:
        # We probably have multiple systems, walk over it just to be sure
        for x in sys:
            _check_for_state_or_transfer(x)

        syslist = sys

    return syslist


def step_response_plot(sys, t=None, style=None, **kwargs):
    """
    Plots the step response of model(s). If the model is MIMO then the
    response is plotted as a subplot from input m to output p on a (p x m)
    plot matrix for each model.

    Parameters
    ----------
    sys : {State, Transfer}, iterable
        The system(s) for which the step response plots will be drawn. For
        multiple plots, place the systems inside a list, tuples, etc.
        Generators will not work as they will be exhausted before the plotting
        is performed.
    t : array_like, optional
        The 1D array that represents the time sequence. If sys is an iterable
        t is used for all systems.
    style : cycler.Cycler, optional
        Matplotlib cycler class instance for defining the properties of the
        plot artists. If not given, the current defaults will be used.

    If any, all remaining kwargs are passed to `matplotlib.pyplot.subplots()`.

    Returns
    -------
    fig : matplotlib.axes._subplots.AxesSubplot
        Returns the figure handle of the step response

    """
    syslist = _check_sys_args(sys)
    # Prepare the axes for the largest model
    max_p, max_m = np.max(np.array([x.shape for x in syslist]), axis=0)

    # Put some more space between columns to avoid ticklabel placement clashes
    gridspec_kw = kwargs.pop('gridspec_kw', None)
    if gridspec_kw is None:
        gridspec_kw = {'wspace': 0.5}
    else:
        wspace = gridspec_kw.get('wspace', 0.5)
        gridspec_kw['wspace'] = wspace

    # MIMO plots needs a bit bigger figure, offer a saner default
    figsize = kwargs.pop('figsize', None)
    if figsize is None:
        figsize = (6.0 + 1.2*(max_m - 1), 4 + 1.5*(max_p - 1))

    if style is None:
        style = mpl.cycler(mpl.rcParams['axes.prop_cycle'])

    # Create fig and axes
    fig, axs = plt.subplots(max_p, max_m,
                            sharex=True, sharey=True,
                            squeeze=False,
                            gridspec_kw=gridspec_kw,
                            figsize=figsize,
                            **kwargs)

    # TODO: If multiple systems are given and no t given we need to find a
    # compromise for the shorter ones. For now, just plot everything on top of
    # each other independently and cut to the shortest. Yes, it sucks; I know.
    tmin = np.inf
    for sys, sty in zip(syslist, style):
        yout, tout = simulate_step_response(sys, t=t)
        tmin = np.min([tout[-1], tmin])
        if sys._isSISO:
            if sys._isdiscrete:
                axs[0, 0].step(tout, yout, where='post', **sty)
            else:
                axs[0, 0].plot(tout, yout, **sty)

            axs[0, 0].grid(b=True)
        else:
            nrows, ncols = (yout.shape[1], 1) if yout.ndim == 2\
                else yout.shape[1:]

            # Get the appropriate plotter line plot or a step plot
            ptype = 'step' if sys._isdiscrete else 'plot'
            w_dict = {'where': 'post'} if sys._isdiscrete else {}
            for row in range(nrows):
                for col in range(ncols):
                    getattr(axs[row, col], ptype)(tout, yout[:, row, col]
                                                  if yout.ndim == 3
                                                  else yout[:, row],
                                                  **w_dict,
                                                  **sty)
                    axs[row, col].grid(b=True)

    axs[0, 0].set_xlim(left=0, right=tmin)
    fig.text(0, .5, 'Amplitude', ha='center', va='center', rotation='vertical')
    fig.text(.5, 0, 'Time', ha='center', va='center')
    fig.suptitle('Step response')

    return axs


def impulse_response_plot(sys, t=None, style=None, **kwargs):
    """
    Plots the impulse response of a model. If the system is MIMO then the
    response is plotted as a subplot from input m to output p on a (p x m)
    plot matrix.

    Parameters
    ----------
    sys : {State, Transfer}
        The system(s) for which the impulse response plots will be drawn. For
        multiple plots, place the systems inside a list, tuples, etc.
        Generators will not work as they will be exhausted before the plotting
        is performed.
    t : array_like, optional
        The 1D array that represents the time sequence. If sys is an iterable
        t is used for all systems.
    style : cycler.Cycler, optional
        Matplotlib cycler class instance for defining the properties of the
        plot artists. If not given, the current defaults will be used.

    If any, all remaining kwargs are passed to `matplotlib.pyplot.subplots()`.

    Returns
    -------
    fig : matplotlib.axes._subplots.AxesSubplot
        Returns the figure handle of the impulse response

    """
    syslist = _check_sys_args(sys)
    # Prepare the axes for the largest model
    max_p, max_m = np.max(np.array([x.shape for x in syslist]), axis=0)

    # Put some more space between columns to avoid ticklabel placement clashes
    gridspec_kw = kwargs.pop('gridspec_kw', None)
    if gridspec_kw is None:
        gridspec_kw = {'wspace': 0.5}
    else:
        wspace = gridspec_kw.get('wspace', 0.5)
        gridspec_kw['wspace'] = wspace

    # MIMO plots needs a bit bigger figure, offer a saner default
    figsize = kwargs.pop('figsize', None)
    if figsize is None:
        figsize = (6.0 + 1.2*(max_m - 1), 4 + 1.5*(max_p - 1))

    if style is None:
        style = mpl.cycler(mpl.rcParams['axes.prop_cycle'])

    # Create fig and axes
    fig, axs = plt.subplots(max_p, max_m, sharex=True, sharey=True,
                            squeeze=False,
                            gridspec_kw=gridspec_kw,
                            figsize=figsize,
                            **kwargs)

    # TODO: If multiple systems are given and no t given we need to find a
    # compromise for the shorter ones. For now, just plot everything on top of
    # each other independently and cut to the shortest. Yes, it sucks; I know.
    tmin = np.inf
    for sys in syslist:
        yout, tout = simulate_impulse_response(sys, t=t)
        tmin = np.min([tout[-1], tmin])
        if sys._isSISO:
            if sys._isdiscrete:
                axs[0, 0].step(tout, yout, where='post')
            else:
                axs[0, 0].plot(tout, yout)

            axs[0, 0].grid(b=True)
        else:
            nrows, ncols = (yout.shape[1], 1) if yout.ndim == 2 else\
                yout.shape[1:]

            # Get the appropriate plotter line plot or a step plot
            ptype = 'step' if sys._isdiscrete else 'plot'
            w_dict = {'where': 'post'} if sys._isdiscrete else {}
            for row in range(nrows):
                for col in range(ncols):
                    getattr(axs[row, col], ptype)(tout, yout[:, row, col]
                                                  if yout.ndim == 3
                                                  else yout[:, row], **w_dict)
                    axs[row, col].grid(b=True)
    axs[0, 0].set_xlim(left=0, right=tmin)
    fig.text(0, .5, 'Amplitude', ha='center', va='center', rotation='vertical')
    fig.text(.5, 0, 'Time', ha='center', va='center')
    fig.suptitle('Impulse response')

    return axs
