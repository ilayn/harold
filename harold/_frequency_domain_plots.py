import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ._frequency_domain import frequency_response, _get_freq_grid
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['bode_plot', 'nyquist_plot']


def _check_sys_args(sys):
    """Check the arguments of frequency plotting funcs.

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


def _get_common_freq_grid(syslist):
    """Compute the common frequency grid for a given number of systems.

    The frequency_response function for a single system without a custom freq
    grid w decides some frequency interval to evaluate the response. Since for
    every system this might vary, this function externally call the freq grid
    generator and evens out the mismatches. As we know (if it does its job)
    that there is not much happening outside the generated frequencies hence
    we sprinkle some more data points to roughly get similar frequency ranges
    for better plotting.

    Unit is always rad/s in, rad/s out.

    The State, Transfer check is skipped since often it is done by the calling
    function

    Parameters
    ----------
    syslist : iterable
        The iterable that contains Transfer, State objects

    Returns
    -------
    w : ndarray
        Resulting common frequency points at which the responses are computed

    """
    # Points per decade to be added
    ppd = 15
    # Get the custom grids
    W = [_get_freq_grid(x, None, None, 'rad/s', 'rad/s') for x in syslist]
    # Frequencies are sorted by default
    minw = min([x[0] for x in W])
    maxw = max([x[-1] for x in W])
    lminw, lmaxw = np.log10(minw), np.log10(maxw)

    # now walk over all frequency grids and pad them to match the min and max
    for ind, w in enumerate(W):

        if minw < w[0]:
            lw0 = np.log10(w[0])
            samples = 5 if syslist[ind]._isgain else int(np.ceil((lw0 - lminw)
                                                                 * ppd))
            W[ind] = np.hstack([np.logspace(start=lminw,
                                            stop=lw0,
                                            num=samples),
                                w])

        # Skip discrete time systems as they will be naturally truncated at
        # Nyquist frequency.
        if maxw > w[-1] and syslist[ind].SamplingSet != 'Z':
            lw1 = np.log10(w[-1])
            samples = 5 if syslist[ind]._isgain else int(np.ceil((lmaxw - lw1)
                                                                 * ppd))
            W[ind] = np.hstack([w,
                                np.logspace(start=lw1,
                                            stop=lmaxw,
                                            num=samples)])

    return W


def bode_plot(sys, w=None, use_db=False, use_hz=True, use_degree=True,
              style=None, **kwargs):
    """Draw the Bode plot of State, Transfer model(s).

    As the name implies, this only creates a plot. For the actual frequency
    response data use the  `frequency_response()` which is also used
    internally.

    Parameters
    ----------
    sys : {State,Transfer}, iterable
        The system(s) for which the Bode plot will be drawn. For multiple
        plots, place the systems inside a list, tuples, etc. Generators will
        not work as they will be exhausted before the plotting is performed.
    w : array_like
        Range of frequencies. For discrete systems the frequencies above the
        nyquist frequency is ignored. If sys is an iterable w is used for
        all systems.
    use_db : bool, optional
        Uses the deciBell unit for the magnitude plots.
    use_hz : bool, optional
        Uses Hz unit for the output frequencies. This also assumes the input
        frequencies are in Hz.
    use_degree : bool, optional
        The phase angle is shown in degrees or in radians.
    style : cycler.Cycler, optional
        Matplotlib cycler class instance for defining the properties of the
        plot artists. If not given, the current defaults will be used.

    If any, all remaining kwargs are passed to `matplotlib.pyplot.subplots()`.

    Returns
    -------
    plot : matplotlib.figure.Figure

    Notes
    -----
    Every curve plotted in the Bode plot is labeled with the convention of
    ``sys_<number>_in_<number>_out_<number>_<mag or phase>`` where numbers are
    all 0-indexed.
    The first index is the model number given in the ``sys`` argument, and
    the other two are only nonzero when MIMO models are used. For discrete
    models the vertical lines are suffixed with ``_nyqline``.

    """
    syslist = _check_sys_args(sys)

    f_unit = 'Hz' if use_hz else 'rad/s'
    db_scale = 20 if use_db else 1

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

    # Create fig and axes
    fig, axs = plt.subplots(2*max_p, max_m, sharex=True, squeeze=False,
                            gridspec_kw=gridspec_kw,
                            figsize=figsize,
                            **kwargs)

    if style is None:
        style = mpl.cycler(mpl.rcParams['axes.prop_cycle'])

    # If multiple systems are given and no freq grid is given we need to find
    # the common freq grid.
    if w is None and len(syslist) > 1:
        W = _get_common_freq_grid(syslist)

    for ind, (G, sty) in enumerate(zip(syslist, style)):
        # If multiple systems are given W is necessarily defined here.
        if w is None and len(syslist) > 1:
            w = W[ind]
            if f_unit == 'Hz':
                w /= 2*np.pi

        isDiscrete = True if G.SamplingSet == 'Z' else False
        if isDiscrete:
            dt = G.SamplingPeriod
            nyq = 1/(2*dt) if use_hz else np.pi/dt

        if w is not None:
            fre, ww = frequency_response(G, w, w_unit=f_unit,
                                         output_unit=f_unit)
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

        p, m = G.shape

        for col in range(m):
            for row in range(p):
                # If SISO expand the arrays to 3D
                if mag.ndim == 1:
                    mag = mag[None, None, :]
                    pha = pha[None, None, :]
                label = f'sys_{ind}_in_{col}_out_{row}_mag'
                axs[2*row, col].semilogx(ww, mag[row, col, :],
                                         label=label,
                                         **sty)
                label = f'sys_{ind}_in_{col}_out_{row}_phase'
                axs[2*row+1, col].semilogx(ww, pha[row, col, :],
                                           label=label,
                                           **sty)

                if isDiscrete:
                    label = f'sys_{ind}_in_{col}_out_{row}_mag_nyqline'
                    axs[2*row, col].axvline(nyq, linestyle='dashed',
                                            linewidth=2,
                                            label=label,
                                            **sty)
                    label = f'sys_{ind}_in_{col}_out_{row}_phase_nyqline'
                    axs[2*row+1, col].axvline(nyq, linestyle='dashed',
                                              linewidth=2,
                                              label=label,
                                              **sty)

                axs[2*row, col].grid(True, which='both')
                axs[2*row+1, col].grid(True, which='both')

            # Only set the last item in that column to have x tick labels
            plt.setp(axs[2*row + 1, col].get_xticklabels(), visible=True)

    # Turn off the unused axes through a kind-of-ugly hack.
    # I can't find any other sane way to do this with current matplotlib
    if len(syslist) > 1:
        dummy_array = np.full([2*max_p, max_m], 0)
        for (x, y) in [x.shape for x in syslist]:
            dummy_array[:2*x, :y] = 1

        # Go through every column and move the x tick labels up before
        # hiding the empty frames - Why is matplotlib so hard?
        for col in range(max_m):
            nonempty_frames = np.count_nonzero(dummy_array[:, col])
            if nonempty_frames < 2*max_p:
                # Get the ticklabels
                axs[nonempty_frames-1, col].tick_params(axis='x',
                                                        labelbottom=True)
                for row in range(nonempty_frames, 2*max_p):
                    axs[row, col].set_visible(False)

    xlabel_text = f'Frequency [{f_unit}]'
    ylabel_text = 'Magnitude {} and Phase {}'.format(
            '[dB]' if use_db else r'[$\mathregular{10^x}$]',
            '[deg]' if use_degree else '[rad]')

    fig.text(0.5, 0, xlabel_text, ha='center')
    fig.text(0, 0.5, ylabel_text, va='center', rotation='vertical')
    fig.align_ylabels()

    return fig


def nyquist_plot(sys, w=None, use_hz=True, negative_freqs=True,
                 style=None, **kwargs):
    """Draw the Nyquist plot of State, Transfer model(s).

    Parameters
    ----------
    sys : {State,Transfer}, iterable
        The system(s) for which the Bode plot will be drawn. For multiple
        plots, place the systems inside a list, tuples, etc. Generators will
        not work as they will be exhausted before the plotting is performed.
    w : array_like
        Range of frequencies. For discrete systems the frequencies above the
        nyquist frequency is ignored. If sys is an iterable w is used for
        all systems.
    use_hz : bool, optional
        Uses Hz unit for the output frequencies. This also assumes the input
        frequencies are in Hz.
    negative_freqs : bool
        Draw or hide the negative frequencies. Default is True
    style : cycler.Cycler, optional
        Matplotlib cycler class instance for defining the properties of the
        plot artists. If not given, the current defaults will be used.

    If any, all remaining kwargs are passed to `matplotlib.pyplot.subplots()`.

    Returns
    -------
    plot : matplotlib.figure.Figure

    """
    syslist = _check_sys_args(sys)

    f_unit = 'Hz' if use_hz else 'rad/s'

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

    # Create fig and axes
    fig, axs = plt.subplots(max_p, max_m, sharex=True, squeeze=False,
                            gridspec_kw=gridspec_kw,
                            figsize=figsize,
                            **kwargs)

    if style is None:
        style = mpl.cycler(mpl.rcParams['axes.prop_cycle'])

    # If multiple systems are given and no freq grid is given we need to find
    # the common freq grid.
    if w is None and len(syslist) > 1:
        W = _get_common_freq_grid(syslist)

    for ind, (G, sty) in enumerate(zip(syslist, style)):
        # If multiple systems are given W is necessarily defined here.
        if w is None and len(syslist) > 1:
            w = W[ind]
            if f_unit == 'Hz':
                w /= 2*np.pi

        if w is not None:
            fre, ww = frequency_response(G, w, w_unit=f_unit,
                                         output_unit=f_unit)
        else:
            fre, ww = frequency_response(G, output_unit=f_unit)

        rr = fre.real
        ii = fre.imag

        p, m = G.shape

        for col in range(m):
            for row in range(p):
                # If SISO expand the arrays to 3D
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

    return fig
