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
            samples = 5 if syslist[ind]._isgain else np.ceil((lw0 - lminw)*ppd)
            W[ind] = np.hstack([np.logspace(start=lminw,
                                            stop=lw0,
                                            num=samples),
                                w])

        # Skip discrete time systems as they will be naturally truncated at
        # Nyquist frequency.
        if maxw > w[-1] and syslist[ind].SamplingSet != 'Z':
            lw1 = np.log10(w[-1])
            samples = 5 if syslist[ind]._isgain else np.ceil((lmaxw - lw1)*ppd)
            W[ind] = np.hstack([w,
                                np.logspace(start=lw1,
                                            stop=lmaxw,
                                            num=samples)])

    return W


def bode_plot(sys, w=None, use_db=False, use_hz=True, use_degree=True,
              style=None):
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

    Returns
    -------
    plot : matplotlib.axes._subplots.AxesSubplot

    """
    syslist = _check_sys_args(sys)

    f_unit = 'Hz' if use_hz else 'rad/s'
    db_scale = 20 if use_db else 1

    # Prepare the axes for the largest model
    max_p, max_m = np.max(np.array([x.shape for x in syslist]), axis=0)
    fig, axs = plt.subplots(2*max_p, max_m, sharex=True, squeeze=False)

    # Turn off the unused axes through a kind-of-ugly hack.
    if len(syslist) > 1:
        dummy_array = np.full([2*max_p, max_m], np.nan)
        for (x, y) in [x.shape for x in syslist]:
            dummy_array[:2*x, :y] = 1

    for x, y in zip(*(np.isnan(dummy_array)).nonzero()):
        axs[x, y].set_axis_off()


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

                axs[2*row, col].semilogx(ww, mag[row, col, :], **sty)
                axs[2*row+1, col].semilogx(ww, pha[row, col, :], **sty)

                if isDiscrete:
                    axs[2*row, col].axvline(nyq, linestyle='dashed',
                                            linewidth=2, **sty)
                    axs[2*row+1, col].axvline(nyq, linestyle='dashed',
                                              linewidth=2, **sty)

                axs[2*row, col].grid(True, which='both')
                axs[2*row+1, col].grid(True, which='both')
                # MIMO Labels and gridding
                if col == 0:
                    axs[2*row, col].set_ylabel(r'Magnitude {}'.format(
                            '(dB)' if use_db else r'($\mathregular{10^x}$)'))
                    axs[2*row+1, col].set_ylabel(r'Phase ({})'.format(
                            'deg' if use_degree else 'rad'))
                if row == p - 1:
                    axs[2*row+1, col].set_xlabel(r'Frequency ({})'
                                                 ''.format(f_unit))

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
    plot : matplotlib.axes._subplots.AxesSubplot

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
