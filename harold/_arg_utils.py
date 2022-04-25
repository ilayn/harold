import numpy as np
from numpy import iscomplexobj, ndarray, asarray
from scipy.linalg import LinAlgWarning as _rcond_warn

__all__ = []


# These checks, purposefully, do not import State or Transfer objects to
# avoid cyclical imports happening everywhere in the library.


def _check_for_state(G, custom_msg=None):
    """
    A helper function to assert whether the given object is a State or
    Transfer.
    """
    if type(G).__name__ != 'State':
        if custom_msg is None:
            custom_msg = 'The argument should be a a State'\
                         '. Instead found a {}'.format(type(G).__qualname__)
        raise ValueError(custom_msg)


def _check_for_transfer(G, custom_msg=None):
    """
    A helper function to assert whether the given object is a State or
    Transfer.
    """
    if type(G).__name__ != 'Transfer':
        if custom_msg is None:
            custom_msg = 'The argument should be a Transfer'\
                         '. Instead found a {}'.format(type(G).__qualname__)
        raise ValueError(custom_msg)


def _check_for_state_or_transfer(G, custom_msg=None):
    """
    A helper function to assert whether the given object is a State or
    Transfer.
    """
    if type(G).__name__ not in ('State', 'Transfer'):
        if custom_msg is None:
            custom_msg = 'The argument should be a Transfer or a State'\
                         '. Instead found a {}'.format(type(G).__qualname__)
        raise ValueError(custom_msg)


def _check_equal_dts(G, H, custom_msg=None):
    """
    A helper function to assert whether the given objects have the same
    sampling periods.
    """
    if not G._dt == H._dt:
        if custom_msg is None:
            custom_msg = '''The sampling periods don\'t match.'''
        raise ValueError(custom_msg)


def _check_for_int_float_array(x, custom_msg=None):
    """
    A helper function to assert whether the given objects are int, float or
    ndarrays with proper data type. Also downcasts complex arrays with real
    data to reals, strips scalars, and make sure ndim is at least 2.
    """
    if isinstance(x, (int, float)):
        # Do nothing, return as is
        return x

    elif isinstance(x, complex):
        if x.imag == 0.:
            return x.real

    elif isinstance(x, ndarray):
        if iscomplexobj(x):
            if np.any(x.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')
            else:
                x = x.real
        return float(x) if x.size == 1 else x

    else:
        # Hopefully an array_like, barely try and repeat array checks
        xx = asarray(x, dtype=float)
        if iscomplexobj(xx):
            if np.any(xx.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')
            else:
                xx = xx.real
        return float(xx) if xx.size == 1 else xx

    if custom_msg is None:
        custom_msg = ' The argument should be a real scalar or a float '\
                     'array. Instead found {}'.format(type(x).__qualname__)
    raise ValueError(custom_msg)
