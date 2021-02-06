import numpy as np
import warnings
from numpy import zeros_like, kron, ndarray, zeros, exp, convolve, spacing
from numpy.random import rand, choice
from scipy.linalg import (eigvals, svdvals, block_diag, qz, norm, solve, expm,
                          inv, LinAlgError)
from scipy.linalg.decomp import _asarray_validated
from scipy.stats import ortho_group
from tabulate import tabulate
from itertools import zip_longest, chain

from ._polynomial_ops import (haroldpoly, haroldpolyadd, haroldpolydiv,
                              haroldpolymul, haroldcompanion, haroldlcm)
from ._array_validators import _assert_square
from ._aux_linalg import haroldsvd
from ._global_constants import _KnownDiscretizationMethods
from copy import deepcopy

__all__ = ['Transfer', 'State', 'state_to_transfer', 'transfer_to_state',
           'transmission_zeros', 'random_state_model',
           'concatenate_state_matrices']


class Transfer:
    """
    A class for creating Transfer functions.
    """

    def __init__(self, num, den=None, dt=None):
        """
        For SISO models, 1D lists or 1D numpy arrays are expected, e.g.,::

            >>> G = Transfer(1,[1,2,1])

        For MIMO systems, the array like objects are expected to be inside the
        appropriate shaped list of lists ::

            >>> G = Transfer([[ [1,3,2], [1,3] ],
            ...               [   [1]  , [1,0] ]],# end of num
            ...              [[ [1,2,1] ,  [1,3,3]  ],
            ...               [ [1,0,0] , [1,2,3,4] ]])

        If the denominator is common then the denominator can be given as a
        single array like object.

            >>> G = Transfer([[ [1,3,2], [1,3] ],
            ...               [   [1]  , [1,0] ]],# end of num
            ...              [1, 2, 3, 4, 5]) # common den

        Setting  ``SamplingPeriod`` property to ``'None'`` will make the
        system continuous time again and relevant properties are reset
        to continuous-time properties. However the numerical data will still
        be the same.
        """
        # Initialization Switch and Variable Defaults

        self._isgain = False
        self._isSISO = False
        self._isstable = False
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._dt = False
        (self._num, self._den,
         self._shape, self._isgain) = self.validate_arguments(num, den)
        self._p, self._m = self._shape
        if self._shape == (1, 1):
            self._isSISO = True
        self.SamplingPeriod = dt
        self._isdiscrete = False if dt is None else True

        self._recalc()

    @property
    def num(self):
        """
        If this property is called ``G.num`` then returns the numerator data.
        Alternatively, if this property is set then the provided value is
        first validated with the existing denominator shape and causality.
        """
        return self._num

    @property
    def den(self):
        """
        If this property is called ``G.den`` then returns the numerator data.
        Alternatively, if this property is set then the provided value is
        first validated with the existing numerator shape and causality.
        """
        return self._den

    @property
    def SamplingPeriod(self):
        """
        If this property is called ``G.SamplingPeriod`` then returns the
        sampling period data. If this property is set to ``False``, the model
        is assumed to be a continuous model. Otherwise, a discrete time model
        is assumed. Upon changing this value, relevant system properties are
        recalculated.
        """
        return self._dt

    @property
    def SamplingSet(self):
        """
        If this property is called ``G.SamplingSet`` then returns the
        set ``Z`` or ``R`` for discrete and continuous models respectively.
        This is a read only property and cannot be set. Instead an appropriate
        setting should be given to the ``SamplingPeriod`` property.
        """
        return self._rz

    @property
    def NumberOfInputs(self):
        """
        A read only property that holds the number of inputs.
        """
        return self._m

    @property
    def NumberOfOutputs(self):
        """
        A read only property that holds the number of outputs.
        """
        return self._p

    @property
    def shape(self):
        """
        A read only property that holds the shape of the system as a tuple
        such that the result is ``(# of outputs, # of inputs)``.
        """
        return self._shape

    @property
    def polynomials(self):
        """
        A read only property that returns the model numerator and the
        denominator as the outputs.
        """
        return self._num, self._den

    @property
    def DiscretizedWith(self):
        """
        This property is used internally to keep track of (if applicable)
        the original method used for discretization. It is used by the
        ``undiscretize()`` function to reach back to the continuous model that
        would hopefully minimize the discretization errors. It is also
        possible to manually set this property such that ``undiscretize``
        uses the provided method.
        """
        if self.SamplingSet == 'R' or self._DiscretizedWith is None:
            return None
        else:
            return self._DiscretizedWith

    @property
    def DiscretizationMatrix(self):
        """
        This 2x2 matrix denoted with ``q`` is used internally to represent
        the upper linear fractional transformation of the operation
        :math:`\\frac{1}{s} I = \\frac{1}{z} I \\star Q`.

        The available methods (and their aliases) can be accessed via the
        internal ``_KnownDiscretizationMethods`` variable.

        .. note:: The common discretization techniques can be selected with
            a keyword argument and this matrix business can safely be
            avoided. This is a rather technical issue and it is best to
            be used sparingly. For the experts, I have to note that
            the transformation is currently not tested for well-posedness.

        .. note:: SciPy actually uses a variant of this LFT
            representation as given in the paper of `Zhang et al.
            <http://dx.doi.org/10.1080/00207170802247728>`_

        """
        if self.SamplingSet == 'R' or not self.DiscretizedWith == 'lft':
            return None
        else:
            return self._DiscretizationMatrix

    @property
    def PrewarpFrequency(self):
        """
        If the discretization method is ``tustin`` then a frequency warping
        correction might be required the match of the discrete time system
        response at the frequency band of interest. Via this property, the
        prewarp frequency can be provided.
        """
        if self.SamplingSet == 'R' or self.DiscretizedWith \
                not in ('tustin', 'bilinear', 'trapezoidal'):
            return None
        else:
            return self._PrewarpFrequency

    @SamplingPeriod.setter
    def SamplingPeriod(self, value):
        if value is not None:
            value = float(value)
            if value <= 0.:
                raise ValueError('SamplingPeriod must be a real positive '
                                 'scalar. But looks like a \"{0}\" is '
                                 'given.'.format(type(value).__name__))

            self._dt = value
            self._rz = 'Z'
            self._isdiscrete = True
        else:
            self._rz = 'R'
            self._dt = None
            self._isdiscrete = False

    @num.setter
    def num(self, value):

        user_num, _, user_shape = self.validate_arguments(value, self._den)[:3]

        if not user_shape == self._shape:
            raise IndexError('Once created, the shape of the transfer '
                             'function \ncannot be changed. I have '
                             'received a numerator with shape {0}x{1} \nbut '
                             'the system has {2}x{3}.'
                             ''.format(*user_shape+self._shape))
        else:
            self._num = user_num
            self._recalc()

    @den.setter
    def den(self, value):

        user_den, user_shape = self.validate_arguments(self._num, value)[1:3]

        if not user_shape == self._shape:
            raise IndexError('Once created, the shape of the transfer '
                             'function \ncannot be changed. I have '
                             'received a denominator with shape {0}x{1} \nbut '
                             'the system has {2}x{3}.'
                             ''.format(*user_shape+self._shape))
        else:
            self._den = user_den
            self._recalc()

    @DiscretizedWith.setter
    def DiscretizedWith(self, value):
        if value in _KnownDiscretizationMethods:
            if self.SamplingSet == 'R':
                raise ValueError('This model is not discretized yet '
                                 'hence you cannot define a method for'
                                 ' it. Discretize the model first via '
                                 '"discretize" function.')
            else:
                self._DiscretizedWith = value
        else:
            raise ValueError('The {} method is unknown.'.format(value))

    @DiscretizationMatrix.setter
    def DiscretizationMatrix(self, value):
        if self._DiscretizedWith == 'lft':
            value = np.atleast_2d(np.asarray(value, dtype=float))
            if value.ndim > 2 or value.shape != (2, 2):
                raise ValueError('The interconnection array needs to be a'
                                 ' 2x2 real-valued array.')
            self._DiscretizationMatrix = value
        else:
            raise ValueError('If the discretization method is not '
                             '\"lft\" then this property is not set.')

    @PrewarpFrequency.setter
    def PrewarpFrequency(self, value):
        if self._DiscretizedWith not in ('tustin', 'bilinear', 'trapezoidal'):
            raise ValueError('If the discretization method is not tustin (or '
                             'its aliases) then this property is not set.')
        else:
            if value > 1/(2*self._dt):
                raise ValueError('Prewarping Frequency is beyond '
                                 'the Nyquist rate.\nIt has to '
                                 'satisfy 0 < w < 1/(2*dt) and dt '
                                 'being the sampling\nperiod in '
                                 'seconds (dt={0} is provided, '
                                 'hence the max\nallowed is '
                                 '{1} Hz.'.format(self._dt, 1/(2*self._dt)))
            else:
                self._PrewarpFrequency = value

    def _recalc(self):
        """
        Internal bookkeeping routine to readjust the class properties
        """
        if self._isgain:
            self.poles = np.array([])
            self.zeros = np.array([])
        else:
            if self._isSISO:
                self.poles = eigvals(haroldcompanion(self._den))
                if self._num.size == 1:
                    self.zeros = np.array([])
                else:
                    self.zeros = eigvals(haroldcompanion(self._num))
            else:
                # Create a dummy statespace and check the zeros there
                zzz = transfer_to_state((self._num, self._den),
                                        output='matrices')
                if zzz[0].size == 0:
                    # Oops, its static gain in disguise with exact cancellation
                    # reset numerator and copy poles to zeros
                    zzz = transfer_to_state(([[1]*self._m]*self._p,
                                             self._den),
                                            output='matrices')
                    self.poles = eigvals(zzz[0])
                    self.zeros = self.poles.copy()
                else:
                    self.zeros = transmission_zeros(*zzz)
                    self.poles = eigvals(zzz[0])

        self._set_stability()
        self._set_representation()

    def _set_stability(self):
        if self._rz == 'Z':
            self._isstable = all(1 > abs(self.poles))
        else:
            self._isstable = all(0 > np.real(self.poles))

    def _set_representation(self):
        self._repr_type = 'Transfer'

    # %% Transfer class arithmetic methods

    # Overwrite numpy array ufuncs
    __array_ufunc__ = None

    def __neg__(self):
        if not self._isSISO:
            newnum = [[None]*self._m for n in range(self._p)]
            for i in range(self._p):
                for j in range(self._m):
                    newnum[i][j] = -self._num[i][j]
        else:
            newnum = -1*self._num

        return Transfer(newnum, self._den, self._dt)

    def __add__(self, other):
        """
        Addition method

        Notice that in case SISO + MIMO, it is broadcasted to a ones matrix
        not an identity (Given a 3x3 system + 5) adds 5*np.ones([3,3]).
        """

        if isinstance(other, State):
            # We still follow the generic rule, State wins over Transfer
            # representations.

            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot add these models.')

            gainflag = sum([self._isgain, other._isgain])

            # If both are static gains
            if gainflag == 2:
                try:
                    return State(self.to_array()+other.to_array(), dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'addition. Model shapes are {0} and'
                                     ' {1}'.format(self._shape, other.shape))

            # If one of them is a static gain
            elif gainflag == 1:
                if self._isgain:
                    return other + self.to_array()
                else:
                    return transfer_to_state(self) + other.to_array()
            # No static gains, carry on
            else:
                pass

            sisoflag = sum([self._isSISO, other._isSISO])

            if sisoflag == 0 and self.shape != other.shape:
                raise ValueError('Shapes are not compatible for '
                                 'addition. Transfer shape is {0}'
                                 ' but the State shape is {1}.'
                                 ''.format(self._shape, other.shape))

            else:
                # In case one of them is SISO and will be broadcasted in the
                # next arrival
                return other + transfer_to_state(self)

        elif isinstance(other, Transfer):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems.')

            gainflag = sum([self._isgain, other._isgain])

            # If both are static gains
            if gainflag == 2:
                try:
                    return Transfer(self.to_array() + other.to_array(),
                                    dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'addition. Model shapes are {0} and'
                                     ' {1}'.format(self._shape, other.shape))

            else:
                pass

            sisoflag = sum([self._isSISO, other._isSISO])

            # Both SISO or both MIMO
            if sisoflag in [0, 2]:

                # Create empty num and den holders.
                newnum = [[None]*self._m for n in range(self._p)]
                newden = [[None]*self._m for n in range(self._p)]
                nonzero_num = np.zeros(self._shape, dtype=bool)

                # in case both are siso wrap the entries into list of lists
                if self._isSISO:
                    snum, sden = [[self._num]], [[self._den]]
                    onum, oden = [[other.num]], [[other.den]]
                else:
                    snum, sden = self._num, self._den
                    onum, oden = other.num, other.den

                # over all rows/cols, SISO is included with p, m = 1
                for row in range(self._p):
                    for col in range(self._m):
                        # in case the denominators are not monic
                        c0, c1 = sden[row][col][0, 0], oden[row][col][0, 0]

                        lcm, mults = haroldlcm(sden[row][col], oden[row][col])

                        newnum[row][col] = np.atleast_2d(haroldpolyadd(
                            convolve(snum[row][col].flatten(), mults[0]*c1),
                            convolve(onum[row][col].flatten(), mults[1]*c0)))

                        newden[row][col] = lcm*c0*c1

                    # Test whether we have at least one numerator entry
                    # that is nonzero. Otherwise return a zero MIMO tf
                        if np.count_nonzero(newnum[row][col]) != 0:
                            nonzero_num[row, col] = True

                # If SISO, unpack the list of lists
                if self._isSISO:
                    newnum, newden = newnum[0][0], newden[0][0]

                if any(nonzero_num.ravel()):
                    return Transfer(newnum, newden, dt=self._dt)
                else:
                    # Numerators all cancelled to zero hence 0-gain SISO/MIMO
                    return Transfer(np.zeros(self._shape).tolist(),
                                    dt=self._dt)

            # One of them is SISO and will be broadcasted here
            else:
                if self._isSISO:
                    return other +\
                        Transfer([[self._num]*other.m for n in range(other.p)],
                                 [[self._den]*other.m for n in range(other.p)],
                                 self._dt)
                else:
                    return self +\
                        Transfer([[other.num]*other.m for n in range(other.p)],
                                 [[other.den]*other.m for n in range(other.p)],
                                 self._dt)

        # Regularize arrays and scalars and consistency checks
        elif isinstance(other, (int, float, np.ndarray)):
            # Complex dtype does not immediately mean complex numbers,
            # check and forgive
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            if isinstance(other, np.ndarray):
                if other.ndim == 1:
                    if other.size == 1:
                        s = float(other)
                    else:
                        s = np.atleast_2d(other.real)
                else:
                    s = other.real

            else:
                s = float(other)

            # Self is,      # other is
            # isgain        1- scalar
            #               2- ndarray
            # isSISO        3- scalar
            #               4- ndarray
            # isMIMO        5- scalar
            #               6- ndarray
            if self._isgain:
                try:
                    # 1, 2
                    mat = self.to_array() + s
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'broadcasted addition. Transfer '
                                     'shape is {0} but the array shape is {1}.'
                                     ''.format(self._shape, other.shape))

                return Transfer(mat, dt=self._dt)

            elif self._isSISO:
                # 3
                if isinstance(s, float):
                    return self + Transfer(s, dt=self._dt)
                # 4
                else:
                    # Broadcast and send to MIMO TF + TF above
                    return (self * np.ones(s.shape)) + Transfer(s.tolist(),
                                                                dt=self._dt)
            else:
                # 5
                if isinstance(s, float):
                    return self + Transfer(np.ones(self.shape)*s, dt=self._dt)
                # 6
                if self.shape != other.shape:
                    raise ValueError('Shapes are not compatible for '
                                     'addition. Transfer shape is {0}'
                                     ' but the array shape is {1}.'
                                     ''.format(self._shape, other.shape))
                return self + Transfer(other.tolist(), dt=self._dt)
        else:
            raise ValueError('I don\'t know how to add a {0} to a '
                             'Transfer representation (yet).'
                             ''.format(type(other).__qualname__))

    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return -self + other

    def __mul__(self, other):
        # TODO: There are a few repeated code segments. Refactor!
        if isinstance(other, (int, float)):
            if self._isSISO:
                if other == 0.:
                    return Transfer(0, 1, dt=self.SamplingPeriod)
                else:
                    return Transfer(other*self._num,
                                    self._den,
                                    dt=self._dt)
            else:
                # Manually multiply each numerator
                t_p = self._p
                t_m = self._m

                newnum = [[None]*t_m for n in range(t_p)]
                newden = [[None]*t_m for n in range(t_p)]
                for row in range(t_p):
                    for col in range(t_m):
                        if other == 0.:
                            newnum[row][col] = np.array([[0.]])
                            newden[row][col] = np.array([[1.]])
                        else:
                            newnum[row][col] = other*self._num[row][col]
                            newden[row][col] = self._den[row][col]

                return Transfer(newnum, newden, dt=self._dt)

        elif isinstance(other, np.ndarray):
            # Complex dtype does not immediately mean complex numbers,
            # check and forgive
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            # It still might be a scalar inside an array
            if other.size == 1:
                return float(other) * self

            if other.ndim == 1:
                arr = np.atleast_2d(other.real)
            else:
                arr = other.real
            t_p, t_m = arr.shape
            newnum = [[None]*t_m for n in range(t_p)]
            newden = [[None]*t_m for n in range(t_p)]
            # if an array multiplied with SISO Transfer, elementwise multiply
            if self._isSISO:
                # Manually multiply numerator
                for row in range(t_p):
                    for col in range(t_m):
                        # If identically zero, empty out num/den
                        if arr[row, col] == 0.:
                            newnum[row][col] = np.array([[0.]])
                            newden[row][col] = np.array([[1.]])
                        else:
                            newnum[row][col] = arr[row, col]*self._num
                            newden[row][col] = self._den

                return Transfer(newnum, newden, dt=self._dt)

            # Reminder: This is elementwise multiplication not __matmul__!!
            elif self._shape == arr.shape:
                # Manually multiply each numerator
                for r in range(t_p):
                    for c in range(t_m):
                        # If identically zero, empty out num/den
                        if arr[r, c] == 0.:
                            newnum[r][c] = np.array([[0.]])
                            newden[r][c] = np.array([[1.]])
                        else:
                            newnum[r][c] = arr[r, c]*self._num[r][c]
                            newden[r][c] = self._den[r][c]

                return Transfer(newnum, newden, dt=self._dt)

            else:
                raise ValueError('Multiplication of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'
                                 ''.format(self._shape, other.shape))
        elif isinstance(other, State):
            # State representations win over the typecasting
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems. ')
            return other*transfer_to_state(self)

        elif isinstance(other, Transfer):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems.')

            # Get SISO and static gain out of the way
            # For gain, convert to ndarray and let previous case handle it
            if self._isgain:
                if self._isSISO:
                    return other * float(self._num)
                else:
                    # recast as a numpy array and multiply
                    # if denominator has non unity entries
                    # rescale numerator
                    mult_arr = np.empty((self._p, self._m))

                    for r in range(self._p):
                        for c in range(self._m):
                            mult_arr[r, c] = self._num[r][c] \
                                if self._den[r][c] == 1. else \
                                self._num[r][c]/self._den[r][c]

                    return other*mult_arr

            elif self._isSISO and other._isSISO:

                if not np.any(self._num) or not np.any(other.num):
                    return Transfer(0, 1, dt=self.SamplingPeriod)

                return Transfer(haroldpolymul(self._num, other.num),
                                haroldpolymul(self._den, other.den),
                                dt=self.SamplingPeriod)

            elif other._isSISO or self._isSISO:
                # Which one is MIMO
                snum = self._num if self._isSISO else other.num
                sden = self._den if self._isSISO else other.den
                mnum = other.num if self._isSISO else self._num
                mden = other.den if self._isSISO else self._den
                t_p, t_m = other.shape if self._isSISO else self._shape

                newnum = [[None]*t_m for n in range(t_p)]
                newden = [[None]*t_m for n in range(t_p)]

                for r in range(t_p):
                    for c in range(t_m):
                        if not np.any(snum) or not np.any(mnum[r][c]):
                            newnum[r][c] = np.array([[0.]])
                            newden[r][c] = np.array([[1.]])
                        else:
                            newnum[r][c] = haroldpolymul(snum, mnum[r][c])
                            newden[r][c] = haroldpolymul(sden, mden[r][c])
                return Transfer(newnum, newden, dt=self.SamplingPeriod)

            else:
                # Both MIMO
                if not self._shape == other.shape:
                    raise ValueError('Cannot multiply Transfer with {0} '
                                     ' shape with {1} with {2} shape.'
                                     ''.format(self._shape,
                                               type(other).__qualname__,
                                               other.shape)
                                     )

                t_p, t_m = self._shape

                newnum = [[None]*t_m for n in range(t_p)]
                newden = [[None]*t_m for n in range(t_p)]
                sn = self._num
                sd = self._den
                on = other.num
                od = other.den

                for r in range(t_p):
                    for c in range(t_m):
                        if not np.any(sn[r][c]) or not np.any(on[r][c]):
                            newnum[r][c] = np.array([[0.]])
                            newden[r][c] = np.array([[1.]])
                        else:
                            newnum[r][c] = haroldpolymul(sn[r][c], on[r][c])
                            newden[r][c] = haroldpolymul(sd[r][c], od[r][c])
                return Transfer(newnum, newden, dt=self.SamplingPeriod)
        else:
            raise ValueError('I don\'t know how to multiply a '
                             '{0} with a Transfer representation '
                             '(yet).'.format(type(other).__name__))

    def __rmul__(self, other):
        # *-multiplication means elementwise multiplication in Python
        # and order doesn't matter so pass it to mul, only because
        # I wrote that one first
        return self * other

    def __truediv__(self, other):
        """Support for G / ...

        """
        # For convenience of scaling the system via G/5 and so on.
        return self @ (1/other)

    def __rtruediv__(self, other):
        """ Support for division .../G

        """
        if self._isSISO:
            numdeg, dendeg = self.num.size, self.den.size
            if numdeg != dendeg:
                raise ValueError('Inverse of the system is noncausal which '
                                 'is not supported.')
            else:
                return other * Transfer(self.den, self.num, dt=self._dt)

        if not np.equal(*self._shape):
            raise ValueError('Nonsquare systems cannot be inverted')

        a, b, c, d = transfer_to_state((self._num, self._den),
                                       output='matrices')

        if np.any(svdvals(d) < np.spacing(1.)):
            raise LinAlgError('The feedthrough term of the system is not'
                              ' invertible.')
        else:
            # A-BD^{-1}C | BD^{-1}
            # -----------|--------
            # -D^{-1}C   | D^{-1}
            if self._isgain:
                ai, bi, ci = a, b, c
            else:
                ai = a - b @ solve(d, c)
                bi = (solve(d.T, b.T)).T
                ci = -solve(d, c)
            di = inv(d)

            num_inv, den_inv = state_to_transfer((ai, bi, ci, di),
                                                 output='polynomials')

            return other * Transfer(num_inv, den_inv, dt=self._dt)

    def __matmul__(self, other):
        # @-multiplication has the following rationale, first two items
        # are for user-convenience in case @ is used for *

        # 1. self is SISO --> whatever other is treat as *-mult -->  __mul__
        # 2. self is MIMO and other is SISO, same as item 1.
        # 3. self is MIMO and other is np.ndarray --> Matrix mult
        # 4. self is MIMO and other is MIMO --> Matrix mult

        # 1.
        if isinstance(other, (int, float)) or self._isSISO:
            return self * other

        # 3.
        if isinstance(other, (np.ndarray)):
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            # It still might be a scalar inside an array
            if other.size == 1:
                return self*float(other)

            if other.ndim == 1:
                arr = np.atleast_2d(other.real).astype(float)
            else:
                arr = other.real.astype(float)

            if not self._m == arr.shape[0]:
                raise ValueError('Size mismatch: Transfer representation '
                                 f'has {self._m} inputs but array has '
                                 f'{arr.shape[0]} rows.')

            # If self is gain, this is just matrix multiplication
            if self._isgain:
                return Transfer(self.to_array() @ arr, dt=self._dt)

            tp, tm = self._shape[0], arr.shape[1]

            newnum = [[None]*tm for n in range(tp)]
            newden = [[None]*tm for n in range(tp)]

            for r in range(tp):
                for c in range(tm):
                    t_G = Transfer(0, 1, dt=self._dt)
                    for ind in range(self._m):
                        t_G += self[r, ind] * other[ind, [c]]
                    newnum[r][c] = t_G.num
                    newden[r][c] = t_G.den

            if (tp, tm) == (1, 1):
                newnum = newnum[0][0]
                newden = newden[0][0]

            return Transfer(newnum, newden, dt=self.SamplingPeriod)

        # 4.
        if isinstance(other, (State, Transfer)):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems.')

            if isinstance(other, State):
                return transfer_to_state(self) @ State

            # 2.
            if other._isSISO:
                return self * other

            if self._shape[1] != other.shape[0]:
                raise ValueError(f'Size mismatch: Left Transfer '
                                 f'has {self._m} inputs but right Transfer '
                                 f'has {other.shape[0]} outputs.')

            tp, tm = self._shape[0], other.shape[1]

            # TODO : unoptimized and too careful
            # Take out the SIMO * MISO case resulting with SISO.
            if (tp, tm) == (1, 1):
                t_G = Transfer(0, 1, dt=self._dt)
                for ind in range(self._m):
                    t_G += self[0, ind] * other[ind, 0]
                return t_G
            else:
                newnum = [[None]*tm for n in range(tp)]
                newden = [[None]*tm for n in range(tp)]

                for r in range(tp):
                    for c in range(tm):
                        t_G = Transfer(0, 1, dt=self._dt)
                        for ind in range(self._m):
                            t_G += self[r, ind] * other[ind, c]

                        newnum[r][c] = t_G.num
                        newden[r][c] = t_G.den

            return Transfer(newnum, newden, dt=self._dt)

        else:
            raise ValueError('I don\'t know how to multiply a '
                             '{0} with a Transfer representation '
                             '(yet).'.format(type(other).__name__))

    def __rmatmul__(self, other):
        # If other is a State or Transfer, it will be handled
        # by other's __matmul__() method. Hence we only take care of the
        # right multiplication with the scalars and arrays. Otherwise
        # rejection is executed
        if isinstance(other, np.ndarray):
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            # It still might be a scalar inside an array
            if other.size == 1:
                return self*float(other)

            if other.ndim == 1:
                arr = np.atleast_2d(other.real)
            else:
                arr = other.real

            return Transfer(arr.tolist(), self._dt) @ self

        elif isinstance(other, (int, float)):
            return self * other
        else:
            raise ValueError('I don\'t know how to multiply a '
                             '{0} with a Transfer representation '
                             '(yet).'.format(type(other).__name__))

    def __getitem__(self, num_or_slice):

        # Check if a double subscript or not
        if isinstance(num_or_slice, tuple):
            rows_of_c, cols_of_b = num_or_slice
        else:
            rows_of_c, cols_of_b = num_or_slice, slice(None, None, None)
        # Eliminate all slices and colons but only indices
        rc = np.arange(self.NumberOfOutputs)[rows_of_c].tolist()
        cb = np.arange(self.NumberOfInputs)[cols_of_b].tolist()

        # if a SISO is sliced only [0,0] will pass, then return self
        if self._isSISO:
            return self

        # Is the result goint to be SISO ?
        if isinstance(rc, int) and isinstance(cb, int):
            return Transfer(self.num[rc][cb], self.den[rc][cb],
                            dt=self._dt)
        else:
            # Nope, release the MIMO bracket hell
            rc = [rc] if isinstance(rc, int) else rc
            cb = [cb] if isinstance(cb, int) else cb
            return Transfer([[self.num[x][y] for y in cb] for x in rc],
                            [[self.den[x][y] for y in cb] for x in rc],
                            dt=self._dt)

    def __setitem__(self, *args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                         'the relevant num, den attributes.')

    # ================================================================
    # __repr__ and __str__ to provide meaningful info about the system
    # The ascii art of matlab for tf won't be implemented.
    # Either proper image with proper superscripts or numbers.
    # ================================================================

    def __repr__(self):
        p, m = self.NumberOfOutputs, self.NumberOfInputs
        if self.SamplingSet == 'R':
            desc_text = 'Continuous-Time Transfer function\n'
        else:
            desc_text = ('Discrete-Time Transfer function with '
                         'sampling time: {0:.3f} ({1:.3f} Hz.)\n'
                         ''.format(float(self.SamplingPeriod),
                                   1/float(self.SamplingPeriod)))

        if self._isgain:
            desc_text += '\n{}x{} Static Gain\n'.format(p, m)
        else:
            desc_text += ' {0} input{2} and {1} output{3}\n'\
                         ''.format(m, p, 's' if m > 1 else '',
                                   's' if p > 1 else '')

            pole_zero_table = zip_longest(np.real(self.poles),
                                          np.imag(self.poles),
                                          np.real(self.zeros),
                                          np.imag(self.zeros)
                                          )

            desc_text += '\n' + tabulate(pole_zero_table,
                                         headers=['Poles(real)',
                                                  'Poles(imag)',
                                                  'Zeros(real)',
                                                  'Zeros(imag)']
                                         )

        desc_text += '\n\n'
        return desc_text

    def pole_properties(self, output_data=False):
        '''
        The resulting array holds the poles in the first column, natural
        frequencies in the second and damping ratios in the third. For
        static gain representations None is returned.

        # TODO : Will be implemented!!!
        The result is an array whose first column is the one of the complex
        pair or the real pole. When tabulated the complex pair is represented
        as "<num> ± <num>j" using single entry. However the data is kept as
        a valid complex number for convenience. If output_data is set to
        True the numerical values will be returned instead of the string
        type tabulars.
        '''
        return _pole_properties(self.poles,
                                self.SamplingPeriod,
                                output_data=output_data)

    def to_array(self):
        '''
        If a Transfer representation is a static gain, this method returns
        a regular 2D-ndarray.
        '''
        if self._isgain:
            if self._isSISO:
                return self._num/self._den
            else:
                num_arr = np.empty((self._p * self._m,))
                num_entries = sum(self._num, [])
                den_entries = sum(self._den, [])

                for x in range(self._p * self._m):
                    num_arr[x] = num_entries[x]
                    num_arr[x] /= den_entries[x]

                return num_arr.reshape(self._p, self._m)
        else:
            raise ValueError('Only static gain models can be converted to '
                             'ndarrays.')

    @staticmethod
    def validate_arguments(num, den, verbose=False):
        """

        A helper function to validate whether given arguments to an
        Transfer instance are valid and compatible for instantiation.

        Since there are many cases that might lead to a valid Transfer
        instance, Pythonic \"try,except\" machinery is not very helpful
        to check every possibility and equally challenging to branch
        off. A few examples of such issues that needs to be addressed
        is static gain, single entry for a MIMO system with common
        denominators and so on.

        Thus, this function provides a front-end to the laborious size
        and type checking which would make the Transfer object itself
        seemingly compatible with duck-typing while keeping the nasty
        branching implementation internal.

        The resulting output is compatible with the main harold
        Transfer class convention such that

          - If the recognized context is MIMO the resulting outputs are
            list of lists with numpy arrays being the polynomial
            coefficient entries.
          - If the recognized context is SISO the entries are numpy
            arrays with any list structure is stripped off.

        Parameters
        ----------

        num :
            The polynomial coefficient containers. Either of them
            can be (not both) None to assume that the context will
            be derived from the other for static gains. Otherwise
            both are expected to be one of np.array, int , float , list ,
            list of lists of lists or numpy arrays.

            For MIMO context, element numbers and causality
            checks are performed such that numerator list of
            list has internal arrays that have less than or
            equal to the internal arrays of the respective
            denominator entries.

            For SISO context, causality check is performed
            between numerator and denominator arrays.

        den :
            Same as num

        verbose : boolean
            A boolean switch to print out what this method thinks about the
            argument context.


        Returns
        -------

        num : List of lists or numpy array (MIMO/SISO)

        den : List of lists or numpy array (MIMO/SISO)

        shape : 2-tuple
            Returns the recognized shape of the system

        Gain_flag : Boolean
            Returns ``True`` if the system is recognized as a static gain
            ``False`` otherwise (for both SISO and MIMO)

        """
        def get_shape_from_arg(arg):
            """
            A static helper method to shorten the repeated if-else branch
            to get the shape of the system

            The functionality is to check the type of the argument and
            accordingly either count the rows/columns of a list of lists
            or get the shape of the numpy array depending on the the
            arguments type.

            Parameters
            ----------
            arg : {List of lists of numpy.array,numpy.array}
                  The argument should be compatible with a Transfer()
                  numerator or denominator/

            Returns
            ----------
            shape : tuple
                    Returns the identified system shape from the SISO/MIMO

            """
            if isinstance(arg, list):
                shape = (len(arg), len(arg[0]))
            else:
                shape = (1, 1)
            return shape

        # A list for storing the regularized entries for num and den
        returned_numden_list = [[], []]

        # Text shortcut for the error messages
        entrytext = ('numerator', 'denominator')

        # Booleans for Nones
        None_flags = [False, False]

        # Booleans for Gains
        Gain_flags = [False, False]

        # A boolean list that holds the recognized MIMO/SISO context
        # for the numerator and denominator respectively.
        # True --> MIMO, False --> SISO
        MIMO_flags = [False, False]

        for numden_index, numden in enumerate((num, den)):
            # Get the SISO/MIMO context for num and den.
            if verbose:
                print('='*40)
                print('Handling {0}'.format(entrytext[numden_index]))
                print('='*40)
            # If obviously static gain, don't bother with the rest
            if numden is None:
                if verbose:
                    print('I found None')
                None_flags[numden_index] = True
                Gain_flags[numden_index] = True
                continue

            # Start with MIMO possibilities first
            if isinstance(numden, list):
                if verbose:
                    print('I found a list')
                # OK, it is a list then is it a list of lists?
                if all([isinstance(x, list) for x in numden]):
                    if verbose:
                        print('I found a list that has only lists')

                    # number of columns in each row (m is a list)
                    m = [len(numden[ind]) for ind in range(len(numden))]
                    # number of rows (p is an integer)
                    p = len(numden)
                    if len(m) == 1 and m[0] == 1 and p == 1:
                        if verbose:
                            print('The list of lists actually contains '
                                  'a single element\nStripped off '
                                  'the lists and converted '
                                  'to a numpy array.')
                        returned_numden_list[numden_index] = np.atleast_2d(
                                                numden[0]).astype(float)
                        continue

                    # It is a list of lists so the context is MIMO
                    MIMO_flags[numden_index] = True

                    # Now try to regularize the entries to numpy arrays
                    # or complain explicitly

                    # Check if the number of elements are consistent
                    if max(m) == min(m):
                        if verbose:
                            print('Every row has consistent '
                                  'number of elements')
                        # Try to numpy-array the elements inside each row
                        try:
                            returned_numden_list[numden_index] = [
                                 [np.array(x, dtype='float', ndmin=2)
                                     for x in y]
                                 for y in numden]
                        except TypeError:
                            raise ValueError(  # something was not float
                                             'Something is not a \"float\" '
                                             'inside the MIMO {0} list of '
                                             'lists.'
                                             ''.format(entrytext[numden_index])
                                             )

                    else:
                        raise IndexError(
                                         'MIMO {0} lists have inconsistent\n'
                                         'number of entries, I\'ve found {1} '
                                         'element(s) in one row and {2} in '
                                         'another row.'
                                         ''.format(entrytext[numden_index],
                                                   max(m), min(m)))

                # We found the list and it wasn't a list of lists.
                # Then it should be a regular list to be np.array'd
                elif all([isinstance(x, (int, float)) for x in numden]):
                    if verbose:
                        print('I found a list that has only scalars')

                    if not any(numden):
                        if verbose:
                            print('The list was all zeros hence truncated '
                                  'to a single zero element.')
                        returned_numden_list[numden_index] = np.array([[0.]])
                    else:
                        returned_numden_list[numden_index] = \
                            np.atleast_2d(np.array(
                                    np.trim_zeros(numden, 'f'), dtype=float))
                    if numden_index == 1:
                        Gain_flags[1] = True
                else:
                    raise ValueError('Something is not a \"float\" inside '
                                     'the {0} list.'
                                     ''.format(entrytext[numden_index]))

            # Now we are sure that there is no dynamic MIMO entry.
            # The remaining possibility is a np.array as a static
            # gain for being MIMO. The rest is SISO.
            # Disclaimer: We hope that the data type is 'float'
            # Life is too short to check everything.

            elif isinstance(numden, np.ndarray):
                if verbose:
                    print('I found a numpy array')
                if numden.ndim > 1 and min(numden.shape) > 1:
                    if verbose:
                        print('The array has multiple elements')
                    returned_numden_list[numden_index] = [
                        [np.array([[x]], dtype='float') for x in y]
                        for y in numden.tolist()
                        ]
                    MIMO_flags[numden_index] = True
                    Gain_flags[numden_index] = True
                else:
                    returned_numden_list[numden_index] = np.atleast_2d(numden)

            # OK, finally check whether and int or float is given
            # as an entry of a SISO Transfer.
            elif isinstance(numden, (int, float)):
                if verbose:
                    print('I found only a float')
                returned_numden_list[numden_index] = np.atleast_2d(
                                                            float(numden))
                Gain_flags[numden_index] = True

            # Neither list of lists, nor lists nor int,floats
            # Reject and complain
            else:
                raise ValueError('{0} must either be a list of lists (MIMO)\n'
                                 'or a an unnested list (SISO). Numpy arrays, '
                                 'or, scalars inside unnested lists such as\n '
                                 '[3] are also accepted as SISO. '
                                 'See the \"Transfer\" docstring.'
                                 ''.format(entrytext[numden_index]))

        # =============================
        # End of the num, den for loop
        # =============================

        # Now we have regularized and also derived the context for
        # both numerator and the denominator. Finally a decision
        # can be made about the intention of the user.

        if verbose:
            print('='*50)
            print('Handling raw entries are done.\nNow checking'
                  ' the SISO/MIMO context and regularization.')
            print('='*50)
        # If both turned out to be MIMO!
        if all(MIMO_flags):
            if verbose:
                print('Both MIMO flags are true')
            # Since MIMO is flagged in both, we expect to have
            # list of lists in both entries.
            num_shape = (
                            len(returned_numden_list[0]),
                            len(returned_numden_list[0][0])
                        )

            den_shape = (
                            len(returned_numden_list[1]),
                            len(returned_numden_list[1][0])
                        )

            if num_shape == den_shape:
                shape = num_shape
            else:
                raise IndexError('I have a {0}x{1} shaped numerator and a '
                                 '{2}x{3} shaped \ndenominator. Hence I can '
                                 'not initialize this transfer \nfunction. '
                                 'I secretly blame you for this.'
                                 ''.format(*num_shape+den_shape)
                                 )

            # if all survived up to here, perform the causality check:
            # zip the num and den entries together and check their array
            # sizes and get the coordinates after trimming the zeros if any

            den_list = [np.trim_zeros(x[0], 'f') for x in
                        chain.from_iterable(returned_numden_list[1])]

            num_list = [np.trim_zeros(x[0], 'f') for x in
                        chain.from_iterable(returned_numden_list[0])]

            noncausal_flat_indices = [ind for ind, (x, y) in enumerate(
                                      zip(num_list, den_list))
                                      if x.size > y.size]

            noncausal_entries = [(x // shape[0], x % shape[1]) for x in
                                 noncausal_flat_indices]
            if not noncausal_entries == []:
                entry_str = ['Row {0}, Col {1}'.format(x[0], x[1]) for x in
                             noncausal_entries]

                raise ValueError('The following entries of numerator and '
                                 'denominator lead\nto noncausal transfers'
                                 '. Though I appreaciate the sophistication'
                                 '\nI don\'t touch descriptor stuff yet.'
                                 '\n{0}'.format('\n'.join(entry_str)))

        # If any of them turned out to be MIMO (ambiguous case)
        elif any(MIMO_flags):
            if verbose:
                print('One of the MIMO flags are true')
            # Possiblities are
            #  1- MIMO num, SISO den
            #  2- MIMO num, None den (gain matrix)
            #  3- SISO num, MIMO den
            #  4- None num, MIMO den

            # Get the MIMO flagged entry, 0-num,1-den
            MIMO_flagged = returned_numden_list[MIMO_flags.index(True)]

            # Case 3,4
            if MIMO_flags.index(True):
                if verbose:
                    print('Denominator is MIMO, Numerator is something else')
                # numerator None?
                if None_flags[0]:
                    if verbose:
                        print('Numerator is None')
                    # Then create a compatible sized ones matrix and
                    # convert it to a MIMO list of lists.

                    # Ones matrix converted to list of lists
                    num_ones = np.ones(
                                (len(MIMO_flagged), len(MIMO_flagged[0]))
                                ).tolist()

                    # Now make all entries 2D numpy arrays
                    # Since Num is None we can directly start adding
                    for row in num_ones:
                        returned_numden_list[0] += [
                                [np.atleast_2d(float(x)) for x in row]
                                ]

                # Numerator is SISO
                else:
                    if verbose:
                        print('Denominator is MIMO, Numerator is SISO')
                    # We have to check noncausal entries
                    # flatten den list of lists and compare the size
                    num_deg = np.trim_zeros(returned_numden_list[0][0],
                                            'f').size

                    flattened_den = sum(returned_numden_list[1], [])

                    noncausal_entries = [flattened_den[x].size < num_deg
                                         for x in range(len(flattened_den))]

                    if True in noncausal_entries:
                        raise ValueError('Given common numerator has '
                                         'a higher degree than some of '
                                         'the denominator entries hence '
                                         'defines noncausal transfer '
                                         'entries which is not allowed.')

                    den_shape = (
                                    len(returned_numden_list[1]),
                                    len(returned_numden_list[1][0])
                                )
                    # Now we know already the numerator is SISO so we copy
                    # it to each entry with a list of list that is compatible
                    # with the denominator shape. !!copy() is needed here.!!

                    # start an empty list and append rows/cols in it
                    kroneckered_num = np.empty((den_shape[0], 0)).tolist()

                    for x in range(den_shape[0]):
                        for y in range(den_shape[1]):
                            kroneckered_num[x].append(
                                    returned_numden_list[0].copy()
                                    )
                    returned_numden_list[0] = kroneckered_num

            # Case 1,2
            else:
                if verbose:
                    print('Numerator is MIMO, Denominator is something else')
                # denominator None?
                if None_flags[1]:
                    if verbose:
                        print('Numerator is a static gain matrix')
                        print('Denominator is None')

                    # This means num can only be a static gain matrix
                    flattened_num = sum(returned_numden_list[0], [])
                    noncausal_entries = [flattened_num[x].size < 2
                                         for x in range(len(flattened_num))]

                    nc_entry = -1
                    try:
                        nc_entry = noncausal_entries.index(False)
                    except ValueError:
                        Gain_flags = [True, True]

                    if nc_entry > -1:
                        raise ValueError('Since the denominator is not '
                                         'given, the numerator can only '
                                         'be a gain matrix such that '
                                         'when completed with a ones '
                                         'matrix as a denominator, there '
                                         'is no noncausal entries.')

                    # Then create a compatible sized ones matrix and
                    # convert it to a MIMO list of lists.
                    num_shape = (
                                 len(returned_numden_list[0]),
                                 len(returned_numden_list[0][0])
                                )

                    # Ones matrix converted to list of lists
                    den_ones = np.ones(num_shape).tolist()

                    # Now make all entries 2D numpy arrays
                    # Since Num is None we can directly start adding
                    for row in den_ones:
                        returned_numden_list[1] += [
                                [np.atleast_2d(float(x)) for x in row]
                                ]

                # Denominator is SISO
                else:
                    if verbose:
                        print('Numerator is MIMO, Denominator is SISO')
                    # We have to check noncausal entries
                    # flatten den list of lists and compare the size
                    den_deg = np.trim_zeros(
                            returned_numden_list[1][0], 'f').size

                    flattened_num = sum(returned_numden_list[0], [])

                    noncausal_entries = [flattened_num[x].size > den_deg
                                         for x in range(len(flattened_num))]

                    if True in noncausal_entries:
                        raise ValueError('Given common denominator has '
                                         'a lower degree than some of '
                                         'the numerator entries hence '
                                         'defines noncausal transfer '
                                         'entries which is not allowed.')

                    num_shape = (
                                    len(returned_numden_list[0]),
                                    len(returned_numden_list[0][0])
                                )

                    # Now we know already the denominator is SISO so we copy
                    # it to each entry with a list of list that is compatible
                    # with the numerator shape. !!copy() is needed here.!!

                    # start an empty list and append rows/cols in it
                    kroneckered_den = np.empty((num_shape[0], 0)).tolist()

                    for x in range(num_shape[0]):
                        for y in range(num_shape[1]):
                            kroneckered_den[x].append(
                                    returned_numden_list[1].copy()
                                    )
                    returned_numden_list[1] = kroneckered_den

        # Finally if both turned out be SISO !
        else:
            if verbose:
                print('Both are SISO')
            if any(None_flags):
                if verbose:
                    print('Something is None')
                if None_flags[0]:
                    if verbose:
                        print('Numerator is None')
                    returned_numden_list[0] = np.atleast_2d([1.0])
                else:
                    if verbose:
                        print('Denominator is None')
                    returned_numden_list[1] = np.atleast_2d([1.0])
                    Gain_flags = [True, True]

            if returned_numden_list[0].size > returned_numden_list[1].size:
                raise ValueError('Noncausal transfer functions are not '
                                 'allowed.')

        [num, den] = returned_numden_list

        shape = get_shape_from_arg(num)

        # Final gateway for the static gain
        if isinstance(den, list):
            # Check the max number of elements in each entry
            max_deg_of_den = max([x.size for x in sum(den, [])])
            # If less than two, then den is a gain matrix.
            Gain_flag = True if max_deg_of_den == 1 else False
            if verbose and Gain_flag:
                print('In the MIMO context and proper entries, I\'ve '
                      'found\nscalar denominator entries hence flagging '
                      'as a static gain.')
        else:
            Gain_flag = True if den.size == 1 else False
            if verbose:
                print('In the SISO context and a proper rational function'
                      ', I\'ve found\na scalar denominator hence '
                      'flagging as a static gain.')

        return num, den, shape, Gain_flag


class State:
    """
    A class for creating State space models.
    """

    def __init__(self, a, b=None, c=None, d=None, dt=None):
        """
        A State object can be instantiated in a straightforward manner by
        entering arraylikes.::

            >>> G = State([[0, 1], [-4, -5]], [[0], [1]], [1, 0], 1)

        For zero feedthrough (strictly proper) models, "d" matrix can be
        skipped and will be replaced with the zeros array whose shape is
        inferred from the rows/columns of "c"/"b" arrays.

        Setting  ``SamplingPeriod`` property to ``'None'`` will make the
        system continuous time again and relevant properties are reset
        to continuous-time properties. However the numerical data will still
        be the same.

        """
        self._dt = False
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._isSISO = False
        self._isgain = False
        self._isstable = False

        *abcd, self._shape, self._isgain = self.validate_arguments(a, b, c, d)

        self._a, self._b, self._c, self._d = abcd
        self._p, self._m = self._shape
        self._n = None if self._isgain else self._a.shape[0]

        if self._shape == (1, 1):
            self._isSISO = True

        self.SamplingPeriod = dt
        self._isdiscrete = False if dt is None else True
        self._recalc()

    @property
    def a(self):
        """
        If this property is called ``G.a`` then returns the matrix data.
        Alternatively, if this property is set then the provided value is
        first validated with the existing system shape and number of states.
        """
        return self._a

    @property
    def b(self):
        """
        If this property is called ``G.b`` then returns the matrix data.
        Alternatively, if this property is set then the provided value is
        first validated with the existing system shape and number of states.
        """
        return self._b

    @property
    def c(self):
        """
        If this property is called ``G.c`` then returns the matrix data.
        Alternatively, if this property is set then the provided value is
        first validated with the existing system shape and number of states.
        """
        return self._c

    @property
    def d(self):
        """
        If this property is called ``G.a`` then returns the matrix data.
        Alternatively, if this property is set then the provided value is
        first validated with the existing system shape.
        """
        return self._d

    @property
    def SamplingPeriod(self):
        """
        If this property is called ``G.SamplingPeriod`` then returns the
        sampling period data. If this property is set to ``False``, the model
        is assumed to be a continuous model. Otherwise, a discrete time model
        is assumed. Upon changing this value, relevant system properties are
        recalculated.
        """
        return self._dt

    @property
    def SamplingSet(self):
        """
        If this property is called ``G.SamplingSet`` then returns the
        set ``Z`` or ``R`` for discrete and continuous models respectively.
        This is a read only property and cannot be set. Instead an appropriate
        setting should be given to the ``SamplingPeriod`` property.
        """
        return self._rz

    @property
    def NumberOfStates(self):
        """
        A read only property that holds the number of states.
        """
        return self._a.shape[0]

    @property
    def NumberOfInputs(self):
        """
        A read only property that holds the number of inputs.
        """
        return self._m

    @property
    def NumberOfOutputs(self):
        """
        A read only property that holds the number of outputs.
        """
        return self._p

    @property
    def shape(self):
        """
        A read only property that holds the shape of the system as a tuple
        such that the result is ``(# of inputs , # of outputs)``.
        """
        return self._shape

    @property
    def matrices(self):
        """
        A read only property that returns the model matrices.
        """
        return self._a, self._b, self._c, self._d

    @property
    def DiscretizedWith(self):
        """
        This property is used internally to keep track of (if applicable)
        the original method used for discretization. It is used by the
        ``undiscretize()`` function to reach back to the continuous model that
        would hopefully minimize the discretization errors. It is also
        possible to manually set this property such that ``undiscretize``
        uses the provided method.
        """
        return self._DiscretizedWith

    @property
    def DiscretizationMatrix(self):
        """
        This matrix denoted with :math:`Q` is internally used to represent
        the upper linear fractional transformation of the operation
        :math:`\\frac{1}{s} I = \\frac{1}{z} I \\star Q`. For example, the
        typical tustin, forward/backward difference methods can be represented
        with

        .. math::

            Q = \\begin{bmatrix} I & \\sqrt{T}I \\\\ \\sqrt{T}I & \\alpha TI
            \\end{bmatrix}


        then for different :math:`\\alpha` values corresponds to the
        transformation given below:

            =============== ===========================
            :math:`\\alpha`  method
            =============== ===========================
            :math:`0`       backward difference (euler)
            :math:`0.5`     tustin
            :math:`1`       forward difference (euler)
            =============== ===========================

        This operation is usually given with a Riemann sum argument however
        for control theoretical purposes a proper mapping argument immediately
        suggests a more precise control over the domain the left half plane is
        mapped to. For this reason, a discretization matrix option is provided
        to the user.

        The available methods (and their aliases) can be accessed via the
        internal ``_KnownDiscretizationMethods`` variable.

        .. note:: The common discretization techniques can be selected with
            a keyword argument and this matrix business can safely be
            avoided. This is a rather technical issue and it is best to
            be used sparingly. For the experts, I have to note that
            the transformation is currently not tested for well-posedness.

        .. note:: SciPy actually uses a variant of this LFT
            representation as given in the paper of `Zhang et al.
            :doi:`10.1080/00207170802247728>`

        """
        return self._DiscretizationMatrix

    @property
    def PrewarpFrequency(self):
        """
        If the discretization method is ``tustin`` then a frequency warping
        correction might be required the match of the discrete time system
        response at the frequency band of interest. Via this property, the
        prewarp frequency can be provided.
        """
        if self.SamplingSet == 'R' or self.DiscretizedWith \
                not in ('tustin', 'bilinear', 'trapezoidal'):
            return None
        else:
            return self._PrewarpFrequency

    @a.setter
    def a(self, value):
        value = self.validate_arguments(
            value,
            np.zeros_like(self._b),
            np.zeros_like(self._c),
            np.zeros_like(self._d)
            )[0]
        self._a = value
        self._recalc()

    @b.setter
    def b(self, value):
        value = self.validate_arguments(
            np.zeros_like(self._a),
            value,
            np.zeros_like(self._c),
            np.zeros_like(self._d)
            )[1]
        self._b = value
        self._recalc()

    @c.setter
    def c(self, value):
        value = self.validate_arguments(
            np.zeros_like(self._a),
            np.zeros_like(self._b),
            value,
            np.zeros_like(self._d)
            )[2]
        self._c = value
        self._recalc()

    @d.setter
    def d(self, value):
        value = self.validate_arguments(
            np.zeros_like(self._a),
            np.zeros_like(self._b),
            np.zeros_like(self._c),
            value
            )[3]
        self._d = value
        self._recalc()

    @SamplingPeriod.setter
    def SamplingPeriod(self, value):
        if value is not None:
            value = float(value)
            if value <= 0.:
                raise ValueError('SamplingPeriod must be a real positive '
                                 'scalar. But looks like a \"{0}\" is '
                                 'given.'.format(type(value).__name__))

            self._dt = value
            self._rz = 'Z'
            self._isdiscrete = True
        else:
            self._rz = 'R'
            self._dt = None
            self._isdiscrete = False

    @DiscretizedWith.setter
    def DiscretizedWith(self, value):
        if value in _KnownDiscretizationMethods:
            if self.SamplingSet == 'R':
                raise ValueError('This model is not discretized yet '
                                 'hence you cannot define a method for'
                                 ' it. Discretize the model first via '
                                 '\"discretize\" function.')
            else:
                self._DiscretizedWith = value
        else:
            raise ValueError(f'{value} is not among the known methods:'
                             f'\n{_KnownDiscretizationMethods}')

    @DiscretizationMatrix.setter
    def DiscretizationMatrix(self, value):
        if self._DiscretizedWith == 'lft':
            self._DiscretizationMatrix = np.array(value, dtype='float')
        else:
            raise ValueError('If the discretization method is not '
                             '\"lft\" then you don\'t need to set '
                             'this property.')

    @PrewarpFrequency.setter
    def PrewarpFrequency(self, value):
        if self._DiscretizedWith not in ('tustin', 'bilinear', 'trapezoidal'):
            raise ValueError('If the discretization method is not '
                             'Tustin then you don\'t need to set '
                             'this property.')
        else:
            if value > 1/(2*self._dt):
                raise ValueError('Prewarping Frequency is beyond '
                                 'the Nyquist rate.\nIt has to '
                                 'satisfy 0 < w < 1/(2*dt) and dt '
                                 'being the sampling\nperiod in '
                                 'seconds (dt={0} is provided, '
                                 'hence the max\nallowed is '
                                 '{1} Hz.'.format(self._dt, 1/(2*self._dt))
                                 )
            else:
                self._PrewarpFrequency = value

    def _recalc(self):
        if self._isgain:
            self.poles = []
            self.zeros = []
        else:
            self.zeros = transmission_zeros(self._a, self._b, self._c, self._d)
            self.poles = eigvals(self._a)

        self._set_stability()
        self._set_representation()

    def _set_stability(self):
        if self._rz == 'Z':
            self._isstable = all(1 > np.abs(self.poles))
        else:
            self._isstable = all(0 > np.real(self.poles))

    def _set_representation(self):
        self._repr_type = 'State'

    # %% State class arithmetic methods

    # Overwrite numpy array ufuncs
    __array_ufunc__ = None

    def __neg__(self):
        if self._isgain:
            return State(-self._d, dt=self._dt)
        else:
            return State(self._a, self._b, -self._c, -self._d, self._dt)

    def __add__(self, other):
        """
        Addition method
        """

        if isinstance(other, State):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot add these models.')

            gainflag = sum([self._isgain, other._isgain])
            sisoflag = sum([self._isSISO, other._isSISO])

            # If both are static gains
            if gainflag == 2:
                try:
                    return State(self.d + other.d, dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'addition. Model shapes are {0} and'
                                     ' {1}'.format(self._shape, other.shape))
            elif gainflag == 1:
                # Convert the static to array
                if self._isgain:
                    return self.to_array() + other
                else:
                    return self + other.to_array()
            else:
                pass

            # If both are SISO or both MIMO, parallel connection
            if sisoflag == 2 or sisoflag == 0:
                if sisoflag == 0 and self.shape != other.shape:
                    raise ValueError('Shapes are not compatible for '
                                     'addtion. Model shapes are {0} and'
                                     ' {1}'.format(self._shape, other.shape))

                adda = block_diag(self._a, other.a)
                addb = np.vstack((self._b, other.b))
                addc = np.hstack((self._c, other.c))
                addd = self._d + other.d
                return State(adda, addb, addc, addd)
            # One of them is SISO and needs to be broadcasted
            else:
                if self._isSISO:
                    p, m = other.shape
                    return State(self.a, kron(np.ones(m), self.b),
                                 np.kron(np.ones(p)[:, None], self.c),
                                 np.ones([p, m])*self.d, dt=self._dt) + other
                else:
                    p, m = self.shape
                    return self + State(other.a, kron(np.ones(m), other.b),
                                        np.kron(np.ones(p)[:, None], other.c),
                                        np.ones([p, m])*other.d,
                                        dt=self._dt)

        elif isinstance(other, Transfer):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems.')

            gainflag = sum([self._isgain, other._isgain])

            # If both are static gains
            if gainflag == 2:
                try:
                    return State(self.to_array()+other.to_array(), dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'addition. Model shapes are {0} and'
                                     ' {1}'.format(self._shape, other.shape))

            # If one of them is a static gain
            elif gainflag == 1:
                if self._isgain:
                    return self.to_array() + other
                else:
                    return self + other.to_array()
            # No static gains, carry on
            else:
                pass

            sisoflag = sum([self._isSISO, other._isSISO])

            if sisoflag == 2 or sisoflag == 0:
                if sisoflag == 0 and self.shape != other.shape:
                    raise ValueError('Shapes are not compatible for '
                                     'addition. State shape is {0}'
                                     ' but the Transfer shape is {1}.'
                                     ''.format(self._shape, other.shape))

                return self + transfer_to_state(other)
            # One of them is SISO and will be broadcasted in the next arrival
            else:
                return self + transfer_to_state(other)

        # Regularize arrays and scalars and consistency checks
        elif isinstance(other, (int, float, np.ndarray)):
            # Complex dtype does not immediately mean complex numbers,
            # check and forgive
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            if isinstance(other, np.ndarray):
                if other.ndim == 1:
                    if other.size == 1:
                        s = float(other)
                    else:
                        s = np.atleast_2d(other.real)
                else:
                    s = other.real

            else:
                s = float(other)

            # isgain matmul 1- scalar
            #               2- ndarray
            # isSISO        3- scalar
            #               4- ndarray
            # isMIMO        5- scalar
            #               6- ndarray
            if self._isgain:
                try:
                    # 1, 2
                    mat = self.to_array() + s
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'addition (including broadcasting). '
                                     'State shape is {0}'
                                     ' but the array shape is {1}.'
                                     ''.format(self._shape, other.shape))

                return State(mat, dt=self._dt)

            elif self._isSISO:
                # 3
                if isinstance(s, float):
                    return State(self._a, self._b, self._c, self._d + s,
                                 dt=self._dt)
                # 4
                else:
                    # Broadcast and send to #6
                    p, m = s.shape
                    return State(self.a, kron(np.ones(m), self.b),
                                 kron(np.ones(p)[:, None], self.c),
                                 np.ones([p, m])*self.d, dt=self._dt) + s
            else:
                # 5, 6
                try:
                    return State(self._a, self._b, self._c, self._d + s,
                                 dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'addition (including broadcasting). '
                                     'State shape is {0}'
                                     ' but the array shape is {1}.'
                                     ''.format(self._shape, other.shape))
        else:
            raise ValueError('I don\'t know how to add a {0} to a '
                             'State representation (yet).'
                             ''.format(type(other).__qualname__))

    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return -self + other

    def __mul__(self, other):
        """
        Multiplication method
        """
        # Elementwise multiplication is removed. Redirect all to matmul
        return self @ other

    def __rmul__(self, other):
        """
        Left Multiplication method
        """
        return other @ self

    def __matmul__(self, other):
        """
        Multiplication method
        """

        if isinstance(other, State):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems.')

            gainflag = sum([self._isgain, other._isgain])

            # If both are static gains
            if gainflag == 2:
                try:
                    return State(self.to_array()@other.to_array(), dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'multiplication. Model shape is {0}'
                                     ' but the array shape is {1}.'
                                     ''.format(self._shape, other.shape))

            # If one of them is a static gain
            elif gainflag == 1:
                if self._isgain:
                    if self._isSISO:
                        return float(self.to_array()) * other
                    else:
                        return self.to_array() @ other
                else:
                    if other._isSISO:
                        return self * float(other.to_array())
                    else:
                        return self @ other.to_array()

            # No static gains, carry on
            else:
                pass

            sisoflag = sum([self._isSISO, other._isSISO])

            # If both are SISO or both MIMO, straightforward series connection
            if sisoflag == 2 or sisoflag == 0:
                if sisoflag == 0 and self._m != other._p:
                    raise ValueError('Shapes are not compatible for '
                                     'multiplication. Model shapes are {0} and'
                                     ' {1}'.format(self._shape, other.shape))

                multa = block_diag(self._a, other.a)
                multa[:self._n, self._n:] = self._b @ other.c
                multb = np.block([[self._b @ other.d], [other.b]])
                multc = np.block([self._c, self._d @ other.c])
                multd = self._d @ other.d
                return State(multa, multb, multc, multd, dt=self._dt)
            # One of them is SISO and needs to be broadcasted
            else:
                # Thanks to commutativity of SISO system we take the minimum
                # of the input and the output of the MIMO system
                if self._isSISO:
                    k = min(*other.shape)
                    if other.NumberOfInputs <= other.NumberOfOutputs:
                        return other @ (self @ np.eye(k))
                    else:
                        return (self @ np.eye(k)) @ other
                else:
                    k = min(self._p, self._m)
                    if self.NumberOfInputs <= self.NumberOfOutputs:
                        return self @ (other @ np.eye(k))
                    else:
                        return (other @ np.eye(k)) @ self

        elif isinstance(other, Transfer):
            if not self._dt == other._dt:
                raise ValueError('The sampling periods don\'t match '
                                 'so I cannot multiply these systems.')

            gainflag = sum([self._isgain, other._isgain])

            # If both are static gains
            if gainflag == 2:
                try:
                    return State(self.to_array()@other.to_array(), dt=self._dt)
                except ValueError:
                    raise ValueError('Shapes are not compatible for '
                                     'multiplication. State shape is {0}'
                                     ' but the Transfer shape is {1}.'
                                     ''.format(self._shape, other.shape))

            # If one of them is a static gain
            elif gainflag == 1:
                if self._isgain:
                    return self.to_array() @ other
                else:
                    return self @ other.to_array()

            # No static gains, carry on
            else:
                pass

            sisoflag = sum([self._isSISO, other._isSISO])

            if sisoflag == 2 or sisoflag == 0:
                if sisoflag == 0 and self._m != other._p:
                    raise ValueError('Shapes are not compatible for '
                                     'multiplication. State shape is {0}'
                                     ' but the Transfer shape is {1}.'
                                     ''.format(self._shape, other.shape))

                return self @ transfer_to_state(other)
            # One of them is SISO and needs to be broadcasted
            else:
                if self._isSISO:
                    k = min(*other.shape)
                    if other.NumberOfInputs <= other.NumberOfOutputs:
                        return transfer_to_state(other) @ (self @ np.eye(k))
                    else:
                        return (self @ np.eye(k)) @ transfer_to_state(other)
                else:
                    k = min(self._p, self._m)
                    if self.NumberOfInputs <= self.NumberOfOutputs:
                        return self @ (transfer_to_state(other) @ np.eye(k))
                    else:
                        return (transfer_to_state(other) @ np.eye(k)) @ self

        # Regularize arrays and scalars and consistency checks
        elif isinstance(other, (int, float, np.ndarray)):
            # Complex dtype does not immediately mean complex numbers,
            # check and forgive
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            if isinstance(other, np.ndarray):
                if other.ndim == 1:
                    if other.size == 1:
                        s = float(other)
                    else:
                        s = np.atleast_2d(other.real)
                else:
                    s = other.real

                # Early shape check
                if self._shape[1] != other.shape[0] and not self._isSISO:
                    raise ValueError('Shapes are not compatible for '
                                     'multiplication. State shape is {0}'
                                     ' but the array shape is {1}.'
                                     ''.format(self._shape, other.shape))

            else:
                s = float(other)

            # isgain matmul 1- scalar
            #               2- ndarray
            # isSISO        3- scalar
            #               4- ndarray
            # isMIMO        5- scalar
            #               6- ndarray
            if self._isgain:
                # 1, 2
                try:
                    # 2
                    mat = self.to_array() @ s
                except ValueError:
                    # 1
                    mat = self.to_array * s

                return State(mat, dt=self._dt)

            elif self._isSISO:
                # 3
                if isinstance(s, float):
                    if s == 0.:
                        return State(0., dt=self._dt)
                    else:
                        return State(self._a, self._b * s, self._c,
                                     self._d * s, dt=self._dt)
                # 4
                else:
                    # if all zero then return 0. static gain
                    if not np.any(s):
                        return State(zeros_like(s), dt=self._dt)

                    p, m = s.shape
                    ba = block_diag(*[self._a]*min(p, m))
                    if p > m:
                        bb = block_diag(*[self._b]*min(p, m))
                        bc = kron(s, self._c)
                    else:
                        bb = kron(s, self._b)
                        bc = block_diag(*[self._c]*min(p, m))

                    bd = s * self._d
                    return State(ba, bb, bc, bd, dt=self._dt)
            else:
                # 5
                if isinstance(s, float):
                    return State(self._a, self._b * s, self._c, self._d * s,
                                 dt=self._dt)
                # 6
                else:
                    return State(self._a, self._b @ s, self._c, self._d @ s,
                                 dt=self._dt)
        else:
            raise ValueError('I don\'t know how to multiply a {0} with a '
                             'State representation (yet).'
                             ''.format(type(other).__qualname__))

    def __rmatmul__(self, other):
        # isgain rmatmul 1- scalar
        #                2- ndarray

        # state rmatmul  3- scalar
        #                4- ndarray

        # Regularize arrays and scalars and consistency checks
        if isinstance(other, (int, float, np.ndarray)):
            # Complex dtype does not immediately mean complex numbers,
            # check and forgive
            if np.iscomplexobj(other) and np.any(other.imag):
                raise ValueError('Complex valued representations are not '
                                 'supported.')

            if isinstance(other, np.ndarray):
                if other.ndim == 1:
                    if other.size == 1:
                        s = float(other)
                    else:
                        s = np.atleast_2d(other.real)
                else:
                    s = other.real

                # Early shape check
                if self._shape[0] != other.shape[1] and not self._isSISO:
                    raise ValueError('Shapes are not compatible for '
                                     'multiplication. Model shape is {0} but'
                                     ' the array shape is {1}.'
                                     ''.format(self._shape, other.shape))

            else:
                s = float(other)

            # isgain matmul 1- scalar
            #               2- ndarray
            # isSISO        3- scalar
            #               4- ndarray
            # isMIMO        5- scalar
            #               6- ndarray
            if self._isgain:
                # 1, 2
                try:
                    # 2
                    mat = s @ self.to_array()
                except ValueError:
                    # 1
                    mat = self.to_array() * s
                return State(mat, dt=self._dt)

            elif self._isSISO:
                # 3
                if isinstance(s, float):
                    if s == 0.:
                        return State(0., dt=self._dt)
                    else:
                        return State(self._a, self._b, self._c * s,
                                     self._d * s, dt=self._dt)
                # 4
                else:
                    # if all zero then return 0. static gain
                    if not np.any(s):
                        return State(zeros_like(s), dt=self._dt)

                    p, m = s.shape
                    ba = block_diag(*[self._a]*min(p, m))
                    if p > m:
                        bb = block_diag(*[self._b]*min(p, m))
                        bc = kron(s, self._c)
                    else:
                        bb = kron(s, self._b)
                        bc = block_diag(*[self._c]*min(p, m))

                    bd = s * self._d
                    return State(ba, bb, bc, bd, dt=self._dt)
            else:
                # 5
                if isinstance(s, float):
                    return State(self._a, self._b, self._c * s, self._d * s,
                                 dt=self._dt)
                # 6
                else:
                    return State(self._a, self._b, s @ self._c, s @ self._d,
                                 dt=self._dt)
        else:
            raise ValueError('I don\'t know how to multiply a {0} with a '
                             'state representation (yet).'
                             ''.format(type(other).__qualname__))

    def __truediv__(self, other):
        """ Support for division G/...

        """
        return self @ (1/other)

    def __rtruediv__(self, other):
        """ Support for division .../G

        """
        if not np.equal(*self._shape):
            raise ValueError('Nonsquare systems cannot be inverted')

        a, b, c, d = self._a, self._b, self._c, self._d

        if np.any(svdvals(d) < np.spacing(1.)):
            raise LinAlgError('The feedthrough term of the system is not'
                              ' invertible.')
        else:
            # A-BD^{-1}C | BD^{-1}
            # -----------|--------
            # -D^{-1}C   | D^{-1}
            if self._isgain:
                ai, bi, ci = None, None, None
            else:
                ai = a - b @ solve(d, c)
                bi = (solve(d.T, b.T)).T
                ci = -solve(d, c)
            di = inv(d)

            return other @ State(ai, bi, ci, di, dt=self._dt)

    def __getitem__(self, num_or_slice):

        # Check if a double subscript or not
        if isinstance(num_or_slice, tuple):
            rows_of_c, cols_of_b = num_or_slice
        else:
            rows_of_c, cols_of_b = num_or_slice, slice(None, None, None)

        # Handle the ndim losing behavior of NumPy indexing
        rc = np.atleast_2d(np.arange(self.NumberOfOutputs)[rows_of_c])
        cb = np.arange(self.NumberOfInputs)[cols_of_b]
        n = np.arange(self.NumberOfStates)

        if rc.size == 1:
            rc = np.squeeze(rc).tolist()
        # Transpose for braadcasting
        elif rc.size > 1:
            rc = rc.T

        if cb.size == 1:
            cb = np.squeeze(cb).tolist()

        if self._isgain:
            return State(self.d[rc, cb], dt=self._dt)

        # Enforce fancyness, avoid mixing. Why do we even have to do this?
        btemp = self.b[n[:, None], cb]
        ctemp = self.c[rc, n]

        return State(self.a,
                     btemp if btemp.ndim > 1 else btemp.reshape(rc, cb),
                     ctemp,
                     self.d[rc, cb],
                     dt=self._dt)

    def __setitem__(self, *args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                         'the relevant A,B,C,D attributes.')

    def __repr__(self):
        p, m, n = self._p, self._m, self.NumberOfStates
        if self._rz == 'R':
            desc_text = '\nContinuous-time state representation\n'
        else:
            desc_text = ('\nDiscrete-Time state representation with '
                         'sampling time: {0:.3f} ({1:.3f} Hz.)\n'
                         ''.format(float(self.SamplingPeriod),
                                   1/float(self.SamplingPeriod)))

        if self._isgain:
            desc_text += '\n{}x{} Static Gain\n'.format(p, m)
        else:
            desc_text += '{0} state{3}, {1} input{4}, and {2} output{5}'\
                         ''.format(n, m, p,
                                   's' if n > 1 else '',
                                   's' if m > 1 else '',
                                   's' if p > 1 else ''
                                   )

            pole_zero_table = zip_longest(np.real(self.poles),
                                          np.imag(self.poles),
                                          np.real(self.zeros),
                                          np.imag(self.zeros)
                                          )

            desc_text += '\n' + tabulate(pole_zero_table,
                                         headers=['Poles(real)',
                                                  'Poles(imag)',
                                                  'Zeros(real)',
                                                  'Zeros(imag)']
                                         )

        desc_text += '\n\n'
        return desc_text

    def pole_properties(self, output_data=False):
        return _pole_properties(self.poles,
                                self.SamplingPeriod,
                                output_data=output_data)
    pole_properties.__doc__ = Transfer.pole_properties.__doc__

    def to_array(self):
        '''
        If a State representation is a static gain, this method returns
        a regular 2D-ndarray.
        '''
        if self._isgain:
            return self._d
        else:
            raise ValueError('Only static gain models can be converted to '
                             'ndarrays.')

    @staticmethod
    def validate_arguments(a, b, c, d, verbose=False):
        """

        An internal command to validate whether given arguments to a
        State() instance are valid and compatible.

        It also checks if the lists are 2D numpy.array'able entries.

        """

        # A list for storing the regularized entries for a,b,c,d (mutable)
        returned_abcd_list = [[], [], [], []]

        # Text shortcut for the error messages
        entrytext = ('A', 'B', 'C', 'D')

        # Booleans for Nones
        None_flags = [False, False, False, False]

        Gain_flag = False

        # Compared to the Transfer() inputs, State() can have relatively
        # saner inputs which is one of the following types, hence the var
        possible_types = (int,
                          np.int32,
                          np.int64,
                          float,
                          list,
                          ndarray,
                          )

        # Start regularizing the input regardless of the intention
        for abcd_index, abcd in enumerate((a, b, c, d)):
            if verbose:
                print('='*40)
                print('Handling {0}'.format(entrytext[abcd_index]))
                print('='*40)
            # User supplied it? if no then don't bother further parsing.
            if abcd is None:
                if verbose:
                    print('{0} is None'.format(entrytext[abcd_index]))
                returned_abcd_list[abcd_index] = np.empty((0, 0))
                None_flags[abcd_index] = True
                continue

            # Check for obvious choices
            if not isinstance(abcd, possible_types):
                raise ValueError('{0} matrix should be, regardless of the'
                                 ' shape, an int, float, list or,\n'
                                 'much better, a properly typed 2D Numpy '
                                 'array. Instead I found a {1} object.'
                                 ''.format(entrytext[abcd_index],
                                           type(abcd).__qualname__))

            else:
                # Row/column consistency is checked by numpy
                try:
                    if verbose:
                        print('Trying to np.array {0}'
                              ''.format(entrytext[abcd_index]))

                    returned_abcd_list[abcd_index] = np.atleast_2d(
                                                np.array(abcd, dtype='float')
                                                )
                except ValueError:
                    raise ValueError('The {0} matrix argument couldn\'t '
                                     'be converted to a 2D array of real'
                                     ' numbers.'
                                     ''.format(entrytext[abcd_index])
                                     )

        # If State() has a single nonzero argument then this is a gain
        # so flip the list and make d nonzero let the rest empty matrix.
        if all(None_flags[1:]):
            if verbose:
                print('Only A matrix is given in the'
                      ' A,B,C,D arguments. Hence I decided'
                      ' that this is a static gain')
            returned_abcd_list = list(reversed(returned_abcd_list))
            Gain_flag = True

        # Or the nonzero argument is given (None,None,None,D) format
        # hence pass with no modification
        elif all(None_flags[:-1]):
            if verbose:
                print('I decided that this is a gain')
            Gain_flag = True

        [a, b, c, d] = returned_abcd_list

        if not Gain_flag:
            # Here check everything is compatible unless we have a
            # static gain
            if verbose:
                print('All seems OK. Moving to shape mismatch check')
            if not a.shape == a.T.shape:
                raise ValueError('A matrix must be a square matrix '
                                 'but I got {0}'.format(a.shape))

            if b.shape[0] != a.shape[0]:
                # Accept annoying 1D inputs for B matrices
                if b.shape[0] == 1 and b.shape[1] == a.shape[0]:
                    if verbose:
                        print('It looks like B was a 1D input hence '
                              'I made it a column vector.')
                    b = b.T.copy()
                else:
                    raise ValueError('B matrix must have the same number '
                                     'of rows with A matrix. I need {:d} '
                                     'but got {:d}.'
                                     ''.format(a.shape[0], b.shape[0]))

            if c.shape[1] != a.shape[1]:
                raise ValueError('C matrix must have the same number of '
                                 'columns with A matrix.\nI need {:d} '
                                 'but got {:d}.'.format(a.shape[1], c.shape[1])
                                 )

            user_shape = (c.shape[0], b.shape[1])
            # To save the user from the incredibly boring d matrix typing
            # when d = 0, check if d is given
            if None_flags[3] is True:
                d = np.zeros(user_shape)

            if d.shape != (user_shape):
                # Accept annoying 1D inputs for D matrices
                if d.shape[0] == 1 and d.shape == (b.shape[1], c.shape[0]):
                    if verbose:
                        print('It looks like D was a 1D input hence '
                              'I made it a column vector.')
                    d = d.reshape(-1, 1)
                else:
                    raise ValueError('D matrix must have the same number of'
                                     'rows/columns \nwith C/B matrices. I '
                                     'need the shape ({0[0]:d},{0[1]:d}) '
                                     'but got ({1[0]:d},{1[1]:d}).'
                                     ''.format(user_shape, d.shape))

            return a, b, c, d, user_shape, Gain_flag
        else:
            return a, b, c, d, d.shape, Gain_flag


def _pole_properties(poles, dt=None, output_data=False):
    '''
    This function provides the natural frequency, damping and time constant
    values of each poles in a tabulated format. Pure integrators have zero
    frequency and NaN as the damping value. Poles at infinity are discarded.

    Parameters
    ----------
    poles : ndarray
        Poles of the system representation. p must be a 1D array.
    dt : float
        Sampling period for discrete-time poles.

    Returns
    -------
    props : ndarray
        The resulting array holds the poles in the first column, natural
        frequencies in the second and damping ratios in the third.
        # TODO : Will be implemented!!!
        The result is an array whose first column is the one of the complex
        pair or the real pole. When tabulated the complex pair is represented
        as "<num> ± <num>j" using single entry. However the data is kept as
        a valid complex number for convenience. If output_data is set to
        True the numerical values will be returned instead of the string
        type tabulars.

    Notes
    -----
    It should be noted that these properties have very little or no importance
    except some second order system examples in the academic setting or beyond
    second order systems. For higher order systems and also for MIMO systems
    these frequencies and damping ratio values hardly ever mean anything
    unless there are separable poles/modes. It is just a quick way to get a
    geometric intuition about the location of the poles.
    '''
    # Protect system pole value info
    p = poles.copy()

    n = np.size(p)
    # If a static gain is given
    if n == 0:
        return None
    freqn = np.empty_like(p, dtype=float)
    damp = np.empty_like(p, dtype=float)

    # Check for pure integrators
    if dt is not None:  # Discrete
        z_p = p == 1
    else:
        z_p = p == 0

    nz_p = np.logical_not(z_p)
    freqn[z_p] = 0
    damp[z_p] = np.NaN

    if dt is not None:
        p[nz_p] = np.log(p[nz_p])/dt

    freqn[nz_p] = np.abs(p[nz_p])
    damp[nz_p] = -np.real(p[nz_p])/freqn[nz_p]
    return np.c_[poles.copy(), freqn, damp]


def state_to_transfer(state_or_abcd, output='system'):
    """
    Converts a :class:`State` to a :class:`Transfer`

    If the input is a :class:`Transfer` object it returns the argument with no
    modifications.

    The algorithm [1]_ can be summarized as iterating over every row/columns
    of C/B to get SISO Transfer representations via :math:`c(sI-A)^{-1}b+d`.

    Parameters
    ----------
    state_or_abcd : State, tuple

    output : str
        Selects whether a State object or individual numerator, denominator
        will be returned via the options ``'system'``,``'polynomials'``.

    Returns
    -------
    G : Transfer, tuple
        If ``output`` keyword is set to ``'system'`` otherwise a 2-tuple of
        ndarrays is returned as ``num`` and ``den`` if the ``output`` keyword
        is set to ``'polynomials'``

    References
    ----------
    .. [1] Varga, Sima, 1981, :doi:`10.1080/00207178108922980`.

    """
    # FIXME : Resulting TFs are not minimal per se. simplify them, maybe?

    if output.lower() not in ('system', 'polynomials', 's', 'p'):
        raise ValueError('The "output" keyword can either be "system" or '
                         '"polynomials". I don\'t know any option as '
                         '"{0}"'.format(output))

    output = output.lower()[0]
    # If a discrete time system is given this will be modified to the
    # SamplingPeriod later.
    dt = None
    system_given = isinstance(state_or_abcd, State)

    if system_given:
        A, B, C, D = state_or_abcd.matrices
        p, m = state_or_abcd.shape
        it_is_gain = state_or_abcd._isgain
        dt = state_or_abcd.SamplingPeriod
    else:
        try:
            A, B, C, D = state_or_abcd
            it_is_gain = True if all([x.size == 0 for x in
                                      [A, B, C]]) else False

        except ValueError:
            # probably static gain
            A, B, C, D = None, None, None, state_or_abcd[0]
            it_is_gain = True
        p, m = D.shape

    if it_is_gain:
        if output == 'p':
            return D, np.ones_like(D)
        return Transfer(D, dt=dt)

    n = A.shape[0]

    p, m = C.shape[0], B.shape[1]
    n = np.shape(A)[0]
    pp = eigvals(A)

    entry_den = np.real(haroldpoly(pp))
    # Allocate some list objects for num and den entries
    num_list = [[None]*m for rows in range(p)]
    den_list = [[entry_den]*m for rows in range(p)]

    for rowind in range(p):  # All rows of C
        for colind in range(m):  # All columns of B

            b = B[:, colind:colind+1]
            c = C[rowind:rowind+1, :]
            # zz might contain noisy imaginary numbers but since the result
            # should be a real polynomial, we should be able to get away
            # with it

            zz = transmission_zeros(A, b, c, np.array([[0]]))

            # For finding k of a G(s) we compute
            #          pole polynomial evaluated at some s0
            # G(s0) * --------------------------------------
            #          zero polynomial evaluated at some s0
            # s0 : some point that is not a pole or a zero

            # Additional factor of 2 are just some tolerances

            if zz.size != 0:
                s0 = max(np.abs([*pp, *zz]).max(), .5)*2
                zero_prod = np.real(np.prod(s0*np.ones_like(zz) - zz))
            else:
                s0 = max(np.abs(pp).max(), .5)*2
                zero_prod = 1.0  # Not zero!

            CAB = c @ np.linalg.lstsq((s0*np.eye(n)-A), b, rcond=-1)[0]
            pole_prod = np.prod(s0 - pp).real
            entry_gain = (CAB*pole_prod/zero_prod).flatten()

            # Now, even if there are no zeros, den(s) x DC gain becomes
            # the new numerator hence endless fun there

            dentimesD = D[rowind, colind] * entry_den
            if zz.size == 0:
                entry_num = entry_gain
            else:
                entry_num = haroldpoly(zz).real
                entry_num = np.convolve(entry_gain, entry_num)
            entry_num = haroldpolyadd(entry_num, dentimesD)
            if entry_num.size == 0:
                entry_num = np.array([0.])
                den_list[rowind][colind] = np.array([[1.]])
            num_list[rowind][colind] = entry_num[None, :]

    # Strip SISO result from List of list and return as arrays.
    if (p, m) == (1, 1):
        num_list = num_list[0][0]
        den_list = den_list[0][0]

    if output == 'p':
        return num_list, den_list

    return Transfer(num_list, den_list, dt=dt)


def transfer_to_state(G, output='system'):
    """
    Converts a :class:`Transfer` to a :class:`State`

    Given a Transfer() object of a tuple of numerator and denominator,
    converts the argument into the state representation. The output can
    be selected as a State() object or the A,B,C,D matrices if 'output'
    keyword is given with the option 'matrices'.

    If the input is a State() object it returns the argument with no
    modifications.

    For SISO systems, the algorithm is returning the controllable
    companion form.

    For MIMO systems a variant of the algorithm given in Section 4.4 of
    W.A. Wolowich, Linear Multivariable Systems (1974). The denominators
    are equaled with haroldlcm() Least Common Multiple function.

    Parameters
    ----------
    G : {Transfer, State, (num, den)}
        The system or a tuple containing the numerator and the denominator
    output : {'system','matrices'}
        Selects whether a State() object or individual state matrices
        will be returned.

    Returns
    -------
    Gs : State
        If 'output' keyword is set to 'system'
    A,B,C,D : {(nxn),(nxm),(p,n),(p,m)} 2D Numpy-arrays
        If the 'output' keyword is set to 'matrices'

    Notes
    -----
    If G is a State object, it is returned directly.
    """
    if output.lower() not in ('system', 'matrices'):
        raise ValueError('The output can either be "system" or "polynomials".'
                         '\nI don\'t know any option as "{0}"'.format(output))

    # mildly check if we have a transfer,state, or (num,den)
    if isinstance(G, tuple):
        num, den = G
        num, den, (p, m), it_is_gain = Transfer.validate_arguments(num, den)
        dt = None
    elif isinstance(G, State):
        return G.matrices if output == 'matrices' else G
    elif isinstance(G, Transfer):
        G = deepcopy(G)
        num = G.num
        den = G.den
        m, p = G.NumberOfInputs, G.NumberOfOutputs
        it_is_gain = G._isgain
        dt = G.SamplingPeriod
    else:
        custom_msg = 'The argument should be a Transfer or a State'\
                     '. Instead found a {}'.format(type(G).__qualname__)
        raise ValueError(custom_msg)

    # Arguments should be regularized here.
    # Check if it is just a gain
    if it_is_gain:
        A, B, C = (np.empty((0, 0)),)*3
        if np.max((m, p)) > 1:
            D = np.empty((p, m), dtype=float)
            for rows in range(p):
                for cols in range(m):
                    D[rows, cols] = num[rows][cols]/den[rows][cols]
        else:
            D = num/den

        return (A, B, C, D) if output == 'matrices' else State(D, dt=dt)

    if (m, p) == (1, 1):  # SISO
        A = haroldcompanion(den)
        B = np.vstack((np.zeros((A.shape[0]-1, 1)), 1))
        # num and den are now flattened
        num = np.trim_zeros(num[0], 'f')
        den = np.trim_zeros(den[0], 'f')

        # Monic denominator
        if den[0] != 1.:
            d = den[0]
            num, den = num/d, den/d

        if num.size < den.size:
            C = np.zeros((1, den.size-1))
            C[0, :num.size] = num[::-1]
            D = np.array([[0]])
        else:
            # Watch out for full cancellation !!
            NumOrEmpty, datanum = haroldpolydiv(num, den)
            # Clean up the tiny entries
            datanum[np.abs(datanum) < spacing(100.)] = 0.

            # If all cancelled datanum is returned empty
            if datanum.size == 0:
                A = None
                B = None
                C = None
            else:
                C = np.zeros((1, den.size-1))
                C[0, :datanum.size] = datanum[::-1]

            D = np.atleast_2d(NumOrEmpty).astype(float)

    # MIMO ! Implement a "Wolowich LMS-Section 4.4 (1974)"-variant.
    else:
        # Allocate D matrix
        D = np.zeros((p, m))

        for x in range(p):
            for y in range(m):
                # Possible cases (not minimality,only properness checked!!!):
                # 1.  3s^2+5s+3 / s^2+5s+3  Proper
                # 2.  s+1 / s^2+5s+3        Strictly proper
                # 3.  s+1 / s+1             Full cancellation
                # 4.  3   /  2              Just gains

                datanum = np.trim_zeros(num[x][y].flatten(), 'f')
                dataden = np.trim_zeros(den[x][y].flatten(), 'f')
                nn, nd = datanum.size, dataden.size

                if nd == 1:  # Case 4 : nn should also be 1.
                    D[x, y] = datanum/dataden if nn > 0 else 0.
                    num[x][y] = np.array([0.])

                elif nd > nn:  # Case 2 : D[x,y] is trivially zero
                    pass  # D[x,y] is already 0.
                else:
                    NumOrEmpty, datanum = haroldpolydiv(datanum, dataden)
                    # Clean up the tiny entries
                    datanum[np.abs(datanum) < spacing(100.)] = 0.

                    # Case 3: If all cancelled datanum is returned empty
                    if np.count_nonzero(datanum) == 0:
                        D[x, y] = NumOrEmpty
                        num[x][y] = np.array([[0.]])
                        den[x][y] = np.array([[1.]])

                    # Case 1: Proper case
                    else:
                        D[x, y] = NumOrEmpty
                        num[x][y] = np.atleast_2d(datanum)

                # Make the denominator entries monic
                if den[x][y][0, 0] != 1.:
                    num[x][y] = np.array([1/den[x][y][0, 0]])*num[x][y]
                    den[x][y] = np.array([1/den[x][y][0, 0]])*den[x][y]

        # anything left for dynamics (Static Gain)?
        if all([np.array_equal(num[x][y].ravel(), np.array([0.]))
                for x in range(p) for y in range(m)]):
            # Do nothing D is populated above anyways
            A, B, C = (np.empty((0, 0)),)*3
            return (A, B, C, D) if output == 'matrices' else State(D, dt=dt)
        # OK first check if the denominator is common in all entries
        elif all([np.array_equal(den[x][y], den[0][0]) for x in range(p)
                  for y in range(m)]):

            # Nice, less work. Off to realization. Decide rows or cols?
            if p >= m:  # Tall or square matrix => Right Coprime Fact.
                factorside = 'r'
            else:  # Fat matrix, pertranspose the List of Lists => LCF.
                factorside = 'l'
                num = [list(i) for i in zip(*num)]
                p, m = m, p

            # Denominator is common pick one
            d = den[0][0].size-1
            A = haroldcompanion(den[0][0])
            B = np.zeros((A.shape[0], 1), dtype=float)
            B[-1, 0] = 1.
            t1, t2 = A, B

            for x in range(m-1):
                A = block_diag(A, t1)
                B = block_diag(B, t2)
            n = A.shape[0]
            C = np.zeros((p, n))
            k = 0
            for y in range(m):
                for x in range(p):
                    C[x, k:k+num[x][y].size] = num[x][y][0, ::-1]
                k += d  # Shift to the next companion group position

            if factorside == 'l':
                A, B, C = A.T, C.T, B.T
        else:  # Off to LCM computation
            # Get every column denominators and compute the LCM
            # and mults then modify denominators accordingly and
            # add multipliers to nums.

            if p >= m:  # Tall or square matrix => Right Coprime Fact.
                factorside = 'r'
            else:  # Fat matrix, pertranspose => Left Coprime Fact.
                factorside = 'l'
                den = [list(i) for i in zip(*den)]
                num = [list(i) for i in zip(*num)]
                p, m = m, p

            coldens = [x for x in zip(*den)]
            for col in range(m):
                lcm, mults = haroldlcm(*coldens[col])
                for row in range(p):
                    den[row][col] = lcm
                    num[row][col] = haroldpolymul(num[row][col][0],
                                                  mults[row],
                                                  trim_zeros=False)[None, :]

                    # if completely zero, then trim to single entry
                    # Notice that trim_zeros removes everything if all zero
                    # Hence work with temporary variable
                    temp = np.trim_zeros(num[row][col][0], 'f')[None, :]
                    if temp.size == 0:
                        num[row][col] = np.array([[0.]])
                    else:
                        num[row][col] = temp

            coldegrees = [x.size-1 for x in den[0]]

            # Make sure that all static columns are handled
            # Since D is extracted it doesn't matter if denominator is not 1.
            Alist = []
            for x in range(m):
                if den[0][x].size > 1:
                    Alist += [haroldcompanion(den[0][x])]
                else:
                    Alist += [np.empty((0, 0))]

            A = block_diag(*Alist)
            n = A.shape[0]
            B = zeros((n, m), dtype=float)
            C = np.zeros((p, n), dtype=float)
            k = 0

            for col in range(m):
                if den[0][col].size > 1:
                    B[sum(coldegrees[:col+1])-1, col] = 1.

                for row in range(p):
                    C[row, k:k+num[row][col].size] = num[row][col][0, ::-1]
                k += coldegrees[col]

            if factorside == 'l':
                A, B, C = A.T, C.T, B.T

    return (A, B, C, D) if output == 'matrices' else State(A, B, C, D, dt)


def transmission_zeros(A, B, C, D):
    """
    Computes the transmission zeros of :class:`State` data arrays ``A``, ``B``,
    ``C``, ``D``

    Parameters
    ----------
    A,B,C,D : ndarray
        The input data matrices with ``n x n``, ``n x m``, ``p x n``, ``p x m``
        shapes.

    Returns
    -------
    z : ndarray
        The array of computed transmission zeros. The array is returned
        empty if no transmission zeros are found.

    Notes
    -----
    This is a straightforward implementation of [1]_ but via skipping the
    descriptor matrix which in turn becomes [2]_.

    References
    ----------
    .. [1] Misra, van Dooren, Varga, 1994, :doi:`10.1016/0005-1098(94)90052-3`
    .. [2] Emami-Naeini, van Dooren, 1979, :doi:`10.1016/0005-1098(82)90070-X`

    """
    n, (p, m) = A.shape[0], D.shape
    r = np.linalg.matrix_rank(D)
    # Trivially zero, transmission zero doesn't make sense
    # and becomes a c'bility/o'bility test. We don't need that.
    if not np.any(B) or not np.any(C):
        return np.zeros((0, 1))
    elif (p == 1 and m == 1 and r > 0) or (r == min(p, m) and p == m):
        Arc, Brc, Crc, Drc = (A, B, C, D)
    else:  # Reduction needed
        if r == p:
            Ar, Br, Cr, Dr = (A, B, C, D)
        else:
            Ar, Br, Cr, Dr = _tzeros_reduce(A, B, C, D)

        if Ar.size == 0:
            return np.zeros((0, 1))

        n, (p, m) = Ar.shape[0], Dr.shape

        if not np.any(np.c_[Cr, Dr]) == 0 or p != m:
            Arc, Crc, Brc, Drc = _tzeros_reduce(Ar.T, Cr.T, Br.T, Dr.T)
            Arc, Crc, Brc, Drc = Arc.T, Crc.T, Brc.T, Drc.T
        else:
            Arc, Brc, Crc, Drc = (Ar, Br, Cr, Dr)

    if Arc.size == 0:
        return np.zeros((0, 1))

    n, (p, m) = Arc.shape[0], Drc.shape

    *_, v = haroldsvd(np.hstack((Drc, Crc)))
    v = np.roll(np.roll(v.T, -m, axis=0), -m, axis=1)
    T = np.hstack((Arc, Brc)) @ v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b, *_ = qz(v[:n, :n], T[:n, :n], output='complex')
    # Handle zeros at infinity
    diaga = np.diag(a)
    idx = np.nonzero(diaga)
    z = np.full_like(diaga, np.inf)
    z[idx] = np.diag(b)[idx]/diaga[idx]
    return np.real_if_close(z)


def _tzeros_reduce(A, B, C, D):
    """
    Basic deflation loop until we get a full row rank feedthrough matrix.
    """
    m_eps = np.spacing(100 * np.sqrt((A.shape[0] + C.shape[0]) * (
                       A.shape[1] + B.shape[1]))) * norm(A, 'fro')

    for x in range(A.shape[0]):  # At most!
        n, (p, m) = A.shape[0], D.shape
        # Is there anything in D?
        if np.any(D):
            q_of_d, ss, vv, sigma = haroldsvd(D, also_rank=1, rank_tol=m_eps)
            r_of_d = ss @ vv
            tau = p - sigma
            if tau == 0:  # In case we have full rank then done
                break
            Cbd = q_of_d.T @ C
        else:
            sigma, tau = 0, p
            Cbd = C
        # Partition C accordingly
        Cbar = Cbd[:sigma, :]
        Ctilde = Cbd[sigma:, :]
        q_of_c, *_, rho = haroldsvd(Ctilde.T, also_rank=1, rank_tol=m_eps)
        nu = n - rho
        if rho == 0:  # [C,D] happen to be compressed simultaneously
            break
        elif nu == 0:  # [C, D] happen to form a invertible matrix
            A, B, C, D = np.array([]), np.array([]), np.array([]), np.array([])
            break

        q_of_c = np.fliplr(q_of_c)  # Compress on the right side of C
        if sigma > 0:
            AC_slice = np.r_[q_of_c.T @ A, Cbar] @ q_of_c
            A, C = AC_slice[:nu, :nu], AC_slice[nu:, :nu]
            BD_slice = np.r_[(q_of_c.T @ B), r_of_d[:sigma, :]]
            B, D = BD_slice[:nu, :], BD_slice[nu:, :]
        else:
            ABCD_slice = q_of_c.T @ np.c_[A @ q_of_c, B]
            A, B, C, D = (ABCD_slice[:nu, :nu], ABCD_slice[:nu, -m:],
                          ABCD_slice[nu:, :nu], ABCD_slice[nu:, -m:])
    return A, B, C, D


def _state_or_abcd(arg, n=4):
    """
    Tests the argument for being a State() object or any number of
    model matrix arguments for testing. The typical use case is to accept
    the arguments regardless of whether the input is a class instance or
    standalone matrices.

    The optional n argument is for testing state matrices less than four.
    For example, the argument should be tested for either being a State()
    object or A,B matrix for controllability. Then we select n=2 such that
    only A,B but not C,D is sought after. The default is all four matrices.

    If matrices are given, it passes the argument through the
    State.validate_arguments() method to regularize and check the sizes etc.

    Parameters
    ----------
    arg : State, tuple
        The argument to be parsed and checked for validity. Expects either
        a State model or a tuple holding the model matrices.
    n : int
        If we let A,B,C,D numbered as 1,2,3,4, defines the test scope such
        that only up to n-th matrix is tested. To test only an A,C use n = -1.

    Returns
    -------
    system_or_not : Boolean
        True if system and False otherwise
    validated_matrices: ndarray
        The validated n-many 2D arrays.

    Notes
    -----
    The n=1 case is just for regularity. Only checks the squareness of A. Also
    if the args is a tuple and the first element is an empty array then n is
    assumed to be 4.
    """
    val_arg = State.validate_arguments

    if n not in [-1, 1, 2, 3, 4]:
        raise ValueError('n must be one of the [-1, 1, 2, 3, 4] but got {}'
                         ''.format(n))
    if isinstance(arg, State):
        return True, None
    elif isinstance(arg, tuple):
        system_or_not = False
        # treat static model early, n = 4 necessarily
        if n != 1 and len(arg) == 1:
            return system_or_not, (None, None, None, arg[0])
        else:
            # Start with squareness of a - always given
            a = arg[0]
            _assert_square(a)

            if n == 1:
                return system_or_not, np.atleast_2d(arg[0])
            elif n == 2:
                b = np.atleast_2d(arg[1])
                n, m = b.shape
                if n != a.shape[0]:
                    raise ValueError('b array should have same number of rows'
                                     'as a. a is {} but b is {}'
                                     ''.format(a.shape, b.shape))
                return system_or_not, (a, b)
            elif n == 3:
                b = np.atleast_2d(arg[1])
                n, m = b.shape
                c = np.atleast_2d(arg[2])
                p, nc = c.shape
                if n != a.shape[0]:
                    raise ValueError('b array should have same number of rows'
                                     'as a. a is {} but b is {}'
                                     ''.format(a.shape, b.shape))

                if nc != a.shape[0]:
                    raise ValueError('c array should have same number of cols'
                                     'as a. a is {} but c is {}'
                                     ''.format(a.shape, b.shape))

                return system_or_not, (a, b, c)
            elif n == 4:
                a, b, c, d, _, _ = val_arg(*arg)
                return system_or_not, (a, b, c, d)
            else:
                c = np.atleast_2d(arg[1])
                p, nc = c.shape
                if nc != a.shape[0]:
                    raise ValueError('c array should have same number of cols'
                                     'as a. a is {} but c is {}'
                                     ''.format(a.shape, b.shape))
                return system_or_not, (a, c)
    else:
        raise ValueError('The argument is neither a tuple nor a State() '
                         'object. The argument is of the type "{}"'
                         ''.format(type(arg).__qualname__))


def random_state_model(n, p=1, m=1, dt=None, prob_dist=None, stable=True):
    """
    Generates a continuous or discrete State model with random data.

    The poles of the model is selected from randomly generated numbers with a
    predefined probability assigned to each pick which can also be provided by
    external array. The specification of the probability is a simple 5-element
    array-like ``[p0, p1, p2, p3]`` denoting ::

        p0 : Probability of having a real pole
        p1 : Probability of having a complex pair anywhere except real line
        p2 : Probability of having an integrator (s or z domain)
        p3 : Probability of having a pair on the imaginary axis/unit circle

    Hence, ``[1, 0, 0, 0]`` would return a model with only real poles. Notice
    that the sum of entries should sum up to 1. See numpy.random.choice for
    more details. The default probabilities are ::

        [p0, p1, p2, p3] = [0.475, 0.475, 0.025, 0.025]

    If ``stable`` keyword is set to True ``prob_dist`` must be 2-entry
    arraylike denoting only ``[p0, p1]``. The default is uniform i.e. ::

        [p0, p1] = [0.5, 0.5]

    Parameters
    ----------
    n : int
        The number of states. For static models use ``n=0``.
    p : int, optional
        The number of outputs. The default is 1
    m : int, optional
        The number of inputs. The default is 1
    prob_dist : arraylike, optional
        An arraylike with 4 nonnegative entries of which the sum adds up to 1.
        If ``stable`` key is True then it needs only 2 entries. Internally, it
        uses ``numpy.random.choice`` for selection.
    dt : float, optional
        If ``dt`` is not None, then the value will be used as sampling period
        to create a discrete time model. This argument must be nonnegative.
    stable : bool, optional
        If True a stable model is returned, otherwise stability model would
        not be checked

    Returns
    -------
    G : State
        The resulting State model

    Notes
    -----
    Internally, n, p, m will be converted to integers if possible. This means
    that no checks about the decimal part will be performed.

    Similarly ``prob_dist`` will be passed directly to a numpy.ndarray with
    explicitly taking the real part.

    Note that probabilities denote the choice not the distribution of the
    poles, in other words for a 3-state model, single real pole and a
    complex pair have the same probability of choice however real pole
    constitute one third.

    """
    # Check arguments
    n, p, m = int(n), int(p), int(m)

    if prob_dist is None:
        if stable:
            pdist = np.array([0.5, 0.5])
        else:
            pdist = np.array([0.475, 0.475, 0.025, 0.025])
    else:
        pdist = _asarray_validated(prob_dist).real

    # pdist will err inside numpy.random.choice so skip checks.

    # Static model
    if n == 0:
        return State(rand(p, m), dt=dt)
    elif n == 1:
        a, b, c, d = rand(1), rand(1, m), rand(p, 1), rand(p, m)
        return State(-a, b, c, d, dt=dt)

    # Get random pole types
    choose_from = [0, 1] if stable else [0, 1, 2, 3]
    diag_i = 0
    a = zeros((n, n))

    # Walk over the diagonal of "a"
    for _ in range(n):

        p_type = choice(choose_from, 1, replace=True, p=pdist)
        if p_type == 0:
            ps = choice([1, -1], 1)
            pr = -exp(exp(rand())) if stable else ps*exp(exp(rand()))
            a[diag_i, diag_i] = pr
            diag_i += 1

        elif p_type == 1:
            if diag_i >= n-1:
                break
            ps = choice([1, -1], 1)
            pr = -exp(exp(rand())) if stable else ps*exp(exp(rand()))
            pi = exp(exp(rand()))
            a[[diag_i, diag_i, diag_i+1, diag_i+1],
              [diag_i, diag_i+1, diag_i, diag_i+1]] = [pr, pi, -pi, pr]
            diag_i += 2

        elif p_type == 2:
            a[diag_i, diag_i] = 0.
            diag_i += 1

        elif p_type == 3:
            if diag_i >= n-1:
                break
            pi = exp(exp(rand()))
            a[[diag_i, diag_i, diag_i+1, diag_i+1],
              [diag_i, diag_i+1, diag_i, diag_i+1]] = [0., pi, -pi, 0.]
            diag_i += 2

        # Finished all diagonals
        if diag_i == n:
            break

    # Complex didn't fit, fill the remaining
    if diag_i == n-1:
        ps = choice([1, -1], 1)
        pr = -exp(exp(rand())) if stable else ps*exp(exp(rand()))
        a[diag_i, diag_i] = pr

    # Convert poles to discrete if dt != None
    if dt is not None:
        a = expm(a*dt)
    # Perform a random similarity transformation to shuffle the data
    T = ortho_group.rvs(n)
    a = solve(T, a) @ T

    return State(a, rand(n, m), rand(p, n), rand(p, m), dt=dt)


def concatenate_state_matrices(G):
    """
    Takes a State() model as input and returns the A, B, C, D matrices
    combined into a full matrix. For static gain models, the feedthrough
    matrix D is returned.

    Parameters
    ----------
    G : State

    Returns
    -------
    M : ndarray
    """
    if not isinstance(G, State):
        raise ValueError('concatenate_state_matrices() works on state '
                         'representations, but I found \"{0}\" object '
                         'instead.'.format(type(G).__name__))
    if G._isgain:
        return G.d

    return np.block([[G.a, G.b], [G.c, G.d]])
