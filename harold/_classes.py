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

from scipy.linalg import eigvals, block_diag, qz, norm
from tabulate import tabulate
from itertools import zip_longest, chain

from ._polynomial_ops import (haroldpoly, haroldpolyadd, haroldpolydiv,
                              haroldpolymul, haroldcompanion,
                              haroldtrimleftzeros, haroldlcm)

from ._aux_linalg import e_i, haroldsvd
#from ._system_funcs import minimal_realization
from ._global_constants import _KnownDiscretizationMethods
from copy import deepcopy

__all__ = ['Transfer', 'State', 'state_to_transfer', 'transfer_to_state',
           'transmission_zeros', 'concatenate_state_matrices']


class Transfer:
    """
    Transfer is the one of two main system classes in harold (together
    with State()).

    Main types of instantiation of this class depends on whether the
    user wants to create a Single Input/Single Output system (SISO) or
    a Multiple Input/Multiple Output system (MIMO) model.

    For SISO system creation, 1D lists or 1D numpy arrays are expected,
    e.g.,::

        >>>> G = Transfer(1,[1,2,1])

    For MIMO systems, depending on the shared denominators, there are
    two distinct ways of entering a MIMO transfer function:

        1.  Entering "list of lists of lists" such that every element of the
        inner lists are numpy array-able (explicitly checked) for numerator
        and entering a 1D list or 1D numpy array for denominator (and
        similarly for numerator)::

            >>>> G = Transfer([[[1,3,2],[1,3]],[[1],[1,0]]],[1,4,5,2])
            >>>> G.shape
            (2,2)

        2. Entering the denominator also as a list of lists for individual
        entries as a bracket nightmare (thanks to Python's nonnative support
        for arrays and tedious array syntax)::

            >>>> G = Transfer([
                     [ [1,3,2], [1,3] ],
                     [   [1]  , [1,0] ]
                   ],# end of num
                   [
                      [ [1,2,1] ,  [1,3,3]  ],
                      [ [1,0,0] , [1,2,3,4] ]
                   ])
           >>>> G.shape
           (2,2)



    There is a very involved validator and if you would like to know
    why or how this input is handled ,provide the same numerator and
    denominator to the static method below with 'verbose=True' keyword
    argument, e.g. ::

        >>>> n , d , shape , is_it_static = Transfer.validate_arguments(
                  [1,3,2], # common numerator
                  [[[1,2,1],[1,3,3]],[[1,0,0],[1,2,3,4]]],# explicit den
                  verbose=True # print the logic it followed
                  )

    would give information about the context together with the
    regularized numerator, denominator, resulting system shape
    and boolean whether or not the system has dynamics.

    However, the preferred way is to make everything a numpy array inside
    the list of lists. That would skip many compatibility checks.
    Once created the shape of the numerator and denominator cannot be
    changed. But compatible sized arrays can be supplied and it will
    recalculate the pole/zero locations etc. properties automatically.

    The Sampling Period can be given as a last argument or a keyword
    with 'dt' key or changed later with the property access.::

        >>>> G = Transfer([1],[1,4,4],0.5)
        >>>> G.SamplingSet
        'Z'
        >>>> G.SamplingPeriod
        0.5
        >>>> F = Transfer([1],[1,2])
        >>>> F.SamplingSet
        'R'
        >>>> F.SamplingPeriod = 0.5
        >>>> F.SamplingSet
        'Z'
        >>>> F.SamplingPeriod
        0.5

    Providing 'False' value to the SamplingPeriod property will make
    the system continous time again and relevant properties are reset
    to CT properties.

    .. warning:: Unlike matlab or other tools, a discrete time system
        needs a specified sampling period (and possibly a discretization
        method if applicable) because a model without a sampling period
        doesn't make sense for analysis. If you don't care, then make up
        a number, say, a million, since you don't care.
    """

    def __init__(self, num, den=None, dt=False):

        # Initialization Switch and Variable Defaults

        self._isgain = False
        self._isSISO = False
        self._isstable = False
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._SamplingPeriod = False
        (self._num, self._den,
         self._shape, self._isgain) = self.validate_arguments(num, den)
        self._p, self._m = self._shape
        if self._shape == (1, 1):
            self._isSISO = True
        self.SamplingPeriod = dt

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
        return self._SamplingPeriod

    @property
    def SamplingSet(self):
        """
        If this property is called ``G.SamplingSet`` then returns the
        set ``Z`` or ``R`` for discrete and continous models respectively.
        This is a read only property and cannot be set. Instead an appropriate
        setting should be given to the ``SamplingPeriod`` property.
        """
        return self._SamplingSet

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
        ``undiscretize()`` function to reach back to the continous model that
        would hopefully minimize the discretization errors. It is also
        possible to manually set this property such that ``undiscretize``
        uses the provided method.
        """
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                    'a discretization method associated with it.')
        elif self._DiscretizedWith is None:
            return ('It is a discrete-time model with no '
                    'discretization method associated with it during '
                    'its creation.')
        else:
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
            <http://dx.doi.org/10.1080/00207170802247728>`_

        """
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                    'a discretization matrix associated with it.')
        elif not self.DiscretizedWith == 'lft':
            return ('This model is discretized with a method that '
                    'has no discretization matrix associated with '
                    'it.')
        elif self._DiscretizedWith is None:
            return ('It is a discrete-time model with no '
                    'discretization method associated with it during '
                    'its creation.')
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
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                    'a discretization matrix associated with it.')
        elif self.DiscretizedWith not in ('tustin',
                                          'bilinear',
                                          'trapezoidal'):
            return ('This model is not discretized with Tustin'
                    'approximation hence prewarping does not apply.')

    @SamplingPeriod.setter
    def SamplingPeriod(self, value):
        if value:
            self._SamplingSet = 'Z'
            if type(value) is bool:  # integer 1 != True
                self._SamplingPeriod = 0.
            elif isinstance(value, (int, float)):
                self._SamplingPeriod = float(value)
            else:
                raise TypeError('SamplingPeriod must be a real positive '
                                'scalar. But looks like a \"{0}\" is '
                                'given.'.format(
                                   type(value).__name__))
        else:
            self._SamplingSet = 'R'
            self._SamplingPeriod = None

    @num.setter
    def num(self, value):

        user_num, _, user_shape = validate_arguments(value, self._den)[:3]

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

        user_den, user_shape = validate_arguments(self._num, value)[1:3]

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
                if value == 'lft':
                    self._DiscretizedWith = value
                    print('\"lft\" method also needs an interconnection'
                          ' matrix. Please don\'t forget to set the '
                          '\"DiscretizationMatrix\" property as well')
                else:
                    self._DiscretizedWith = value
        else:
            raise ValueError('Excuse my ignorance but I don\'t know '
                             'that method.')

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
            raise TypeError('If the discretization method is not '
                            'Tustin then you don\'t need to set '
                            'this property.')
        else:
            if value > 1/(2*self._SamplingPeriod):
                raise ValueError('Prewarping Frequency is beyond '
                                 'the Nyquist rate.\nIt has to '
                                 'satisfy 0 < w < 1/(2*dt) and dt '
                                 'being the sampling\nperiod in '
                                 'seconds (dt={0} is provided, '
                                 'hence the max\nallowed is '
                                 '{1} Hz.'.format(dt, 1/(2*dt)))
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
                zzz = transfer_to_state(self._num, self._den,
                                        output='matrices')
                self.zeros = transmission_zeros(*zzz)
                self.poles = eigvals(zzz[0])

        self._set_stability()
        self._set_representation()

    def _set_stability(self):
        if self._SamplingSet == 'Z':
            self._isstable = all(1 > abs(self.poles))
        else:
            self._isstable = all(0 > np.real(self.poles))

    def _set_representation(self):
        self._repr_type = 'Transfer'

    # =================================
    # Transfer class arithmetic methods
    # =================================

    def __neg__(self):
        if not self._isSISO:
            newnum = [[None]*self._m for n in range(self._p)]
            for i in range(self._p):
                for j in range(self._m):
                    newnum[i][j] = -self._num[i][j]
        else:
            newnum = -1*self._num

        return Transfer(newnum, self._den, self._SamplingPeriod)

    def __add__(self, other):
        # Addition to a Transfer object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float that is multiplied with a proper "ones" matrix
        # 4. A shape matching numpy array

        # Notice that in case 3 it is a ones matrix not an identity!!
        # (Given a 1x3 system + 5) adds [[5,5,5]].

        if isinstance(other, (Transfer, State)):
            # Trivial Rejections:
            # ===================
            # Reject 'ct + dt' or 'dt + dt' with different sampling periods
            #
            # A future addition would be converting everything to the slowest
            # sampling system but that requires pretty comprehensive change.

            if not self._SamplingPeriod == other._SamplingPeriod:
                raise TypeError('The sampling periods don\'t match '
                                'so I cannot\nadd these systems. '
                                'If you still want to add them as if '
                                'they are\ncompatible, carry the data '
                                'to a compatible system model and then '
                                'add.'
                                )

        # Reject if the size don't match
            if not self._shape == other.shape:
                raise IndexError('Addition of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'.format(
                                                self._shape,
                                                other.shape)
                                 )

        # ===================
            if isinstance(other, Transfer):
                # First get the static gain case out of the way.
                if self._isgain and other._isgain:
                        return Transfer(self._num + other.num,
                                        dt=self._SamplingPeriod)

                # Now, we are sure that there are no possibility other than
                # list of lists or np.arrays hence concatenation should be OK.

                if self._isSISO:
                    lcm, mults = haroldlcm(self._den, other.den)
                    newnum = haroldpolyadd(
                        np.convolve(self._num.flatten(), mults[0]),
                        np.convolve(other.num.flatten(), mults[1]))
                    if np.count_nonzero(newnum) == 0:
                        return Transfer(0, 1)
                    else:
                        return Transfer(newnum, lcm)

                else:
                    # Create empty num and den holders.
                    newnum = [[None]*self._m for n in range(self._p)]
                    newden = [[None]*self._m for n in range(self._p)]
                    nonzero_num = np.zeros(self._shape, dtype=bool)
                    # Same as SISO but over all rows/cols
                    for row in range(self._p):
                        for col in range(self._m):
                            lcm, mults = haroldlcm(
                                            self._den[row][col],
                                            other.den[row][col]
                                            )

                            newnum[row][col] = np.atleast_2d(
                                    haroldpolyadd(
                                        np.convolve(
                                            self._num[row][col].flatten(),
                                            mults[0]
                                        ),
                                        np.convolve(
                                            other.num[row][col].flatten(),
                                            mults[1]
                                        )
                                    )
                                )

                            newden[row][col] = lcm

                        # Test whether we have at least one numerator entry
                        # that is nonzero. Otherwise return a zero MIMO tf
                            if np.count_nonzero(newnum[row][col]) != 0:
                                nonzero_num[row, col] = True

                    if any(nonzero_num.ravel()):
                        return Transfer(newnum, newden,
                                        dt=self._SamplingPeriod)
                    else:
                        # Numerators all cancelled to zero hence 0-gain MIMO
                        return Transfer(np.zeros(self._shape).tolist())
            else:
                return other + transfer_to_state(self)

        # Last chance for matrices, convert to static gain matrices and add
        elif isinstance(other, (int, float)):
            return Transfer((other * np.ones(self._shape)).tolist(),
                            dt=self._SamplingPeriod) + self

        elif isinstance(other, type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self + float(other)

            if self._shape == other.shape:
                return self + Transfer(other, dt=self._SamplingPeriod)
            else:
                raise IndexError('Addition of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'.format(
                                                    self._shape, other.shape))
        else:
            raise TypeError('I don\'t know how to add a '
                            '{0} to a state representation '
                            '(yet).'.format(type(other).__name__))

    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return -self + other

    def __mul__(self, other):
        # Multiplication with a Transfer object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float
        # 4. A shape matching numpy array

        if isinstance(other, (Transfer, State)):
            # Trivial Rejections:
            # ===================
            # Reject 'ct + dt' or 'dt + dt' with different sampling periods
            #
            # A future addition would be converting everything to the slowest
            # sampling system but that requires pretty comprehensive change.

            if not self._SamplingPeriod == other._SamplingPeriod:
                raise TypeError('The sampling periods don\'t match '
                                'so I cannot\nmultiply these systems. '
                                'If you still want to multiply them as'
                                'if they are\ncompatible, carry the data '
                                'to a compatible system model and then '
                                'multiply.'
                                )

        # Reject if the size don't match
            if not self._shape[1] == other.shape[0]:
                raise IndexError('Multiplication of systems requires '
                                 'their shape to match but the system '
                                 'shapes I got are {0} vs. {1}'.format(
                                                self._shape,
                                                other.shape))
        # ===================

            if isinstance(other, Transfer):

                # First get the static gain case out of the way.
                if self._isgain and other._isgain:
                        return State(self.num.dot(other.num),
                                     dt=self._SamplingPeriod)

                if self._isSISO:
                    return Transfer(
                            haroldpolymul(self._num.flatten(),
                                          other.num.flatten()),
                            haroldpolymul(self._den.flatten(),
                                          other.den.flatten()),
                            dt=self._SamplingPeriod)
                else:
                    # Bah.. Here we go!
                    # Same as SISO but over all rows/cols:
                    # Also if the result would be a SISO extra steps
                    # such as stripping off the list of lists

                    # So we have a (p x k) and (k x m) shapes hence
                    # the temporary size variables
                    t_p = self._p
                    t_k = self._m
                    t_m = other.shape[1]

                    newnum = [[None]*t_m for n in range(t_p)]
                    newden = [[None]*t_m for n in range(t_p)]

                    # Here we have again a looping fun:
                    # What we do is to rely on the Transfer()
                    # __add__ method for the SISO case recursively.

                    # Suppose we have a multiplication of a 1x5 Transfer
                    # multiplied with a 5x1 Transfer

                    #                [1]
                    #  [a b c d e] * [2]
                    #                [3]
                    #                [4]
                    #                [5]

                    # What we do here is to form each a1,b2,...e5
                    # and convert each of them to a temporary Transfer
                    # object and then add --> a1 + b2 + c3 + d4 + e5
                    # such that possible common poles don't get spurious
                    # multiplicities.

                    # Recursion part is the fact that we are already
                    # inside a Transfer() method but forming temporary
                    # Transfer()objects within. Fingers crossed...

                    for row in range(t_p):
                        for col in range(t_m):
                            # Zero out the temporary Transfer()
                            t_G = Transfer(0, 1)

                            # for all elements in row/col multiplication
                            for elem in range(t_k):

                                t_num = haroldpolymul(self._num[row][elem],
                                                      other.num[elem][col])

                                t_den = haroldpolymul(self._den[row][elem],
                                                      other.den[elem][col])

                                t_G += Transfer(t_num, t_den)
                            # Add the resulting arrays to the containers.
                            newnum[row][col] = t_G.num
                            newden[row][col] = t_G.den

                    # If the resulting shape is SISO, strip off the lists
                    if (t_p, t_m) == (1, 1):
                        newnum = newnum[0][0]
                        newden = newden[0][0]

                    # Finally return the result.
                    return Transfer(newnum, newden, dt=self._SamplingPeriod)

            elif isinstance(other, State):
                return transfer_to_state(self) * other

        elif isinstance(other, (int, float)):
            return self * Transfer(np.atleast_2d(other),
                                   dt=self._SamplingPeriod)

        # Last chance for matrices, convert to static gain matrices and mult
        elif isinstance(other, type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self * Transfer(
                            np.atleast_2d(other), dt=self._SamplingPeriod)

            if self._shape[1] == other.shape[0]:
                return self * Transfer(other, dt=self._SamplingPeriod)
            else:
                raise IndexError('Multiplication of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'
                                 ''.format(self._shape, other.shape))
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__name__))

    def __rmul__(self, other):
        # Notice that if other is a State or Transfer, it will be handled
        # by other's __mul__() method. Hence we only take care of the
        # right multiplication of the scalars and arrays. Otherwise
        # rejection is executed
        if isinstance(other, (int, float)):
            if self._isSISO:
                return Transfer(other, dt=self._SamplingPeriod) * self
            else:
                return Transfer(np.ones((self._shape))*other,
                                dt=self._SamplingPeriod) * self
        elif isinstance(other, type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return float(other) * self
            elif self._shape[0] == other.shape[1]:
                return Transfer(other, dt=self._SamplingPeriod) * self
            else:
                raise IndexError('Multiplication of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'
                                 ''.format(self._shape, other.shape))
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__name__))

    # ================================================================
    # __getitem__ to provide input-output selection of a tf
    #
    # TODO: How to validate strides
    # ================================================================

#    def __getitem__(self,num_or_slice):
#        print('Lalala I"m not listening lalala')

    def __setitem__(self, *args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                         'the relevant num,den or a,b,c,d properties. '
                         'This might be\nincluded in the future though.')

    # ================================================================
    # __repr__ and __str__ to provide meaningful info about the system
    # The ascii art of matlab for tf won't be implemented.
    # Either proper image with proper superscripts or numbers.
    # ================================================================

    def __repr__(self):
        if self.SamplingSet == 'R':
            desc_text = 'Continous-Time Transfer function\n'
        else:
            desc_text = ('Discrete-Time Transfer function with: '
                         'sampling time: {0:.3f} \n'
                         ''.format(float(self.SamplingPeriod)))

        if self._isgain:
            desc_text += '\n{}x{} Static Gain\n'.format(self.NumberOfOutputs,
                                                        self.NumberOfInputs)
        else:
            desc_text += ' {0} input(s) and {1} output(s)\n'.format(
                                                        self.NumberOfInputs,
                                                        self.NumberOfOutputs
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
            A boolean switch to print out what this method thinksabout the
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
                                 [
                                  np.atleast_2d(np.array(x, dtype='float'))
                                  for x in y
                                 ]
                                 for y in numden
                            ]
                        except:
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
                                         ''.format(entrytext[numden_index]),
                                         max(m), min(m))

                # We found the list and it wasn't a list of lists.
                # Then it should be a regular list to be np.array'd
                elif all([isinstance(x, (int, float)) for x in numden]):
                    if verbose:
                        print('I found a list that has only scalars')
                    try:
                        returned_numden_list[numden_index] = np.atleast_2d(
                                            np.array(numden, dtype='float')
                                            )
                        if numden_index == 1:
                            Gain_flags[1] = True
                    except:
                        raise ValueError('Something is not a \"float\" inside '
                                         'the {0} list.'
                                         ''.format(entrytext[numden_index]))

            # Now we are sure that there is no dynamic MIMO entry.
            # The remaining possibility is a np.array as a static
            # gain for being MIMO. The rest is SISO.
            # Disclaimer: We hope that the data type is 'float'
            # Life is too short to check everything.

            elif isinstance(numden, type(np.array([0.]))):
                if verbose:
                    print('I found a numpy array')
                if numden.ndim > 1 and min(numden.shape) > 1:
                    if verbose:
                        print('The array has multiple elements')
                    returned_numden_list[numden_index] = [
                        [np.array(x, dtype='float') for x in y]
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
                raise TypeError(
                                '{0} must either be a list of lists (MIMO)\n'
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

            den_list = [haroldtrimleftzeros(x) for x in
                        chain.from_iterable(returned_numden_list[1])]

            num_list = [haroldtrimleftzeros(x) for x in
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

            # TODO: Transfer([0,0,0],[1]) leads to error!!

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
                    num_deg = haroldtrimleftzeros(returned_numden_list[0]).size

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
                    except:
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
                    den_deg = haroldtrimleftzeros(returned_numden_list[1]).size

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

        # TODO : Gain_flags are not robust, remove them and make the
        # check below be decisive

        # Final gateway for the static gain
        if isinstance(den, list):
            # Check the max number of elements in each entry
            max_deg_of_den = max([x.size for x in sum(den, [])])
            if verbose:
                print('In the MIMO context and proper entries, I\'ve '
                      'found\nscalar denominator entries hence flagging '
                      'as a static gain.')
            # If less than two, then den is a gain matrix.
            Gain_flag = True if max_deg_of_den == 1 else False
        else:
            Gain_flag = True if den.size == 1 else False
            if verbose:
                print('In the SISO context and a proper rational function'
                      ', I\'ve found\na scalar denominator hence '
                      'flagging as a static gain.')

        return num, den, shape, Gain_flag


class State:
    """

    State() is the one of two main system classes in harold (together with
    Transfer() ).

    A State object can be instantiated in a straightforward manner by
    entering 2D arrays, floats, 1D arrays for row vectors and so on.::

        >>>> G = State([[0,1],[-4,-5]],[[0],[1]],[[1,0]],1)


    However, the preferred way is to make everything a numpy array.
    That would skip many compatibility checks. Once created the shape
    of the system matrices cannot be changed. But compatible
    sized arrays can be supplied and it will recalculate the pole/zero
    locations etc. properties automatically.

    The Sampling Period can be given as a last argument or a keyword
    with 'dt' key or changed later with the property access.::

        >>>> G = State([[0,1],[-4,-5]],[[0],[1]],[[1,0]],[1],0.5)
        >>>> G.SamplingSet
        'Z'
        >>>> G.SamplingPeriod
        0.5
        >>>> F = State(1,2,3,4)
        >>>> F.SamplingSet
        'R'
        >>>> F.SamplingPeriod = 0.5
        >>>> F.SamplingSet
        'Z'
        >>>> F.SamplingPeriod
        0.5

    Setting  SamplingPeriod property to 'False' value to the will make
    the system continous time again and relevant properties are reset
    to continuous-time properties.

    Warning: Unlike matlab or other tools, a discrete time system
    needs a specified sampling period (and possibly a discretization
    method if applicable) because a model without a sampling period
    doesn't make sense for analysis. If you don't care, then make up
    a number, say, a million, since you don't care.
    """

    def __init__(self, a, b=None, c=None, d=None, dt=False):

        self._SamplingPeriod = False
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._isSISO = False
        self._isgain = False
        self._isstable = False

        *abcd, self._shape, self._isgain = self.validate_arguments(a, b, c, d)

        self._a, self._b, self._c, self._d = abcd
        self._p, self._m = self._shape

        if self._shape == (1, 1):
            self._isSISO = True

        self.SamplingPeriod = dt
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
        return self._SamplingPeriod

    @property
    def SamplingSet(self):
        """
        If this property is called ``G.SamplingSet`` then returns the
        set ``Z`` or ``R`` for discrete and continous models respectively.
        This is a read only property and cannot be set. Instead an appropriate
        setting should be given to the ``SamplingPeriod`` property.
        """
        return self._SamplingSet

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
        ``undiscretize()`` function to reach back to the continous model that
        would hopefully minimize the discretization errors. It is also
        possible to manually set this property such that ``undiscretize``
        uses the provided method.
        """
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                    'a discretization method associated with it.')
        elif self._DiscretizedWith is None:
            return ('It is a discrete-time model with no '
                    'discretization method associated with it during '
                    'its creation.')
        else:
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
            <http://dx.doi.org/10.1080/00207170802247728>`_

        """
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                    'a discretization matrix associated with it.')
        elif not self.DiscretizedWith == 'lft':
            return ('This model is discretized with a method that '
                    'has no discretization matrix associated with '
                    'it.')
        elif self._DiscretizedWith is None:
            return ('It is a discrete-time model with no '
                    'discretization method associated with it during '
                    'its creation.')
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
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                    'a discretization matrix associated with it.')
        elif self.DiscretizedWith not in ('tustin',
                                          'bilinear',
                                          'trapezoidal'):
            return ('This model is not discretized with Tustin'
                    'approximation hence prewarping does not apply.')
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
        if value:
            self._SamplingSet = 'Z'
            if type(value) is bool:  # integer 1 != True
                self._SamplingPeriod = 0.
            elif isinstance(value, (int, float)):
                self._SamplingPeriod = float(value)
            else:
                raise TypeError('SamplingPeriod must be a real scalar.'
                                'But looks like a \"{0}\" is given.'.format(
                                 type(value).__name__))
        else:
            self._SamplingSet = 'R'
            self._SamplingPeriod = None

    @DiscretizedWith.setter
    def DiscretizedWith(self, value):
        if value in _KnownDiscretizationMethods:
            if self.SamplingSet == 'R':
                raise ValueError('This model is not discretized yet '
                                 'hence you cannot define a method for'
                                 ' it. Discretize the model first via '
                                 '\"discretize\" function.')
            else:
                if value == 'lft':
                    self._DiscretizedWith = value
                    print('\"lft\" method also needs an interconnection'
                          ' matrix. Please don\'t forget to set the '
                          '\"DiscretizationMatrix\" property as well')
                else:
                    self._DiscretizedWith = value
        else:
            raise ValueError('Excuse my ignorance but I don\'t know '
                             'that method.')

    @DiscretizationMatrix.setter
    def DiscretizationMatrix(self, value):
        if self._DiscretizedWith == 'lft':
            self._DiscretizationMatrix = np.array(value, dtype='float')
        else:
            raise TypeError('If the discretization method is not '
                            '\"lft\" then you don\'t need to set '
                            'this property.')

    @PrewarpFrequency.setter
    def PrewarpFrequency(self, value):
        if self._DiscretizedWith not in ('tustin', 'bilinear', 'trapezoidal'):
            raise TypeError('If the discretization method is not '
                            'Tustin then you don\'t need to set '
                            'this property.')
        else:
            if value > 1/(2*self._SamplingPeriod):
                raise ValueError('Prewarping Frequency is beyond '
                                 'the Nyquist rate.\nIt has to '
                                 'satisfy 0 < w < 1/(2*dt) and dt '
                                 'being the sampling\nperiod in '
                                 'seconds (dt={0} is provided, '
                                 'hence the max\nallowed is '
                                 '{1} Hz.'.format(dt, 1/(2*dt))
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
        if self._SamplingSet == 'Z':
            self._isstable = all(1 > np.abs(self.poles))
        else:
            self._isstable = all(0 > np.real(self.poles))

    def _set_representation(self):
        self._repr_type = 'State'

    # ===========================
    # State class arithmetic methods
    # ===========================

    def __neg__(self):
        if self._isgain:
            return State(-self._d, dt=self._SamplingPeriod)
        else:
            newC = -1. * self._c
            return State(self._a, self._b, newC, self._d, self._SamplingPeriod)

    def __add__(self, other):
        # Addition to a State object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float that is multiplied with a proper "ones" matrix
        # 4. A shape matching numpy array

        # Notice that in case 3 it is a ones matrix not an identity!!
        # (Given a 1x3 system + 5) adds [[5,5,5]] to D matrix.

        if isinstance(other, (Transfer, State)):
            # Trivial Rejections:
            # ===================
            # Reject 'ct + dt' or 'dt + dt' with different sampling periods
            #
            # A future addition would be converting everything to the slowest
            # sampling system but that requires pretty comprehensive change.

            if not self._SamplingPeriod == other._SamplingPeriod:
                raise TypeError('The sampling periods don\'t match '
                                'so I cannot\nadd these systems. '
                                'If you still want to add them as if '
                                'they are\ncompatible, carry the data '
                                'to a compatible system model and then '
                                'add.'
                                )

        # Reject if the size don't match
            if not self._shape == other.shape:
                raise IndexError('Addition of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'.format(
                                                self._shape,
                                                other.shape)
                                 )

        # ===================

            if isinstance(other, State):

                # First get the static gain case out of the way.
                if self._isgain:
                    if other._isgain:
                        return State(self.d + other.d,
                                     dt=self._SamplingPeriod)
                    else:
                        return State(other.a,
                                     other.b,
                                     other.c,
                                     self.d + other.d,
                                     dt=self._SamplingPeriod
                                     )
                else:
                    if other._isgain:  # And self is not? Swap, come again
                        return other + self

                # Now, we are sure that there are no empty arrays in the
                # system matrices hence concatenation should be OK.

                adda = block_diag(self._a, other.a)
                addb = np.vstack((self._b, other.b))
                addc = np.hstack((self._c, other.c))
                addd = self._d + other.d
                return State(adda, addb, addc, addd)

            else:
                return self + transfer_to_state(other)

        # Last chance for matrices, convert to static gain matrices and add
        elif isinstance(other, (int, float)):
            return State(np.ones_like(self.d)*other,
                         dt=self._SamplingPeriod) + self

        elif isinstance(other, type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self + float(other)

            if self._shape == other.shape:
                return State(self._a,
                             self._b,
                             self._c,
                             self._d + other,
                             dt=self._SamplingPeriod)
            else:
                raise IndexError('Addition of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'.format(
                                                    self._shape, other.shape))
        else:
            raise TypeError('I don\'t know how to add a '
                            '{0} to a state representation '
                            '(yet).'.format(type(other).__name__))

    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return -self + other

    def __mul__(self, other):
        # Multiplication with a State object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float
        # 4. A shape matching numpy array

        if isinstance(other, (Transfer, State)):
            # Trivial Rejections:
            # ===================
            # Reject 'ct + dt' or 'dt + dt' with different sampling periods
            #
            # A future addition would be converting everything to the slowest
            # sampling system but that requires pretty comprehensive change.

            if not self._SamplingPeriod == other._SamplingPeriod:
                raise TypeError('The sampling periods don\'t match '
                                'so I cannot\nmultiply these systems. '
                                'If you still want to multiply them as'
                                'if they are\ncompatible, carry the data '
                                'to a compatible system model and then '
                                'multiply.'
                                )

        # Reject if the size don't match
            if not self._shape[1] == other.shape[0]:
                raise IndexError('Multiplication of systems requires '
                                 'their shape to match but the system '
                                 'shapes I got are {0} vs. {1}'.format(
                                                self._shape,
                                                other.shape))
        # ===================
            if isinstance(other, State):

                # First get the static gain case out of the way.
                if self._isgain:
                    if other._isgain:
                        return State(self.d.dot(other.d),
                                     dt=self._SamplingPeriod)
                    else:
                        return State(other.a,
                                     other.b,
                                     self.d.dot(other.c),
                                     self.d.dot(other.d),
                                     dt=self._SamplingPeriod
                                     )
                else:
                    if other._isgain:  # And self is not? Swap, come again
                        return State(self.a,
                                     self.b.dot(other.d),
                                     self.c,
                                     self.d.dot(other.d),
                                     dt=self._SamplingPeriod
                                     )

                # Now, we are sure that there are no empty arrays in the
                # system matrices hence concatenation should be OK.

                multa = block_diag(self._a, other.a)
                multa[self._a.shape[1]:, :other.a.shape[0]] = self._b.dot(
                                                                    other.c)
                multb = np.vstack((self._b.dot(other.d), other.b))
                multc = np.hstack((self._c, self._d.dot(other.c)))
                multd = self._d.dot(other.d)
                return State(multa, multb, multc, multd,
                             dt=self._SamplingPeriod)

        elif isinstance(other, Transfer):
                return self * transfer_to_state(other)
        elif isinstance(other, (int, float)):
            return self * State(np.atleast_2d(other), dt=self._SamplingPeriod)
        # Last chance for matrices, convert to static gain matrices and mult
        elif isinstance(other, type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self * State(
                            np.atleast_2d(other), dt=self._SamplingPeriod)

            if self._shape[1] == other.shape[0]:
                return self * State(other, dt=self._SamplingPeriod)
            else:
                raise IndexError('Multiplication of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'.format(
                                                    self._shape, other.shape))
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__qualname__))

    def __rmul__(self, other):
        # Notice that if other is a State or Transfer, it will be handled
        # by other's __mul__() method. Hence we only take care of the
        # right multiplication of the scalars and arrays. Otherwise
        # rejection is executed
        if isinstance(other, (int, float)):
            if self._isgain:
                return State(self.d * other, dt=self._SamplingPeriod)
            else:
                return State(self._a,
                             self._b,
                             self._c * other,
                             self._d * other,
                             dt=self._SamplingPeriod
                             )
        elif isinstance(other, type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return float(other) * self
            elif self._shape[0] == other.shape[1]:
                return State(other, dt=self._SamplingPeriod) * self
            else:
                raise IndexError('Multiplication of systems requires their '
                                 'shape to match but the system shapes '
                                 'I got are {0} vs. {1}'.format(
                                                               self._shape,
                                                               other.shape))
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__name__))

    # ================================================================
    # __getitem__ to provide input-output selection of an ss
    #
    # TODO: How to validate strides
    # ================================================================

#    def __getitem__(self, num_or_slice):
#        print('La la')

    def __setitem__(self, *args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                         'the relevant num,den or A,B,C,D attributes. '
                         'This might be\nincluded in the future though.')

    def __repr__(self):
        if self._SamplingSet == 'R':
            desc_text = '\n Continous-time state represantation\n'
        else:
            desc_text = ('Discrete-Time state representation with: '
                         'sampling time: {0:.3f} \n'
                         ''.format(float(self.SamplingPeriod)))

        if self._isgain:
            desc_text += '\n{}x{} Static Gain\n'.format(self.NumberOfOutputs,
                                                        self.NumberOfInputs)
        else:
            desc_text += ' {0} input(s) and {1} output(s)\n'.format(
                                                        self.NumberOfInputs,
                                                        self.NumberOfOutputs
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
                          float,
                          list,
                          type(np.array([0.0])),
                          type(np.array([[1]])[0, 0]))

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
                returned_abcd_list[abcd_index] = np.array([])
                None_flags[abcd_index] = True
                continue

            # Check for obvious choices
            if not isinstance(abcd, possible_types):
                raise TypeError('{0} matrix should be, regardless of the shape'
                                ', an int, float, list or,\n'
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
                raise ValueError('B matrix must have the same number of '
                                 'rows with A matrix. I need {:d} but '
                                 'got {:d}.'.format(a.shape[0], b.shape[0]))

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
                raise ValueError('D matrix must have the same number of'
                                 'rows/columns \nwith C/B matrices. I '
                                 'need the shape ({0[0]:d},{0[1]:d}) '
                                 'but got ({1[0]:d},{1[1]:d}).'
                                 ''.format((c.shape[0], b.shape[1]), d.shape))

            return a, b, c, d, user_shape, Gain_flag
        else:
            return a, b, c, d, d.shape, Gain_flag


def state_to_transfer(*state_or_abcd, output='system'):
    """
    Given a State() object of a tuple of A,B,C,D array-likes, converts
    the argument into the transfer representation. The output can be
    selected as a Transfer() object or the numerator, denominator if
    'output' keyword is given with the option 'polynomials'.

    If the input is a Transfer() object it returns the argument with no
    modifications.


    The algorithm is to first get the minimal realization of the State()
    representation. Then implements the conversion ala Varga,Sima 1981
    which can be summarized as iterating over every row/cols of B and C
    to get SISO Transfer representations via c*(sI-A)^(-1)*b+d

    Parameters
    ----------
    state_or_abcd : State() or a tuple of A,B,C,D matrices.
    output : {'system','polynomials'}
        Selects whether a State() object or individual numerator, denominator
        will be returned.

    Returns
    -------
    G : Transfer()
        If ``output`` keyword is set to 'system'

    num : {List of lists of 2D-numpy arrays for MIMO case,
              2D-Numpy arrays for SISO case}
        If the ``output`` keyword is set to ``polynomials``

    den : Same as num

    """
    # FIXME : Resulting TFs are not minimal per se. simplify them, maybe?

    if output not in ('system', 'polynomials'):
        raise ValueError('The output can either be "system" or "matrices".\n'
                         'I don\'t know any option as "{0}"'.format(output))

    # If a discrete time system is given this will be modified to the
    # SamplingPeriod later.
    ZR = None
    system_given, validated_matrices = _state_or_abcd(state_or_abcd[0], 4)

    if system_given:
        A, B, C, D = state_or_abcd[0].matrices
        p, m = state_or_abcd[0].shape
        it_is_gain = state_or_abcd[0]._isgain
        ZR = state_or_abcd[0].SamplingPeriod
    else:
        A, B, C, D, (p, m), it_is_gain = State.validate_arguments(
                                                    *validated_matrices)

    if it_is_gain:
        return Transfer(D)

    if A.size == 0:
        if output is 'polynomials':
            return D, np.ones_like(D)
        return Transfer(D, np.ones_like(D), ZR)

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
            # zz might contain noisy imaginary numbers but since
            # the result should be a real polynomial, we can get
            # away with it (on paper)

            zz = transmission_zeros(A, b, c, np.array([[0]]))

            # For finding k of a G(s) we compute
            #          pole polynomial evaluated at s0
            # G(s0) * ---------------------------------
            #          zero polynomial evaluated at s0
            # s0 : some point that is not a pole or a zero

            # Additional *2 are just some tolerances

            if zz.size != 0:
                s0 = max(np.max(np.abs(np.real(np.hstack((pp, zz))))), 1)*2
            else:
                s0 = max(np.max(np.abs(np.real(pp))), 1.0)*2

            CAB = c.dot(np.linalg.lstsq((s0*np.eye(n)-A), b)[0])
            if np.size(zz) != 0:
                zero_prod = np.real(np.prod(s0*np.ones_like(zz) - zz))
            else:
                zero_prod = 1.0  # Not zero!

            pole_prod = np.real(np.prod(s0 - pp))

            entry_gain = (CAB*pole_prod/zero_prod).flatten()

            # Now, even if there are no zeros (den x DC gain) becomes
            # the new numerator hence endless fun there

            dentimesD = D[rowind, colind] * entry_den
            if zz.size == 0:
                entry_num = entry_gain
            else:
                entry_num = np.real(haroldpoly(zz))
                entry_num = np.convolve(entry_gain, entry_num)

            entry_num = haroldpolyadd(entry_num, dentimesD)
            num_list[rowind][colind] = np.array(entry_num)

    # Strip SISO result from List of list and return as arrays.
    if (p, m) == (1, 1):
        num_list = num_list[0][0]
        den_list = den_list[0][0]

    if output is 'polynomials':
        return (num_list, den_list)
    return Transfer(num_list, den_list, ZR)


def transfer_to_state(*tf_or_numden, output='system'):
    """
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
    tf_or_numden : Transfer() or a tuple of numerator and denominator.
        For MIMO numerator and denominator arguments see Transfer()
        docstring.
    output : {'system','matrices'}
        Selects whether a State() object or individual state matrices
        will be returned.

    Returns
    -------
    G : State()
        If 'output' keyword is set to 'system'
    A,B,C,D : {(nxn),(nxm),(p,n),(p,m)} 2D Numpy-arrays
        If the 'output' keyword is set to 'matrices'
    """
    if output not in ('system', 'matrices'):
        raise ValueError('The output can either be "system" or "polynomials".'
                         '\nI don\'t know any option as "{0}"'.format(output))

    # mildly check if we have a transfer,state, or (num,den)
    if len(tf_or_numden) > 1:
        num, den = tf_or_numden[:2]
        num, den, (p, m), it_is_gain = Transfer.validate_arguments(num, den)
    elif isinstance(tf_or_numden[0], State):
        return tf_or_numden[0]
    else:
        try:
            G = deepcopy(tf_or_numden[0])
            num = G.num
            den = G.den
            m, p = G.NumberOfInputs, G.NumberOfOutputs
            it_is_gain = G._isgain
        except AttributeError:
            raise TypeError('I\'ve checked the argument for being a Transfer, '
                            'a State,\nor a pair for (num,den) but'
                            ' none of them turned out to be the\ncase. Hence'
                            ' I don\'t know how to convert a {0} to a State'
                            ' object.'
                            ''.format(type(tf_or_numden[0]).__qualname__))

    # Check if it is just a gain
    if it_is_gain:
        A = np.array([], dtype=float)
        B = np.array([], dtype=float)
        C = np.array([], dtype=float)
        if np.max((m, p)) > 1:
            D = np.empty((m, p), dtype=float)
            for rows in range(p):
                for cols in range(m):
                    D[rows, cols] = num[rows][cols]/den[rows][cols]
        else:
            D = num/den

        return None, None, None, D

    if (m, p) == (1, 1):  # SISO
        A = haroldcompanion(den)
        B = np.vstack((np.zeros((A.shape[0]-1, 1)), 1))
        # num and den are now flattened
        num = haroldtrimleftzeros(num)
        den = haroldtrimleftzeros(den)

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
            # If all cancelled datanum is returned empty
            if datanum.size == 0:
                A = None
                B = None
                C = None
            else:
                C = np.zeros((1, den.size-1))
                C[0, :datanum.size] = datanum[::-1]

            D = np.atleast_2d(NumOrEmpty).astype(float)

    else:  # MIMO ! Implement a "Wolowich LMS-Section 4.4 (1974)"-variant.

        # Allocate D matrix
        D = np.zeros((p, m))

        for x in range(p):
            for y in range(m):

                # Possible cases (not minimality,only properness checked!!!):
                # 1.  3s^2+5s+3 / s^2+5s+3  Proper
                # 2.  s+1 / s^2+5s+3        Strictly proper
                # 3.  s+1 / s+1             Full cancellation
                # 4.  3   /  2              Just gains

                datanum = haroldtrimleftzeros(num[x][y].flatten())
                dataden = haroldtrimleftzeros(den[x][y].flatten())
                nn, nd = datanum.size, dataden.size

                if nd == 1:  # Case 4 : nn should also be 1.
                    D[x, y] = datanum/dataden
                    num[x][y] = np.array([0.])

                elif nd > nn:  # Case 2 : D[x,y] is trivially zero
                    pass  # D[x,y] is already 0.

                else:
                    NumOrEmpty, datanum = haroldpolydiv(datanum, dataden)
                    # Case 3: If all cancelled datanum is returned empty
                    if np.count_nonzero(datanum) == 0:
                        D[x, y] = NumOrEmpty
                        num[x][y] = np.atleast_2d([[0.]])
                        den[x][y] = np.atleast_2d([[1.]])

                    # Case 1: Proper case
                    else:
                        D[x, y] = NumOrEmpty
                        num[x][y] = datanum

                # Make the denominator entries monic
                if den[x][y][0, 0] != 1.:
                    if np.abs(den[x][y][0, 0]) < 1e-5:
                        print('transfer_to_state Warning:\n'
                              ' The leading coefficient of the ({0},{1}) '
                              'denominator entry is too small (<1e-5). '
                              'Expect some nonsense in the state space '
                              'matrices.'.format(x, y), end='\n')

                    num[x][y] = np.array([1/den[x][y][0, 0]])*num[x][y]
                    den[x][y] = np.array([1/den[x][y][0, 0]])*den[x][y]

        # OK first check if the denominator is common in all entries
        if all([np.array_equal(den[x][y], den[0][0])
                for x in range(len(den)) for y in range(len(den[0]))]):

            # Nice, less work. Off to realization. Decide rows or cols?
            if p >= m:  # Tall or square matrix => Right Coprime Fact.
                factorside = 'r'
            else:  # Fat matrix, pertranspose the List of Lists => LCF.
                factorside = 'l'
                den = [list(i) for i in zip(*den)]
                num = [list(i) for i in zip(*num)]
                p, m = m, p

            d = den[0][0].size-1
            A = haroldcompanion(den[0][0])
            B = np.vstack((np.zeros((A.shape[0]-1, 1)), 1))
            t1, t2 = A, B

            for x in range(m-1):
                A = block_diag(A, t1)
                B = block_diag(B, t2)
            n = A.shape[0]
            C = np.zeros((p, n))
            k = 0
            for y in range(m):
                for x in range(p):
                    C[x, k:k+num[x][y].size] = num[x][y]
                k += d  # Shift to the next canonical group position

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
            for x in range(m):
                lcm, mults = haroldlcm(*coldens[x])
                for y in range(p):
                    den[y][x] = lcm
                    num[y][x] = np.atleast_2d(
                                    haroldpolymul(
                                        num[y][x].flatten(), mults[y],
                                        trimzeros=False
                                    )
                                )
                    # if completely zero, then trim to single entry
                    num[y][x] = np.atleast_2d(haroldtrimleftzeros(num[y][x]))

            coldegrees = [x.size-1 for x in den[0]]

            A = haroldcompanion(den[0][0])
            B = e_i(A.shape[0], -1)

            for x in range(1, m):
                Atemp = haroldcompanion(den[0][x])
                Btemp = e_i(Atemp.shape[0], -1)

                A = block_diag(A, Atemp)
                B = block_diag(B, Btemp)

            n = A.shape[0]
            C = np.zeros((p, n))
            k = 0

            for y in range(m):
                for x in range(p):
                    C[x, k:k+num[x][y].size] = num[x][y][0, ::-1]

                k += coldegrees[y]

            if factorside == 'l':
                A, B, C = A.T, C.T, B.T

    try:  # if the arg was a Transfer object
        is_ct = tf_or_numden[0].SamplingSet is 'R'
        if is_ct:
            return (A, B, C, D) if output == 'matrices' else State(A, B, C, D)
        else:
            return (A, B, C, D) if output == 'matrices' else \
                                    State(A, B, C, D, G.SamplingPeriod)
    except AttributeError:  # the arg was num,den
        return (A, B, C, D) if output == 'matrices' else State(A, B, C, D)


def transmission_zeros(A, B, C, D):
    """
    Computes the transmission zeros of a (A,B,C,D) system matrix quartet.

    .. note:: This is a straightforward implementation of the algorithm of
              Misra, van Dooren, Varga 1994 but skipping the descriptor matrix
              which in turn becomes Emami-Naeini, van Dooren 1979.

    Parameters
    ----------
    A,B,C,D : ndarray
        The input data matrices with (nxn), (nxm), (p,n), (p,m) shapes.

    Returns
    -------
    z : ndarray
        The array of computed transmission zeros. The array is returned
        empty if no transmission zeros are found.

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

        if np.count_nonzero(np.c_[Cr, Dr]) == 0 or p != m:
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
    a, b, *_ = qz(v[:n, :n], T[:n, :n], output='complex')
    z = np.diag(b)/np.diag(a)
    for ind in range(z.size):
        z[ind] = np.real_if_close(z[ind])
    return z


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
    arguments for testing. The typical use case is to accept the arguments
    regardless of whether the input is a class instance or standalone
    matrices.

    The optional n argument is for testing state matrices less than four.
    For example, the argument should be tested for either being a State()
    object or A,B matrix for controllability. Then we select n=2 such that
    only A,B but not C,D is sought after. The default is all four matrices.

    If matrices are given, it passes the argument through the
    State.validate_arguments() method to regularize and check the sizes etc.

    Parameters
    ----------
    arg : State() or tuple of 2D Numpy arrays
        The argument to be parsed and checked for validity.
    n : integer {-1,1,2,3,4}
        If we let A,B,C,D numbered as 1,2,3,4, defines the test scope such
        that only up to n-th matrix is tested. To test only an A,C use n = -1

    Returns
    -------
    system_or_not : Boolean
        True if system and False otherwise
    validated_matrices: ndarray
        The validated n-many 2D arrays.
    """
    try:
        repr_str = arg._repr_type
        if repr_str == 'State':
            return True, None
        else:
            raise TypeError('The argument needs to be a State model '
                            'but it is a {} model'.format(repr_str))
    except AttributeError:  # It is not a model so must be tuple of matrices
        pass
    if isinstance(arg, tuple):
        system_or_not = False
        if len(arg) == n or (n == -1 and len(arg) == 2):
            z, zz = arg[0].shape
            if n == 1:
                if z != zz:
                    raise ValueError('A matrix is not square.')
                else:
                    returned_args = arg[0]
            elif n == 2:
                m = arg[1].shape[1]
                returned_args = State.validate_arguments(
                                *arg,
                                c=np.zeros((1, z)),
                                d=np.zeros((1, m))
                                )[:2]
            elif n == 3:
                m = arg[1].shape[1]
                p = arg[2].shape[0]
                returned_args = State.validate_arguments(
                                *arg,
                                d=np.zeros((p, m)))[:3]
            elif n == 4:
                m = arg[1].shape[1]
                p = arg[2].shape[0]
                returned_args = State.validate_arguments(*arg)[:4]
            else:
                p = arg[1].shape[0]
                returned_args = tuple(State.validate_arguments(
                                      arg[0],
                                      np.zeros((z, 1)),
                                      arg[1],
                                      np.zeros((p, 1))
                                      )[x] for x in [0, 2])
        else:
            raise ValueError('_state_or_abcd error:\n'
                             'Not enough elements in the argument to test.'
                             'Maybe you forgot to modify the n value?')
    else:
        raise TypeError('The argument is neither a tuple of matrices nor '
                        'a State() object. The argument is of the type "{}"'
                        ''.format(type(arg).__qualname__))

    return system_or_not, returned_args


def concatenate_state_matrices(G):
    """
    Takes a State() model as input and returns the matrix

    .. math::

        \\left[\\begin{array}{c|c}A&B\\\\ \\hline C&D\\end{array}\\right]

    Parameters
    ----------

    G : State()

    Returns
    -------

    M : 2D Numpy array

    """
    if not isinstance(G, State):
        raise TypeError('concatenate_state_matrices() works on state '
                        'representations, but I found \"{0}\" object '
                        'instead.'.format(type(G).__name__))
    H = np.vstack((np.hstack((G.a, G.b)), np.hstack((G.c, G.d))))
    return H
