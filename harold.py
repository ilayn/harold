# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:30:46 2014
Rewrite of old ltisys class file
@author: ilayn
"""


# %% License Notice
"""

The MIT License (MIT)

Copyright (c) 2015 Ilhan Polat

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

# %% Imports/shorthands
import numpy as np
import scipy as sp
from tabulate import tabulate
from scipy.signal import deconvolve
from itertools import zip_longest,chain
import collections
from copy import deepcopy

# %% Module Definitions

_KnownDiscretizationMethods = ('bilinear',
                              'tustin',
                              'zoh',
                              'forward difference',
                              'forward euler',
                              'forward rectangular',                              
                              'backward difference',
                              'backward euler',
                              'backward rectangular',                              
                              'lft','>>','<<',
                              )

# %% TF & SS classes
class Transfer:
    """
    Transfer is the one of two main system classes in harold (together 
    with State()). 

    Main types of instantiation of this class depends on whether the 
    user wants to create a Single Input/Single Output system (SISO) or
    a Multiple Input/Multiple Output system (MIMO). 
    
    For SISO system creation, 1D lists or 1D numpy arrays are expected,
    e.g.,
    
    >>>> G = Transfer([1],[1,2,1])
    
    Notice that, matlab-like scalar inputs are not recognized as 1D
    arrays hence would lead to TypeError. 
    
    For MIMO systems, depending on the shared denominators, there are 
    two distinct ways of entering a MIMO transfer function

        1-  Entering list of lists such that every element of the inner 
            lists are numpy array-able (explicitly checked) for numerator
            and entering a 1D list or 1D numpy array for denominator
            
    >>>> G = Transfer([[[1,3,2],[1,3]],[[1],[1,0]]],[1,4,5,2])
    >>>> G.shape
    (2,2)
    
        2- Entering the denominator also as a list of lists for 
        individual entries as a bracket nightmare (thanks to 
        Python's nonnative support for arrays and tedious array
        syntax). 
        
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
        
    However, the preferred way is to make everything a numpy array inside
    the list of lists. That would skip many compatibility checks. 
    Once created the shape of the numerator and denominator cannot be 
    changed. But compatible sized arrays can be supplied and it will 
    recalculate the pole/zero locations etc. properties automatically.

    The Sampling Period can be given as a last argument or a keyword 
    with 'dt' key or changed later with the property access.
    
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

    Warning: Unlike matlab or other tools, a discrete time system 
    needs a specified sampling period (and possibly a discretization 
    method if applicable) because a model without a sampling period 
    doesn't make sense for analysis. If you don't care, then make up 
    a number, say, a million, since you don't care.

    """

    def __init__(self,num,den=None,dt=False):

        # Initialization Switch and Variable Defaults

        self._isgain = False
        self._isSISO = False
        self._isstable = False
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._SamplingPeriod = False
        self._num,self._den,self._shape,self._isgain = \
                                        self.validate_arguments(num,den)
        self._p,self._m = self._shape
        if self._shape == (1,1):
            self._isSISO = True
        self.SamplingPeriod = dt
        
        self._recalc()

    @property
    def num(self): return self._num
    
    @property
    def den(self): return self._den    

    @property
    def SamplingPeriod(self): return self._SamplingPeriod
        
    @property
    def SamplingSet(self): return self._SamplingSet

    @property
    def NumberOfInputs(self): return self._m

    @property
    def NumberOfOutputs(self): return self._p

    @property
    def shape(self): return self._shape
        
    @property
    def polynomials(self): return self._num,self._den

    @property
    def DiscretizedWith(self): 
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
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                  'a discretization matrix associated with it.')
        elif not self.DiscretizedWith in ('tustin',
                                          'bilinear',
                                          'trapezoidal'):
            return ('This model is not discretized with Tustin'
                    'approximation hence prewarping does not apply.')

    @SamplingPeriod.setter
    def SamplingPeriod(self,value):
        if value:
            self._SamplingSet = 'Z'
            if type(value) is bool:# integer 1 != True
                self._SamplingPeriod = 0.
            elif isinstance(value,(int,float)):
                self._SamplingPeriod = float(value)
            else:
                raise TypeError('SamplingPeriod must be a real positive '
                                'scalar. But looks like a \"{0}\" is '
                                'given.'.format(
                                   type(value).__name__) )
        else:
            self._SamplingSet    = 'R'
            self._SamplingPeriod = None

    @num.setter
    def num(self, value):

        user_num,_,user_shape = validate_arguments(value,self._den)[:3]

        if not user_shape == self._shape:
            raise IndexError('Once created, the shape of the transfer '
                            'function \ncannot be changed. I have '
                            'received a numerator with shape {0}x{1} \nbut '
                            'the system has {2}x{3}.'
                            ''.format(*user_shape+self._shape)
                            )        
        else:
            self._num = user_num
            self._recalc()

            
    @den.setter
    def den(self, value):

        user_den,user_shape = validate_arguments(self._num,value)[1:3]

        if not user_shape == self._shape:
            raise IndexError('Once created, the shape of the transfer '
                            'function \ncannot be changed. I have '
                            'received a denominator with shape {0}x{1} \nbut '
                            'the system has {2}x{3}.'
                            ''.format(*user_shape+self._shape)
                            )        
        else:
            self._den = user_den
            self._recalc()


    @DiscretizedWith.setter
    def DiscretizedWith(self,value):
        if value in _KnownDiscretizationMethods:
            if self.SamplingSet == 'R':
                raise ValueError('This model is not discretized yet '
                                'hence you cannot define a method for'
                                ' it. Discretize the model first via '
                                '\"discretize\" function.')
            else:
                if value=='lft':
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
    def DiscretizationMatrix(self,value):
        if self._DiscretizedWith == 'lft':
            self._DiscretizationMatrix =  np.array(value,dtype='float')
        else:
            raise ValueError('If the discretization method is not '
                             '\"lft\" then you don\'t need to set '
                             'this property.')
            

    @PrewarpFrequency.setter
    def PrewarpFrequency(self,value):
        if not self._DiscretizedWith in ('tustin','bilinear','trapezoidal'):
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
                                 '{1} Hz.'.format(dt,1/(2*dt))
                                 )
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
                self.poles = np.linalg.eigvals(haroldcompanion(self._den))
                if self._num.size == 1:
                    self.zeros = np.array([])
                else:
                    self.zeros = np.linalg.eigvals(haroldcompanion(self._num))
            else:
                # Create a dummy statespace and check the zeros there
                zzz = transfertostate(self._num,self._den,output='matrices')
                self.zeros = transmission_zeros(*zzz)
                self.poles = np.linalg.eigvals(zzz[0])
        
        self._set_stability()

    def _set_stability(self):
        if self._SamplingSet == 'Z':
            self._isstable = all(1>abs(self.poles))
        else:
            self._isstable = all(0>np.real(G.poles))

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
            
        return Transfer(newnum,self._den,self._SamplingPeriod)

    def __add__(self,other):
        # Addition to a Transfer object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float that is multiplied with a proper "ones" matrix
        # 4. A shape matching numpy array
    
        # Notice that in case 3 it is a ones matrix not an identity!!
        # (Given a 1x3 system + 5) adds [[5,5,5]]. 

        if isinstance(other,(Transfer,State)):
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
            if isinstance(other,Transfer):
                # First get the static gain case out of the way.
                if self._isgain and other._isgain:
                        return Transfer(self._num + other.num, 
                                             dt = self._SamplingPeriod)
            
                # Now, we are sure that there are no possibility other than
                # list of lists or np.arrays hence concatenation should be OK. 
    
                if self._isSISO:
                    lcm,mults = haroldlcm(self._den,other.den)
                    newnum = haroldpolyadd(
                        np.convolve(self._num.flatten(),mults[0]),
                        np.convolve(other.num.flatten(),mults[1]))
                    if np.count_nonzero(newnum) == 0:
                        return Transfer(0,1)
                    else:
                        return Transfer(newnum,lcm)

                else:
                    # Create empty num and den holders.                        
                    newnum = [[None]*self._m for n in range(self._p)]
                    newden = [[None]*self._m for n in range(self._p)]
                    nonzero_num = np.zeros(self._shape,dtype=bool)
                    # Same as SISO but over all rows/cols
                    for row in range(self._p):
                        for col in range(self._m):
                            lcm,mults = haroldlcm(
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
                                nonzero_num[row,col] = True
                            
                    if any(nonzero_num.ravel()):
                        return Transfer(newnum,newden,dt=self._SamplingPeriod)
                    else:
                        # Numerators all cancelled to zero hence 0-gain MIMO
                        return Transfer(np.zeros(self._shape).tolist())
            else:
                return other + transfertostate(self)
    
        # Last chance for matrices, convert to static gain matrices and add
        elif isinstance(other,(int,float)):
            return Transfer((other * np.ones(self._shape)).tolist(),
                             dt = self._SamplingPeriod) + self

        elif isinstance(other,type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self + float(other)
            
            if self._shape == other.shape:
                return self + Transfer(other,dt= self._SamplingPeriod)
            else:
                raise IndexError('Addition of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                    self._shape,other.shape))  
        else:
            raise TypeError('I don\'t know how to add a '
                            '{0} to a state representation '
                            '(yet).'.format(type(other).__name__))


    def __radd__(self,other): return self + other

    def __sub__(self,other): return self + (-other)
        
    def __rsub__(self,other): return -self + other

    def __mul__(self,other):
        # Multiplication with a Transfer object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float 
        # 4. A shape matching numpy array
    

        if isinstance(other,(Transfer,State)):
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
                                
            if isinstance(other,Transfer):

                # First get the static gain case out of the way.
                if self._isgain and other._isgain:
                        return State(self.num.dot(other.num), 
                                             dt = self._SamplingPeriod)

                if self._isSISO:
                    return Transfer(
                            haroldpolymul(self._num.flatten(),
                                          other.num.flatten()),
                            haroldpolymul(self._den.flatten(),
                                          other.den.flatten()),
                            dt = self._SamplingPeriod)
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
                            t_G = Transfer(0,1)
                            
                            # for all elements in row/col multiplication
                            for elem in range(t_k):

                                t_num = haroldpolymul(self._num[row][elem],
                                                      other.num[elem][col])

                                t_den = haroldpolymul(self._den[row][elem],
                                                      other.den[elem][col])
                                
                                t_G += Transfer(t_num,t_den)
                            # Add the resulting arrays to the containers.
                            newnum[row][col] = t_G.num
                            newden[row][col] = t_G.den

                    # If the resulting shape is SISO, strip off the lists
                    if (t_p,t_m) == (1,1):
                        newnum = newnum[0][0]
                        newden = newden[0][0]


                    # Finally return the result. 
                    return Transfer(newnum,newden,dt=self._SamplingPeriod)

            elif isinstance(other,State):
                    return transfertostate(self) * other
                    
        elif isinstance(other,(int,float)):
            return self * Transfer(np.atleast_2d(other),
                                   dt = self._SamplingPeriod)
                
                
        # Last chance for matrices, convert to static gain matrices and mult
        elif isinstance(other,type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self * Transfer(
                            np.atleast_2d(other),dt = self._SamplingPeriod)
            
            if self._shape[1] == other.shape[0]:
                return self * Transfer(other,dt= self._SamplingPeriod)
            else:
                raise IndexError('Multiplication of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                    self._shape,other.shape))  
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__name__))        
        

    def __rmul__(self,other):
        # Notice that if other is a State or Transfer, it will be handled 
        # by other's __mul__() method. Hence we only take care of the 
        # right multiplication of the scalars and arrays. Otherwise 
        # rejection is executed
        if isinstance(other,(int,float)):
            if self._isSISO:
                return Transfer(other,dt = self._SamplingPeriod) * self
            else:                    
                return Transfer(np.ones((self._shape))*other,
                                dt = self._SamplingPeriod) * self
        elif isinstance(other,type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return float(other) * self
            elif self._shape[0] == other.shape[1]:
                return Transfer(other,dt= self._SamplingPeriod) * self
            else:
                raise IndexError('Multiplication of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                    self._shape,other.shape))
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

    def __setitem__(self,*args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                        'the relevant num,den or a,b,c,d properties. '
                        'This might be\nincluded in the future though.')


    # ================================================================
    # __repr__ and __str__ to provide meaningful info about the system
    # The ascii art of matlab for tf won't be implemented.
    # Either proper image with proper superscripts or numbers.
    # ================================================================

    def __repr__(self):
        if self.SamplingSet=='R':
            desc_text = 'Continous-Time Transfer function with:\n'
        else:
            desc_text = ('Discrete-Time Transfer function with: '
                        'sampling time: {0:.3f} \n'.format(
                                              float(self.SamplingPeriod)
                                              )
                                        )
                        
        if self._isgain:
            desc_text += '\nStatic Gain\n' 
        else:
            desc_text += ' {0} input(s) and {1} output(s)\n'.format(
                                                        self.NumberOfInputs,
                                                        self.NumberOfOutputs
                                                        )                  
    
            pole_zero_table=  zip_longest(np.real(self.poles),np.imag(self.poles),
                                          np.real(self.zeros),np.imag(self.zeros)
                                          )
            
            desc_text += '\n' + tabulate(pole_zero_table,
                                         headers=['Poles(real)',
                                                  'Poles(imag)',
                                                  'Zeros(real)',
                                                  'Zeros(imag)']
                                        )

        desc_text += '\n\n'+'End of {0} object description'.format(
                                                        __class__.__qualname__
                                                        )
        return desc_text
        

    @staticmethod
    def validate_arguments(num,den,verbose=False):
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
        num , den : The polynomial coefficient containers. Etiher of them
                    can be (not both) None to assume that the context will
                    be derived from the other for static gains. Otherwise
                    both are expected to be one of 
                    
                    np.array, int , float , list , 
                    list of lists of lists or numpy arrays. 
            
                    For MIMO context, element numbers and causality
                    checks are performed such that numerator list of 
                    list has internal arrays that have less than or 
                    equal to the internal arrays of the respective 
                    denominator entries. 
                    
                    For SISO context, causality check is performed 
                    between numerator and denominator arrays.
                    
        verbose   : boolean switch to print out what this method thinks
                    about the argument context. 
                    
            
        Returns
        -------
    
        num, den : {(m,p),(m,p)} list of lists of 2D numpy arrays (MIMO)
                   {(1,s),(1,r)} 2D numpy arrays (SISO)
                   
                   m,p integers are the shape of the MIMO system
                   r,s integers are the degree of the SISO num,den
            
            
        shape    : 2-tuple
                    Returns the recognized shape of the system

        Gain_flag: 
                    Returns True if the system is recognized as a static 
                    gain False otherwise (for both SISO and MIMO)

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
            if isinstance(arg,list):
                shape = (len(arg),len(arg[0]))
            else:
                shape = (1,1)
            return shape

        
        # A list for storing the regularized entries for num and den
        returned_numden_list = [[],[]]

        # Text shortcut for the error messages
        entrytext = ('numerator','denominator')
        
        # Booleans for Nones
        None_flags = [False,False]
        
        # Booleans for Nones
        Gain_flags = [False,False]
        
        # A boolean list that holds the recognized MIMO/SISO context 
        # for the numerator and denominator respectively.
        # True --> MIMO, False --> SISO
        MIMO_flags = [False,False]
        
        
        for numden_index,numden in enumerate((num,den)):
        # Get the SISO/MIMO context for num and den.
            if verbose: 
                print('='*40)
                print('Handling {0}'.format(entrytext[numden_index]))
                print('='*40)
            # If obviously static gain, don't bother with the rest
            if numden is None:
                if verbose: print('I found None')
                None_flags[numden_index] = True
                Gain_flags[numden_index] = True
                continue

            # Start with MIMO possibilities first
            if isinstance(numden,list):
                if verbose: print('I found a list')
                # OK, it is a list then is it a list of lists? 
                if all([isinstance(x,list) for x in numden]):
                    if verbose: print('I found a list that has only lists')

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
                                   np.atleast_2d(np.array(x,dtype='float')) 
                                   for x in y
                                 ] 
                              for y in numden
                            ]
                        except:
                            raise ValueError(# something was not float
                            'Something is not a \"float\" inside the MIMO '
                            '{0} list of lists.'
                            ''.format(entrytext[numden_index]))
                            
                    else:
                        raise IndexError( 
                        'MIMO {0} lists have inconsistent\n'
                        'number of entries, I\'ve found {1} element(s) '
                        'in one row and {2} in another row'
                        ''.format(entrytext[numden_index]),max(m),min(m))
                        
                # We found the list and it wasn't a list of lists.
                # Then it should be a regular list to be np.array'd
                elif all([isinstance(x,(int,float)) for x in numden]):
                    if verbose: print('I found a list that has only scalars')
                    try:
                        returned_numden_list[numden_index] = np.atleast_2d(
                                            np.array(numden,dtype='float')
                                            )
                        if numden_index == 1:
                            Gain_flags[1] = True
                    except:
                        raise ValueError(# something was not float
                            'Something is not a \"float\" inside the '
                            '{0} list.'
                            ''.format(entrytext[numden_index]))


            # Now we are sure that there is no dynamic MIMO entry.
            # The remaining possibility is a np.array as a static 
            # gain for being MIMO. The rest is SISO.
            # Disclaimer: We hope that the data type is 'float' 
            # Life is too short to check everything.

            elif isinstance(numden,type(np.array([0.]))):
                if verbose: print('I found a numpy array')
                if numden.ndim > 1 and min(numden.shape) > 1:
                    if verbose: print('The array has multiple elements')
                    returned_numden_list[numden_index] = [
                        [np.array(x,dtype='float') for x in y] 
                        for y in numden.tolist()
                        ]
                    MIMO_flags[numden_index] = True
                    Gain_flags[numden_index] = True
                else:
                    returned_numden_list[numden_index] = np.atleast_2d(numden)

            # OK, finally check whether and int or float is given
            # as an entry of a SISO Transfer. 
            elif isinstance(numden,(int,float)):
                if verbose: print('I found only a float')
                returned_numden_list[numden_index]=np.atleast_2d(float(numden))
                Gain_flags[numden_index] = True
                
            # Neither list of lists, nor lists nor int,floats
            # Reject and complain 
            else:
                raise TypeError(
                '{0} must either be a list of lists (MIMO)\n'
                'or a an unnested list (SISO). Numpy arrays, or, scalars' 
                ' inside unnested lists such as\n [3] are also '
                'accepted as SISO. See the \"Transfer\" docstring.'
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
            if verbose: print('Both MIMO flags are true')
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
            
            noncausal_flat_indices = [ind for ind, (x,y) 
                    in enumerate(zip(num_list,den_list)) if x.size > y.size]

            noncausal_entries = [(x // shape[0], x % shape[1]) for x in 
                                                    noncausal_flat_indices]
            if not noncausal_entries == []:
                entry_str = ['Row {0}, Col {1}'.format(x[0],x[1]) for x in 
                                                            noncausal_entries]
                                            
                raise ValueError('The following entries of numerator and '
                                 'denominator lead\nto noncausal transfers'
                                 '. Though I appreaciate the sophistication'
                                 '\nI don\'t touch descriptor stuff yet.'
                                 '\n{0}'.format('\n'.join(entry_str)))
            


        # If any of them turned out to be MIMO (ambiguous case)
        elif any(MIMO_flags):
            if verbose: print('One of the MIMO flags are true')
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
                if verbose: print('Denominator is MIMO, Numerator '
                                    'is something else')
                # numerator None? 
                if None_flags[0]:
                    if verbose: print('Numerator is None')
                    # Then create a compatible sized ones matrix and 
                    # convert it to a MIMO list of lists.

                    # Ones matrix converted to list of lists
                    num_ones = np.ones(
                                (len(MIMO_flagged),len(MIMO_flagged[0]))
                                ).tolist()
                    
                    
                    # Now make all entries 2D numpy arrays
                    # Since Num is None we can directly start adding
                    for row in num_ones:
                        returned_numden_list[0] += [
                                [np.atleast_2d(float(x)) for x in row]
                                ]

                # Numerator is SISO
                else:
                    if verbose: print('Denominator is MIMO, '
                                        'Numerator is SISO')
                    # We have to check noncausal entries                     
                    # flatten den list of lists and compare the size 
                    num_deg = haroldtrimleftzeros(returned_numden_list[0]).size

                    flattened_den = sum(returned_numden_list[1],[])

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
                    kroneckered_num = np.empty((den_shape[0],0)).tolist()
                    
                    for x in range(den_shape[0]):
                        for y in range(den_shape[1]):
                            kroneckered_num[x].append(
                                    returned_numden_list[0].copy()
                                    )
                    returned_numden_list[0] = kroneckered_num

            # Case 1,2                
            else:
                if verbose: print('Numerator is MIMO, '
                                    'Denominator is something else')
                # denominator None? 
                if None_flags[1]:
                    if verbose: 
                        print('Numerator is a static gain matrix')
                        print('Denominator is None')
                    
                    # This means num can only be a static gain matrix
                    flattened_num = sum(returned_numden_list[0],[])
                    noncausal_entries = [flattened_num[x].size < 2 
                                          for x in range(len(flattened_num))]
                                         
                    nc_entry = -1
                    try:
                        nc_entry = noncausal_entries.index(False)
                    except:
                        Gain_flags = [True,True]

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
                    if verbose: print('Numerator is MIMO, Denominator is SISO')
                    # We have to check noncausal entries                     
                    # flatten den list of lists and compare the size 
                    den_deg = haroldtrimleftzeros(returned_numden_list[1]).size
                    
                    flattened_num = sum(returned_numden_list[0],[])
                    
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
                    kroneckered_den = np.empty((num_shape[0],0)).tolist()
                    
                    for x in range(num_shape[0]):
                        for y in range(num_shape[1]):
                            kroneckered_den[x].append(
                                    returned_numden_list[1].copy()
                                    )
                    returned_numden_list[1] = kroneckered_num

                    
        # Finally if both turned out be SISO !
        else:
            if verbose: print('Both are SISO')
            if any(None_flags):
                if verbose: print('Something is None')
                if None_flags[0]:
                    if verbose: print('Numerator is None')
                    returned_numden_list[0] = np.atleast_2d([1.0])
                else:
                    if verbose: print('Denominator is None')
                    returned_numden_list[1] = np.atleast_2d([1.0])
                    Gain_flags = [True,True]

            if returned_numden_list[0].size > returned_numden_list[1].size:
                raise ValueError('Noncausal transfer functions are not '
                                'allowed.')
            

        [num , den] = returned_numden_list
        
        shape = get_shape_from_arg(num)
        
        

        # TODO : Gain_flags are not robust, remove them and make the
        # check below be decisive

        # Final gateway for the static gain
        if isinstance(den,list):
            # Check the max number of elements in each entry
            max_deg_of_den = max([x.size for x in sum(den,[])])

            # If less than two, then den is a gain matrix.
            Gain_flag = True if max_deg_of_den == 1 else False
        else:
            Gain_flag = True if den.size == 1 else False
            

        return num , den , shape , Gain_flag
        

#=======================================
#=======================================
# End of Transfer Class
#=======================================
#=======================================
       
class State:
    """
    
    State() is the one of two main system classes in harold (together with
    Transfer() ). 
    
    A State object can be instantiated in a straightforward manner by 
    entering 2D arrays. 
    
    >>>> G = State([[0,1],[-4,-5]],[[0],[1]],[[1,0]],[1])
    
    
    However, the preferred way is to make everything a numpy array.
    That would skip many compatibility checks. Once created the shape 
    of the numerator and denominator cannot be changed. But compatible 
    sized arrays can be supplied and it will recalculate the pole/zero 
    locations etc. properties automatically.

    The Sampling Period can be given as a last argument or a keyword 
    with 'dt' key or changed later with the property access.
    
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

    def __init__(self,a,b=None,c=None,d=None,dt=False):

        self._SamplingPeriod = False
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._isSISO = False
        self._isgain = False
        self._isstable = False
        

        *abcd , self._shape , self._isgain = self.validate_arguments(a,b,c,d)

        self._a , self._b , self._c , self._d = abcd
        self._p , self._m = self._shape
        
        if self._shape == (1,1):
            self._isSISO = True

        self.SamplingPeriod = dt
        self._recalc()


    @property
    def a(self): return self._a
    
    @property
    def b(self): return self._b

    @property
    def c(self): return self._c
        
    @property
    def d(self): return self._d

    @property
    def SamplingPeriod(self): return self._SamplingPeriod
        
    @property
    def SamplingSet(self): return self._SamplingSet

    @property
    def NumberOfStates(self): return self._a.shape[0]

    @property
    def NumberOfInputs(self): return self._m

    @property
    def NumberOfOutputs(self): return self._p

    @property
    def shape(self): return self._shape
        
    @property
    def matrices(self): return self._a,self._b,self._c,self._d 

    @property
    def DiscretizedWith(self): 
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
        if self.SamplingSet == 'R':
            return ('It is a continous-time model hence does not have '
                  'a discretization matrix associated with it.')
        elif not self.DiscretizedWith in ('tustin',
                                          'bilinear',
                                          'trapezoidal'):
            return ('This model is not discretized with Tustin'
                    'approximation hence prewarping does not apply.')
        else:
            return self._PrewarpFrequency
            
    @a.setter
    def a(self,value):
        value = self.validate_arguments(
            value,
            np.zeros_like(self._b),
            np.zeros_like(self._c),
            np.zeros_like(self._d)
            )[0]
        self._a = value
        self._recalc()

    @b.setter
    def b(self,value):
        value = self.validate_arguments(
            np.zeros_like(self._a),
            value,
            np.zeros_like(self._c),
            np.zeros_like(self._d)
            )[1]
        self._b = value
        self._recalc()
            
    @c.setter
    def c(self,value):
        value = self.validate_arguments(
            np.zeros_like(self._a),
            np.zeros_like(self._b),
            value,
            np.zeros_like(self._d)
            )[2]
        self._c = value
        self._recalc()

    @d.setter
    def d(self,value):
        value = self.validate_arguments(
            np.zeros_like(self._a),
            np.zeros_like(self._b),
            np.zeros_like(self._c),
            value
            )[3]
        self._d = value
        self._recalc()

    @SamplingPeriod.setter
    def SamplingPeriod(self,value):
        if value:
            self._SamplingSet = 'Z'
            if type(value) is bool:# integer 1 != True
                self._SamplingPeriod = 0.
            elif isinstance(value,(int,float)):
                self._SamplingPeriod = float(value)
            else:
                raise TypeError('SamplingPeriod must be a real scalar.'
                                'But looks like a \"{0}\" is given.'.format(
                                   type(value).__name__) )
        else:
            self._SamplingSet    = 'R'
            self._SamplingPeriod = None


    @DiscretizedWith.setter
    def DiscretizedWith(self,value):
        if value in _KnownDiscretizationMethods:
            if self.SamplingSet == 'R':
                raise ValueError('This model is not discretized yet '
                                'hence you cannot define a method for'
                                ' it. Discretize the model first via '
                                '\"discretize\" function.')
            else:
                if value=='lft':
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
    def DiscretizationMatrix(self,value):
        if self._DiscretizedWith == 'lft':
            self._DiscretizationMatrix =  np.array(value,dtype='float')
        else:
            raise TypeError('If the discretization method is not '
                             '\"lft\" then you don\'t need to set '
                             'this property.')

    @PrewarpFrequency.setter
    def PrewarpFrequency(self,value):
        if not self._DiscretizedWith in ('tustin','bilinear','trapezoidal'):
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
                                 '{1} Hz.'.format(dt,1/(2*dt))
                                 )
           else:
               self._PrewarpFrequency = value
           

    def _recalc(self):
        if self._isgain:
            self.poles = []
            self.zeros = []
        else:
            self.zeros = transmission_zeros(self._a,self._b,self._c,self._d)
            self.poles = np.linalg.eigvals(self._a)

        self._set_stability()

    def _set_stability(self):
        if self._SamplingSet == 'Z':
            self._isstable = all(1>abs(self.poles))
        else:
            self._isstable = all(0>np.real(G.poles))


    # ===========================
    # ss class arithmetic methods
    # ===========================

    def __neg__(self):
        if self._isgain:
            return State(-self._d, dt=self._SamplingPeriod)
        else:
            newC = -1. * self._c
            return State(self._a,self._b, newC, self._d,self._SamplingPeriod)


    def __add__(self,other):
        # Addition to a State object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float that is multiplied with a proper "ones" matrix
        # 4. A shape matching numpy array
    
        # Notice that in case 3 it is a ones matrix not an identity!!
        # (Given a 1x3 system + 5) adds [[5,5,5]] to D matrix. 



        if isinstance(other,(Transfer,State)):
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

                                
            if isinstance(other,State):

                # First get the static gain case out of the way.
                if self._isgain:
                    if other._isgain:
                        return State(self.d + other.d, 
                                             dt = self._SamplingPeriod)
                    else:
                        return State(other.a,
                                     other.b,
                                     other.c,
                                     self.d + other.d, 
                                     dt = self._SamplingPeriod
                                     )
                else:
                    if other._isgain: # And self is not? Swap, come again
                        return other + self
            
            
                # Now, we are sure that there are no empty arrays in the 
                # system matrices hence concatenation should be OK. 

                adda = blockdiag(self._a,other.a)
                addb = np.vstack((self._b,other.b))
                addc = np.hstack((self._c,other.c))
                addd = self._d + other.d
                return State(adda,addb,addc,addd)

            else:
                return self + transfertostate(other)

        # Last chance for matrices, convert to static gain matrices and add
        elif isinstance(other,(int,float)):
            return State(np.ones_like(self.d)*other,
                             dt = self._SamplingPeriod) + self

        elif isinstance(other,type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self + float(other)
            
            if self._shape == other.shape:
                return State(self._a,
                             self._b,
                             self._c,
                             self._d + other,
                             dt= self._SamplingPeriod)
            else:
                raise IndexError('Addition of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                    self._shape,other.shape))  
        else:
            raise TypeError('I don\'t know how to add a '
                            '{0} to a state representation '
                            '(yet).'.format(type(other).__name__))
    
    def __radd__(self,other): return self + other

    def __sub__(self,other):  return self + (-other)
        
    def __rsub__(self,other): return -self + other

    def __mul__(self,other):
        # Multiplication with a State object is possible via four types
        # 1. Another shape matching State()
        # 2. Another shape matching Transfer()
        # 3. Integer or float 
        # 4. A shape matching numpy array
    


        if isinstance(other,(Transfer,State)):
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
                                
            if isinstance(other,State):

                # First get the static gain case out of the way.
                if self._isgain:
                    if other._isgain:
                        return State(self.d.dot(other.d), 
                                             dt = self._SamplingPeriod)
                    else:
                        return State(other.a,
                                     other.b,
                                     self.d.dot(other.c),
                                     self.d.dot(other.d), 
                                     dt = self._SamplingPeriod
                                     )
                else:
                    if other._isgain: # And self is not? Swap, come again
                        return State(self.a,
                                     self.b.dot(other.d),
                                     self.c,
                                     self.d.dot(other.d), 
                                     dt = self._SamplingPeriod
                                     )


                # Now, we are sure that there are no empty arrays in the 
                # system matrices hence concatenation should be OK. 

                multa = blockdiag(self._a,other.a)
                multa[self._a.shape[1]:,:other.a.shape[0]] = self._b.dot(
                                                                    other.c)
                multb = np.vstack((self._b.dot(other.d),other.b))
                multc = np.hstack((self._c,self._d.dot(other.c)))
                multd = self._d.dot(other.d)
                return State(multa,multb,multc,multd,dt=self._SamplingPeriod)

        elif isinstance(other,Transfer):
                return self * transfertostate(other)
        elif isinstance(other,(int,float)):
            return self * State(np.atleast_2d(other),dt = self._SamplingPeriod)
        # Last chance for matrices, convert to static gain matrices and mult
        elif isinstance(other,type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return self * State(
                            np.atleast_2d(other),dt = self._SamplingPeriod)
            
            if self._shape[1] == other.shape[0]:
                return self * State(other,dt= self._SamplingPeriod)
            else:
                raise IndexError('Multiplication of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                    self._shape,other.shape))  
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__qualname__))


    def __rmul__(self,other):
        # Notice that if other is a State or Transfer, it will be handled 
        # by other's __mul__() method. Hence we only take care of the 
        # right multiplication of the scalars and arrays. Otherwise 
        # rejection is executed
        if isinstance(other,(int,float)):
            if self._isgain:
                return State(self.d * other, dt = self._SamplingPeriod)
            else:
                return State(self._a,
                             self._b,
                             self._c * other,
                             self._d * other, 
                             dt = self._SamplingPeriod
                             )
        elif isinstance(other,type(np.array([0.]))):
            # It still might be a scalar inside an array
            if other.size == 1:
                return float(other) * self
            elif self._shape[0] == other.shape[1]:
                return State(other,dt= self._SamplingPeriod) * self
            else:
                raise IndexError('Multiplication of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                    self._shape,other.shape))
        else:
            raise TypeError('I don\'t know how to multiply a '
                            '{0} with a state representation '
                            '(yet).'.format(type(other).__name__))

    # ================================================================
    # __getitem__ to provide input-output selection of an ss
    #
    # TODO: How to validate strides
    # ================================================================

    def __getitem__(self,num_or_slice):
        print('La la')

    def __setitem__(self,*args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                        'the relevant num,den or A,B,C,D attributes. '
                        'This might be\nincluded in the future though.')




        
    def __repr__(self):
        if self._SamplingSet=='R':
            desc_text = '\n Continous-time state represantation\n'
        else:
            desc_text = ('Discrete-time state represantation with: '
                  'sampling time: %.3f \n'% self.SamplingPeriod)
        
        desc_text += (' {0} input(s) and {1} output(s)\n'.format(
                                                    self.NumberOfInputs,
                                                    self.NumberOfOutputs))
        poles_real_part = np.real(self.poles)
        poles_imag_part = np.imag(self.poles)
        zeros_real_part = np.real(self.zeros)
        zeros_imag_part = np.imag(self.zeros)
        pole_zero_table=  zip_longest(
                                                poles_real_part,
                                                poles_imag_part,
                                                zeros_real_part,
                                                zeros_imag_part
                                                )
        
        desc_text += '\n' + tabulate(pole_zero_table,headers=['Poles(real)',
                                                'Poles(imag)',
                                                'Zeros(real)',
                                                'Zeros(imag)'])
#        desc_text += '\n\n'+str('End of object description') 
#                                                % __class__.__qualname__
        return desc_text


    @staticmethod
    def validate_arguments(a,b,c,d,verbose=False):
        """
        
        An internal command to validate whether given arguments to a
        State() instance are valid and compatible.
        
        It also checks if the lists are 2D numpy.array'able entries.
        
        """
        
        # A list for storing the regularized entries for a,b,c,d (mutable)
        returned_abcd_list = [[],[],[],[]]

        # Text shortcut for the error messages
        entrytext = ('A','B','C','D')
        
        # Booleans for Nones
        None_flags = [False,False,False,False]
        
        Gain_flag = False
        
        # Compared to the Transfer() inputs, State() can have relatively
        #saner inputs which is one of the following types, hence the var
        possible_types = (int,
                          float,
                          list,
                          type(np.array([0.0])),
                          type(np.array([[1]])[0,0]))

        # Start regularizing the input regardless of the intention
        for abcd_index , abcd in enumerate((a,b,c,d)):
            if verbose: 
                print('='*40)
                print ('Handling {0}'.format(entrytext[abcd_index]))
                print('='*40)
            # User supplied it? if no then don't bother further parsing.
            if abcd is None:
                if verbose: print('{0} is None'.format(entrytext[abcd_index]))
                returned_abcd_list[abcd_index] = np.array([])
                None_flags[abcd_index] = True
                continue
            

            # Check for obvious choices
            
            if not isinstance(abcd,possible_types):
                raise TypeError('{0} matrix should be, regardless of the shape,'
                                ' an int, float, list or,\n'
                                'much better, a properly typed 2D Numpy '
                                'array. Instead I found a {1} object.'.format(
                                    entrytext[abcd_index] ,
                                    type(abcd).__qualname__
                                    )
                                )

            else:
                # Row/column consistency is checked by numpy 
                try:
                    if verbose: 
                        print('Trying to np.array {0}'
                                      ''.format(entrytext[abcd_index]))
                                      
                    returned_abcd_list[abcd_index] = np.atleast_2d(
                                                np.array(abcd,dtype='float')
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
            if verbose: print('I decided that this is a gain')
            returned_abcd_list = list(reversed(returned_abcd_list))
            Gain_flag = True

        [a , b , c , d] = returned_abcd_list
        
        if not Gain_flag:
            #  Here check everything is compatible unless we have a 
            # static gain
            if verbose: print('All seems OK. Moving to shape mismatch check')
            if not a.shape == a.T.shape:
                raise ValueError('A matrix must be a square matrix '
                                'but I got {0}'.format(a.shape))
            
            if b.shape[0]!=a.shape[0]:
                raise ValueError('B matrix must have the same number of '
                                'rows with A matrix. I need {:d} but '
                                'got {:d}.'.format(
                                a.shape[0], b.shape[0])
                                )
        
            if c.shape[1]!=a.shape[1]:
                raise ValueError('C matrix must have the same number of '
                                'columns with A matrix.\nI need {:d} '
                                'but got {:d}.'.format(
                                a.shape[1], c.shape[1])
                                )


            user_shape = (c.shape[0],b.shape[1])    
            # To save the user from the incredibly boring d matrix typing
            # when d = 0, check if d is given
            if None_flags[3] is True:
                d = np.zeros(user_shape)
            
            if d.shape != (user_shape):
                raise ValueError('D matrix must have the same number of'
                                 'rows/columns \nwith C/B matrices. I '
                                 'need the shape ({0[0]:d},{0[1]:d}) '
                                 'but got ({1[0]:d},{1[1]:d}).'.format(
                                                (c.shape[0],b.shape[1]),
                                                 d.shape
                                                )
                                )
            return a,b,c,d,user_shape,Gain_flag
        else:
            return a,b,c,d,d.shape,Gain_flag

#=======================================
#=======================================
# End of State Class
#=======================================
#=======================================
            

# %% State <--> Transfer conversion

def statetotransfer(*state_or_abcd,output='system'):
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
        If 'output' keyword is set to 'system'
        
    num,den : {List of lists of 2D-numpy arrays for MIMO case,
              2D-Numpy arrays for SISO case}
        If the 'output' keyword is set to 'polynomials'

      
    """    

    #FIXME : Resulting TFs are not minimal per se. simplify them, maybe?
    
    if not output in ('system','polynomials'):
        raise ValueError('The output can either be "system" or "matrices".\n'
                         'I don\'t know any option as "{0}"'.format(output))

    # If a discrete time system is given this will be modified to the 
    # SamplingPeriod later.
    ZR = None
                         
    system_given, validated_matrices = _state_or_abcd(state_or_abcd[0],4)
    
    if system_given:
        A , B , C , D = state_or_abcd[0].matrices
        p , m = state_or_abcd[0].shape
        it_is_gain = state_or_abcd[0]._isgain
        ZR = state_or_abcd[0].SamplingPeriod
    else:
        A,B,C,D,(p,m),it_is_gain = State.validate_arguments(validated_matrices)
        
    if it_is_gain:
        return Transfer(D)
    
    A,B,C = minimal_realization(A,B,C)
    if A.size == 0:
        if output is 'polynomials':
            return D,np.ones_like(D) 
        return Transfer(D,np.ones_like(D),ZR)

    
    n = A.shape[0]

    p,m = C.shape[0],B.shape[1]
    n = np.shape(A)[0]
    pp = np.linalg.eigvals(A)
    
    entry_den = np.real(haroldpoly(pp))
    # Allocate some list objects for num and den entries
    num_list = [[None]*m for rows in range(p)] 
    den_list = [[entry_den]*m for rows in range(p)] 
    
    
    for rowind in range(p):# All rows of C
        for colind in range(m):# All columns of B

            b = B[:,colind:colind+1]
            c = C[rowind:rowind+1,:]
            # zz might contain noisy imaginary numbers but since 
            # the result should be a real polynomial, we can get 
            # away with it (on paper)

            zz = transmission_zeros(A,b,c,np.array([[0]]))

            # For finding k of a G(s) we compute
            #          pole polynomial evaluated at s0
            # G(s0) * ---------------------------------
            #          zero polynomial evaluated at s0
            # s0 : some point that is not a pole or a zero

            # Additional *2 are just some tolerances

            if zz.size!=0:
                s0 = max(np.max(np.abs(np.real(np.hstack((pp,zz))))),1)*2
            else:
                s0 = max(np.max(np.abs(np.real(pp))),1.0)*2 
                

            CAB = c.dot(np.linalg.lstsq((s0*np.eye(n)-A),b)[0])
            if np.size(zz) != 0:
                zero_prod = np.real(np.prod(s0*np.ones_like(zz) - zz))
            else:
                zero_prod = 1.0 # Not zero!

            pole_prod = np.real(np.prod(s0 - pp))

            
            entry_gain  = (CAB*pole_prod/zero_prod).flatten()

            # Now, even if there are no zeros (den x DC gain) becomes 
            # the new numerator hence endless fun there
            
            dentimesD = D[rowind,colind]*entry_den
            if zz.size==0:
                entry_num = entry_gain
            else:
                entry_num = np.real(haroldpoly(zz))
                entry_num = np.convolve(entry_gain,entry_num)
               
            entry_num = haroldpolyadd(entry_num,dentimesD)
            num_list[rowind][colind] =  np.array(entry_num)
            
    #Strip SISO result from List of list and return as arrays.            
    if (p,m) == (1,1):
        num_list = num_list[0][0]
        den_list = den_list[0][0]

    if output is 'polynomials':
        return (num_list,den_list) 
    return Transfer(num_list,den_list,ZR)

def transfertostate(*tf_or_numden,output='system'):
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
    
    
    if not output in ('system','matrices'):
        raise ValueError('The output can either be "system" or "polynomials".'
                         '\nI don\'t know any option as "{0}"'.format(output))
        
    # mildly check if we have a transfer,state, or (num,den)
    if len(tf_or_numden) > 1:
        num , den = tf_or_numden[:2]
        num,den,(p,m),it_is_gain = Transfer.validate_arguments(num,den)
    elif isinstance(tf_or_numden[0],State):
        return tf_or_numden[0]
    else:
        try:
            G = deepcopy(tf_or_numden[0])
            num = G.num
            den = G.den
            m,p = G.NumberOfInputs,G.NumberOfOutputs
            it_is_gain = G._isgain
        except AttributeError: 
            raise TypeError('I\'ve checked the argument for being a' 
                   ' Transfer, a State,\nor a pair for (num,den) but'
                   ' none of them turned out to be the\ncase. Hence'
                   ' I don\'t know how to convert a {0} to a State'
                   ' object.'.format(type(tf_or_numden[0]).__qualname__))


    # Check if it is just a gain
    if it_is_gain:
        empty_size = max(m,p)
        A = np.empty((empty_size,empty_size))
        B = np.empty((empty_size,m))
        C = np.empty((p,empty_size))
        D = num/den
        return A,B,C,D

    if (m,p) == (1,1): # SISO
        A = haroldcompanion(den)
        B = np.vstack((np.zeros((A.shape[0]-1,1)),1))
        num = haroldtrimleftzeros(num)
        den = haroldtrimleftzeros(den)

        # Monic denominator
        if den[0,0] != 1.:
            d = den[0,0]
            num,den = num/d,den/d

        if num.shape[1] < den.shape[1]:
            C = np.zeros((1,den.shape[1]-1))
            C[0,:num.shape[1]] = num[::-1]
            D = np.array([[0]])
        else: 
            # Watch out for full cancellation !!
            NumOrEmpty , datanum = haroldpolydiv(num.flatten(),den.flatten())
            
            # If all cancelled datanum is returned empty
            if datanum.size==0:
                A = None
                B = None
                C = None
            else:
                C = np.zeros((1,den.shape[1]-1))
                C[0,:datanum.size] = datanum[::-1]
                
            D = np.atleast_2d(NumOrEmpty).astype(float)

    else:# MIMO ! Implement a "Wolowich LMS-Section 4.4 (1974)"-variant.

        # Allocate D matrix
        D = np.zeros((p,m))

        for x in range(p):
            for y in range(m):
                
                # Possible cases (not minimality,only properness checked!!!): 
                # 1.  3s^2+5s+3 / s^2+5s+3  Proper
                # 2.  s+1 / s^2+5s+3        Strictly proper
                # 3.  s+1 / s+1             Full cancellation
                # 4.  3   /  2              Just gains

                
                datanum = haroldtrimleftzeros(num[x][y].flatten())
                dataden = haroldtrimleftzeros(den[x][y].flatten())
                nn , nd = datanum.size , dataden.size
                
                if nd == 1: # Case 4 : nn should also be 1.
                    D[x,y] = datanum/dataden
                    num[x][y] = np.array([0.])

                elif nd > nn: # Case 2 : D[x,y] is trivially zero
                    pass # D[x,y] is already 0.

                else:
                    NumOrEmpty , datanum = haroldpolydiv(datanum,dataden)
                    # Case 3: If all cancelled datanum is returned empty
                    if np.count_nonzero(datanum) == 0:
                        D[x,y] = NumOrEmpty
                        num[x][y] = np.atleast_2d([[0.]])
                        den[x][y] = np.atleast_2d([[1.]])
                        
                    # Case 1: Proper case
                    else:
                        D[x,y] = NumOrEmpty
                        num[x][y] = datanum

                # Make the denominator entries monic
                if den[x][y][0,0] != 1.:
                    if np.abs(den[x][y][0,0])<1e-5:
                        print(
                          'transfertostate Warning:\n The leading coefficient '
                          'of the ({0},{1}) denominator entry is too '
                          'small (<1e-5). Expect some nonsense in the '
                          'state space matrices.'.format(x,y),end='\n')
                          
                    num[x][y] = np.array([1/den[x][y][0,0]])*num[x][y]
                    den[x][y] = np.array([1/den[x][y][0,0]])*den[x][y]

        # OK first check if the denominator is common in all entries
        if all([np.array_equal(den[x][y],den[0][0])
            for x in range(len(den)) for y in range(len(den[0]))]):

            # Nice, less work. Off to realization. Decide rows or cols?
            if p >= m:# Tall or square matrix => Right Coprime Fact.
               factorside = 'r'
            else:# Fat matrix, pertranspose the List of Lists => LCF.
               factorside = 'l'
               den = [list(i) for i in zip(*den)]
               num = [list(i) for i in zip(*num)]
               p,m = m,p

            d = den[0][0].size-1
            A = haroldcompanion(den[0][0])
            B = np.vstack((np.zeros((A.shape[0]-1,1)),1))
            t1 , t2 = A , B

            for x in range(m-1):
                A = blockdiag(A,t1)
                B = blockdiag(B,t2)
            n = A.shape[0]
            C = np.zeros((p,n))
            k = 0
            for y in range(m):
                for x in range(p):
                    C[x,k:k+num[x][y].size] = num[x][y]
                k += d # Shift to the next canonical group position

            if factorside == 'l':
                A, B, C = A.T, C.T, B.T

        else: # Off to LCM computation
              # Get every column denominators and compute the LCM 
              # and mults then modify denominators accordingly and 
              # add multipliers to nums. 
        
            if p >= m:# Tall or square matrix => Right Coprime Fact.
               factorside = 'r'
            else:# Fat matrix, pertranspose => Left Coprime Fact.
               factorside = 'l'
               den = [list(i) for i in zip(*den)]
               num = [list(i) for i in zip(*num)]
               p,m = m,p


            coldens = [x for x in zip(*den)]
            for x in range(m):
                lcm,mults = haroldlcm(*coldens[x])
                for y in range(p):
                    den[y][x] = lcm
                    num[y][x] = np.atleast_2d(
                                    haroldpolymul(
                                        num[y][x].flatten(),mults[y],
                                        trimzeros=False
                                    )
                                )
                    # if completely zero, then trim to single entry
                    num[y][x] = np.atleast_2d(haroldtrimleftzeros(num[y][x]))

            coldegrees = [x.size-1 for x in den[0]]

            A = haroldcompanion(den[0][0])
            B = eyecolumn(A.shape[0],-1)

            for x in range(1,m):
                Atemp = haroldcompanion(den[0][x])
                Btemp = eyecolumn(Atemp.shape[0],-1)

                A = blockdiag(A,Atemp)
                B = blockdiag(B,Btemp)

            n = A.shape[0]
            C = np.zeros((p,n))
            k = 0

            for y in range(m):
                for x in range(p):
                    C[x,k:k+num[x][y].size] = num[x][y][0,::-1]

                k += coldegrees[y] 
            
            if factorside == 'l':
                A, B, C = A.T, C.T, B.T
      
    try:# if the arg was a Transfer object
        is_ct = tf_or_numden[0].SamplingSet is 'R'
        if is_ct:
            return (A,B,C,D) if output=='matrices' else State(A,B,C,D)
        else:
            return (A,B,C,D) if output=='matrices' else State(A,B,C,D,
                                                            G.SamplingPeriod)
    except AttributeError:# the arg was num,den
        return (A,B,C,D) if output=='matrices' else State(A,B,C,D)





# %% Transmission zeros of a state space system

"""

TODO Though the descriptor code also works up-to-production, I truncated 
to explicit systems. I better ask around if anybody needs them (though 
the answer to such question is always a yes).

TODO: I've tested with random systems both in Matlab and Python
Seemingly there is no problem. But it needs optimization. Also
zero dynamics systems will fail here. Find out why it is curiously 
much faster than matlab.
"""

def transmission_zeros(A,B,C,D):
    """
    Computes the transmission zeros of a (A,B,C,D) system matrix quartet. 

    This is a straightforward implementation of the algorithm of Misra, 
    van Dooren, Varga 1994 but skipping the descriptor matrix which in 
    turn becomes Emami-Naeini,van Dooren 1979. I don't know if anyone 
    actually uses descriptor systems in practice so I removed the 
    descriptor parts to reduce the clutter. Hence, it is possible to 
    directly row/column compress the matrices without caring about the 
    upper Hessenbergness of E matrix. 

    Parameters
    ----------
    A,B,C,D : {(nxn),(nxm),(p,n),(p,m) 2D Numpy arrays} 
        
    
    Returns
    -------
    z : {1D Numpy array}
        The array of computed transmission zeros. The array is returned 
        empty if no transmission zeros are found. 

      
 
    """    
    n , (p , m) = np.shape(A)[0] , np.shape(D)
    r = np.linalg.matrix_rank(D)
        
    if (p==1 and m==1 and r>0) or (r == min(p,m) and p==m):
        z = _tzeros_final_compress(A,B,C,D,n,p,m)
        return z
    else:# Reduction needed
        if r == p:
            Ar,Br,Cr,Dr = (A,B,C,D)
        else:
            Ar,Br,Cr,Dr = _tzeros_reduce(A,B,C,D)
        
        n , (p , m) = np.shape(Ar)[0] , np.shape(Dr)

        if np.count_nonzero(np.c_[Cr,Dr])==0 or p != m:
            Arc,Crc,Brc,Drc = _tzeros_reduce(Ar.T,Cr.T,Br.T,Dr.T)
            Arc,Crc,Brc,Drc = Arc.T,Crc.T,Brc.T,Drc.T
        else:
            Arc,Brc,Crc,Drc = (Ar,Br,Cr,Dr)

        n , (p , m) = np.shape(Arc)[0] , np.shape(Drc)

        if n!=0:
            z = _tzeros_final_compress(Arc,Brc,Crc,Drc,n,p,m)
        else:
            z = np.zeros((0,1))
        return z

def _tzeros_reduce(A,B,C,D):
    """
    Basic deflation loop until we get a full row rank feedthrough matrix. 
    """
    for x in range(A.shape[0]):# At most!
        n , (p , m) = np.shape(A)[0] , np.shape(D)
        u,s,v,r = haroldsvd(D,also_rank=True)
        # Do we have full rank D already? 
        if r == D.shape[0]:
            break
        
        Dt = s.dot(v)
        Ct = u.T.dot(C)[r-p:,]

        vc , mm = haroldsvd(Ct,also_rank=True)[2:]
        T = np.roll(vc.T,-mm,axis=1)
        
        Sysmat = blockdiag(T,u).T.dot(
            np.vstack((
                np.hstack((A,B)),np.hstack((C,D))
            )).dot(blockdiag(T,np.eye(m)))
            )

        Sysmat = np.delete(Sysmat,np.s_[r-p:],0)
        Sysmat = np.delete(Sysmat,np.s_[n-mm:n],1)

        A,B,C,D = matrixslice(Sysmat,(n-mm,n-mm))
        if A.size==0 or np.count_nonzero(np.c_[C,D])==0:
            break
    return A,B,C,D

        
def _tzeros_final_compress(A,B,C,D,n,p,m):
    """
    Internal command for finding the Schur form of a full rank and 
    row/column compressed C,D pair. 
    
    TODO: Clean up the numerical noise and switch to Householder maybe? 

    TODO : Rarely z will include 10^15-10^16 entries instead of 
    infinite zeros. Decide on a reasonable bound to discard.
    """     

    v = haroldsvd(np.hstack((D,C)))[-1]
    T = np.hstack((A,B)).dot(np.roll(np.roll(v.T,-m,axis=0),-m,axis=1))
    S = blockdiag(
            np.eye(n),
            np.zeros((p,m))
            ).dot(np.roll(np.roll(v.T,-m,axis=0),-m,axis=1))
    a,b = sp.linalg.qz(S[:n,:n],T[:n,:n],output='complex')[:2]
    z = np.diag(b)/np.diag(a)

    return z

# %% Continous - Discrete Conversions

def discretize(G,dt,method='tustin',PrewarpAt = 0.,q=None):
    if not isinstance(G,(Transfer,State)):
        raise TypeError('I can only convert State or Transfer objects but I '
                        'found a \"{0}\" object.'.format(type(G).__name__)
                        )
    if G.SamplingSet == 'Z':
        raise TypeError('The argument is already modeled as a '
                        'discrete-time system.')

    if isinstance(G,Transfer):
        T = transfertostate(G)
    else:
        T = G

    args = __discretize(T,dt,method,PrewarpAt,q)

    if isinstance(G,State):
        Gd = State(*args)
        Gd.DiscretizedWith = method
    else:
        Gss = State(*args)
        Gd = statetotransfer(Gss)
        Gd.DiscretizedWith = method        

    if method =='lft':
        Gd.DiscretizationMatrix = q

    elif method in ('tustin','bilinear','trapezoidal'):
        Gd.PrewarpFrequency = PrewarpAt

    return Gd
    

def __discretize(T,dt,method,PrewarpAt,q):
    """
    Actually, I think that presenting this topic as a numerical
    integration problem is confusing more than it explains. Most 
    items here can be presented as conformal mappings and nobody
    needs to be limited to riemann sums of particular shape. As 
    I found that scipy version of this, adopts Zhang SICON 2007
    parametrization which surprisingly fresh!
    
    Here I "generalized" to any rational function representation 
    if you are into that mathematician lingo (see the 'fancy'
    ASCII art below). I used LFTs instead, for all real rational 
    approx. mappings (for whoever wants to follow that rabbit).
    """


    (p,m),n = T.shape,T.NumberOfStates

    if method == 'zoh':
        """
        Zero-order hold is not much useful for linear systems and 
        in fact it should be discouraged since control problems 
        don't have boundary conditions as in stongly nonlinear 
        FEM simulations of CFDs so on. Most importantly it is not 
        stability-invariant which defeats its purpose. But whatever
        
        
        
        This conversion is usually done via the expm() identity
        
            [A | B]   [ exp(A) | int(exp(A))*B ]   [ Ad | Bd ]
        expm[- - -] = [------------------------] = [---------]
            [0 | 0]   [   0    |       I       ]   [ C  | D  ]
           
        TODO: I really want to display a warning here against 'zoh' use 
        """
        
        M = np.r_[np.c_[T.a,T.b],np.zeros((m,m+n))]
        eM = sp.linalg.expm(M*dt)
        Ad , Bd , Cd , Dd = eM[:n,:n] , eM[:n,n:] , T.c , T.d
        
    elif method == 'lft':
        """
        Here we form the following star product
                                      _
                       ---------       |
                       |  1    |       |                        
                    ---| --- I |<--    |
                    |  |  z    |  |    |                    
                    |  ---------  |    |
                    |             |    |> this is the lft of (1/s)*I
                    |   -------   |    |  
                    --->|     |----    |
                        |  Q  |        |
                    --->|     |----    |
                    |   -------   |   _|
                    |             |
                    |   -------   |
                    ----|     |<---  
                        |  T  |
                    <---|     |<---
                        -------

        Here Q is whatever the rational mapping that links s to z In 
        the floowing sense:
        
         1         1
        --- = F_u(---,Q)
         s         z
        
        where F_u denotes the upper linear fractional representation. 
        For exemaple, the usual case of Tustin, Euler etc. the map is 
        
                  [     I     |  sqrt(T)*I ]
              Q = [-----------|------------]
                  [ sqrt(T)*I |    T*x*I   ]

        with alpha defined as in Zhang 2007 SICON. 
        x = 0   --> backward diff, (backward euler)
        x = 0.5 --> Tustin,
        x = 1   --> forward difference (forward euler)

        """

        # TODO: Check if interconnection is well-posed !!!!

        if q is None:
            raise ValueError('\"lft\" method requires an interconnection '
                             'matrix. Consider providing a matrix \"q". '
                             )

        # Copy n times for n integrators
        q11 , q12 , q21 , q22 = (
                    sp.linalg.kron(np.eye(n),x) for x in 
                    ssslice(q,-1)                    
                    )

        # Compute the star product
        ZAinv = sp.linalg.solve(np.eye(n)-q22.dot(T.a),q21)
        AZinv = sp.linalg.solve(np.eye(n)-T.a.dot(q22),T.b)

        Ad = q11 + q12.dot(T.a.dot(ZAinv))
        Bd = q12.dot(AZinv)
        Cd = T.c.dot(ZAinv)
        Dd = T.d + T.c.dot(q22.dot(AZinv))
        

    elif method in ('bilinear','tustin','trapezoidal'):
        if not PrewarpAt == 0.:
            if 1/(2*dt) < PrewarpAt:
                raise ValueError('Prewarping Frequency is beyond '
                                 'the Nyquist rate.\nIt has to '
                                 'satisfy 0 < w < 1/(2*dt) and dt '
                                 'being the sampling\nperiod in '
                                 'seconds (dt={0} is provided, '
                                 'hence the max\nallowed is '
                                 '{1} Hz.'.format(dt,1/(2*dt))
                                 )
               
            PrewarpAt *= 2*np.pi               
            TwoTanw_Over_w = 2*np.tan(PrewarpAt*dt/2)/PrewarpAt
            q = np.array(
                [
                  [         1             , np.sqrt(TwoTanw_Over_w)],
                  [np.sqrt(TwoTanw_Over_w),    TwoTanw_Over_w      ]
                ])
        else:
            q = np.array(
                [
                  [    1     , np.sqrt(dt)],
                  [np.sqrt(dt),    dt/2    ]
                ])
            
        return __discretize(T,dt,"lft",0.,q)

    elif method in ('forward euler', 
                    'forward difference',
                    'forward rectangular',
                    '>>'):# pff....
        return __discretize(T, dt,"lft",0,q = np.array(
                              [
                                [    1,      np.sqrt(dt)],
                                [np.sqrt(dt),    0      ]
                              ]
                            )
                          )
                          
    elif method in ('backward euler',
                    'backward difference',
                    'backward rectangular',
                    '<<'):
        return __discretize(T, dt,"lft",0,q = np.array(
                              [
                                [    1,      np.sqrt(dt)],
                                [np.sqrt(dt),     dt    ]
                              ]
                            )
                          )

    else:
        raise ValueError('I don\'t know that discretization method. But '
                        'I know {0} methods.'
                        ''.format(_KnownDiscretizationMethods)
                        )
                        
    return Ad , Bd , Cd , Dd , dt


def undiscretize(G,OverrideWith = None):
    if not isinstance(G,(Transfer,State)):
        raise TypeError('The argument is not transfer '
        'function or a state\nspace model.'
            )

    if G.SamplingSet == 'R':
        raise TypeError('The argument is already modeled as a '
                        'continuous time system.')


    args = __undiscretize(G)

    if isinstance(G,State):
        Gc = State(*args)
    else:
        Gss = State(*args)
        Gc = statetotransfer(State(Gss))
        
    return Gc


def __undiscretize(G):

    if isinstance(G,Transfer):
        T = transfertostate(G)
    else:
        T = G

    (p,m),n = T.shape,T.NumberOfStates
    dt = G.SamplingPeriod
    
    missing_method = False
    if 'with it' in G.DiscretizedWith:# Check if warning comes back
        missing_method = True
        
    
    if G.DiscretizedWith == 'zoh':
        M = np.r_[
                   np.c_[T.a,T.b],
                   np.c_[np.zeros((m,n)),np.eye(m)]
                  ]
        eM = sp.linalg.logm(M)*(1/T.SamplingPeriod)
        Ac , Bc , Cc , Dc = eM[:n,:n] , eM[:n,n:] , T.c , T.d
        
    elif (G.DiscretizedWith in ('bilinear','tustin','trapezoidal')
            or
          missing_method# Manually created DT system
         ):
         
         X = np.eye(n)+T.a
         if 1/np.linalg.cond(X) < 1e-8: # TODO: Totally psychological limit
         
             raise ValueError('The discrete A matrix has eigenvalue(s) '
                              'very close to -1 (rcond of I+Ad is {0})'
                              ''.format(1/np.linalg.cond(X)))


         iX = 2/dt*np.linalg.inv(np.eye(n)+T.a)
         Ac = -iX.dot(np.eye(n)-T.a)
         Bc = iX.dot(T.b)
         Cc = T.c.dot(np.eye(n)+0.5*dt*iX.dot(np.eye(n)-T.a))
         Dc = T.d - 0.5*dt*T.c.dot(iX.dot(T.b))

    elif (G.DiscretizedWith in ('forward euler', 'forward difference',
                    'forward rectangular','>>')):
         iX = 1/dt*np.eye(n)
         Ac = -iX.dot(np.eye(n)-T.a)
         Bc = iX.dot(T.b)
         Cc = T.c
         Dc = T.d
         
    elif (G.DiscretizedWith in ('backward euler','backward difference',
                    'backward rectangular','<<')):
         X = T.a
         if 1/np.linalg.cond(X) < 1e-8: # TODO: Totally psychological limit
             raise ValueError('The discrete A matrix has eigenvalue(s) '
                              'very close to 0 (rcond of I+Ad is {0})'
                              ''.format(1/np.linalg.cond(X)))


         iX = 1/dt*np.linalg.inv(T.a)
         Ac = -iX.dot(np.eye(n)-T.a)
         Bc = iX.dot(T.b)
         Cc = T.c.dot(np.eye(n)+dt*iX.dot(np.eye(n)-T.a))
         Dc = T.d - dt*T.c.dot(iX.dot(T.b))

    return Ac , Bc , Cc , Dc        
        
        

def rediscretize(G,dt,method='tustin',alpha=0.5):
    pass


# %% Kalman Ops

def kalman_controllability(G,compress=False):
    """
    Computes the Kalman controllability related quantities. The algorithm
    is the literal computation of the controllability matrix with increasing
    powers of A. Numerically, this test is not robust and prone to errors if 
    the A matrix is not well-conditioned or its entries have varying order
    of magnitude as at each additional power of A the entries blow up or 
    converge to zero rapidly. 
    
    Parameters
    ----------
    G : State() or tuple of {(n,n),(n,m)} array_like matrices
        System or matrices to be tested

    compress : Boolean
        If set to True, then the returned controllability matrix is row
        compressed, and in case of uncontrollable modes, has that many
        zero rows.
        
    Returns
    -------

    Cc : {(n,nxm)} 2D numpy array
        Kalman Controllability Matrix 
    T : (n,n) 2D numpy arrays
        The transformation matrix such that T^T * Cc is row compressed 
        and the number of zero rows at the bottom corresponds to the number
        of uncontrollable modes.
    r: integer
        Numerical rank of the controllability matrix 
    
    """
    
    sys_flag,mats = _state_or_abcd(G,2)
    if sys_flag:
        A = G.a
        B = G.b
    else:
        A , B = mats
        
    n = A.shape[0]
    Cc = B.copy()
    
    for i in range(1,n):
        Cc = np.hstack((Cc,np.linalg.matrix_power(A,i).dot(B)))

    if compress:
        T,S,V,r = haroldsvd(Cc,also_rank=True)
        return S.dot(V.T) , T , r

    T,*_,r = haroldsvd(Cc,also_rank=True)
    return Cc , T , r

def kalman_observability(G,compress=False):
    """
    Computes the Kalman observability related objects. The algorithm
    is the literal computation of the observability matrix with increasing
    powers of A. Numerically, this test is not robust and prone to errors if 
    the A matrix is not well-conditioned or too big as at each additional 
    power of A the entries blow up or converge to zero rapidly. 
    
    Parameters
    ----------
    G : State() or {(n,n),(n,m)} array_like matrices
        System or matrices to be tested

    compress : Boolean
        If set to True, then the returned observability matrix is row
        compressed, and in case of unobservability modes, has that many
        zero rows.
        
    Returns
    -------

    Co : {(n,nxm)} 2D numpy array
        Kalman observability matrix 
    T : (n,n) 2D numpy arrays
        The transformation matrix such that T^T * Cc is row compressed 
        and the number of zero rows on the right corresponds to the number
        of unobservable modes.
    r: integer
        Numerical rank of the observability matrix 
    
    """    
    sys_flag , mats = _state_or_abcd(G,-1)
    
    if sys_flag:
        A = G.a
        C = G.c
    else:
        A , C = mats

    n = A.shape[0]
    Co = C.copy()

    for i in range(1,n):
        Co = np.vstack((Co,C.dot(np.linalg.matrix_power(A,i))))

    if compress:
        T,S,V,r = haroldsvd(Co,also_rank=True)
        return T.dot(S) , V.T , r

    *_, T , r = haroldsvd(Co,also_rank=True)
    return Co , T , r

def kalman_decomposition(G,compute_T=False,output='system',cleanup_threshold=1e-9):
    """
    By performing a sequence of similarity transformations the State
    representation is transformed into a special structure such that
    if the system has uncontrollable/unobservable modes, the corresponding
    rows/columns of the B/C matrices have zero blocks and the modes
    are isolated in the A matrix. That is to say, there is no contribution
    of the controllable/observable states on the dynamics of these modes.
   
   
    Note that, Kalman operations are numerically not robust. Hence the
    resulting decomposition might miss some 'almost' pole-zero cancellations.
    Hence, this should be used as a rough assesment tool but not as
    actual minimality check or maybe to demonstrate the concepts academic
    purposes to show the modal decomposition. Use canceldistance() and
    minimal_realization() functions instead with better numerical properties.

    Example usage and verification : 
    
    G = State([[2,1,1],[5,3,6],[-5,-1,-4]],[[1],[0],[0]],[[1,0,0]],0)
    print('Is it Kalman Cont\'ble ? ',is_kalman_controllable(G))
    print('Is it Kalman Obsv\'ble ? ',is_kalman_observable(G))
    F = kalman_decomposition(G)
    print(F.a,F.b,F.c,sep='\n\n')
    H = minimal_realization(F.a,F.b,F.c)
    print('\nThe minimal system matrices are:\n',*H)
    
    Expected output : 
    Is it Kalman Cont'ble ?  False
    Is it Kalman Obsv'ble ?  False
    [[ 2.          0.         -1.41421356]
     [ 7.07106781 -3.         -7.        ]
     [ 0.          0.          2.        ]]
    
    [[-1.]
     [ 0.]
     [ 0.]]
    
    [[-1.  0.  0.]]
    
    The minimal system matrices are:
     [[ 2.]] [[ 1.]] [[ 1.]]

    Parameters:
    ------
   
    G : State()
        The state representation that is to be converted into the block
        triangular form such that unobservable/uncontrollable modes
        corresponds to zero blocks in B/C matrices
        
    compute_T : boolean
        Selects whether the similarity transformation matrix will be 
        returned.
        
    output : {'system','matrices'}
        Selects whether a State() object or individual state matrices 
        will be returned.
    
    cleanup_threshold : float
        After the similarity transformation, the matrix entries smaller
        than this threshold in absolute value would be zeroed. Setting 
        this value to zero turns this behavior off. 
    
    Returns:
    --------
    Gk : State() or if output = 'matrices' is selected (A,B,C,D) tuple
        Returns a state representation or its matrices as a tuple
        
    T  : (nxn) 2D-numpy array
        If compute_T is True, returns the similarity transform matrix
        that brings the state representation in the resulting decomposed
        form such that
       
            Gk.a = inv(T)*G.a*T
            Gk.b = inv(T)*G.b
            Gk.c = G.c*T
            Gk.d = G.d

    """
    if not isinstance(G,State):
        raise TypeError('The argument must be a State() object')

    # If a static gain, then skip and return the argument    
    if G._isgain:
        if output == 'matrices':
            return G.matrices
        
        return G
    
    # TODO: This is an unreliable test anyways but at least check 
    # which rank drop of Cc, Co is higher and start from that 
    # to get a tiny improvement
    
    # First check if controllable 
    if not is_kalman_controllable(G):
        Tc , r = kalman_controllability(G)[1:]
    else:
        Tc = np.eye(G.a.shape[0])
        r = G.a.shape[0]

    
    ac = np.linalg.solve(Tc,G.a).dot(Tc)
    bc = np.linalg.solve(Tc,G.b)
    cc = G.c.dot(Tc)
    ac[ abs(ac) < cleanup_threshold ] = 0.
    bc[ abs(bc) < cleanup_threshold ] = 0.
    cc[ abs(cc) < cleanup_threshold ] = 0.

    if r == 0:
        raise ValueError('The system is trivially uncontrollable.'
                         'Probably B matrix is numerically all zeros.')
    elif r != G.a.shape[0]:
        aco , auco = ac[:r,:r] , ac[r:,r:]
        bco = bc[:r,:]
        cco , cuco = cc[:,:r] , cc[:,r:]
        do_separate_obsv = True
    else:
        aco , bco , cco = ac , bc , cc
        auco , cuco = None , None
        do_separate_obsv = False
        
    if do_separate_obsv:
        To_co = kalman_observability((aco,cco))[1]
        To_uco = kalman_observability((auco,cuco))[1]
        To = blockdiag(To_co,To_uco)
    else:
        if not is_kalman_observable((ac,cc)):
            To , r = kalman_observability((ac,cc))[1:]
        else:
            To = np.eye(ac.shape[0])
       
    A = np.linalg.solve(To,ac).dot(To)
    B = np.linalg.solve(To,bc)
    C = cc.dot(To)
    
    # Clean up the mess, if any, for the should-be-zero entries
    A[ abs(A) < cleanup_threshold ] = 0.
    B[ abs(B) < cleanup_threshold ] = 0.
    C[ abs(C) < cleanup_threshold ] = 0.
    D = G.d.copy()
    
    if output == 'matrices':
        if compute_T:
            return (A,B,C,D),Tc.dot(To)
        
        return (A,B,C,D)
    
    if compute_T:
        return State(A,B,C,D,G.SamplingPeriod),Tc.dot(To)
    
    return State(A,B,C,D,G.SamplingPeriod)

    
    
def is_kalman_controllable(G):
    """
    Tests the rank of the Kalman controllability matrix and compares it 
    with the A matrix size, returns a boolean depending on the outcome. 
    
    Parameters:
    ------
    
    G : State() or tuple of {(nxn),(nxm)} array_like matrices
        The system or the (A,B) matrix tuple    
        
    Returns:
    --------
    test_bool : Boolean
        Returns True if the input is Kalman controllable
    
    """
    sys_flag,mats = _state_or_abcd(G,2)
    if sys_flag:
        A = G.a
        B = G.b
    else:
        A , B = mats

    r = kalman_controllability((A,B))[-1]

    if A.shape[0] > r:
        return False
        
    return True
    
def is_kalman_observable(G):
    """
    Tests the rank of the Kalman observability matrix and compares it 
    with the A matrix size, returns a boolean depending on the outcome. 
    
    Parameters:
    ------
    
    G : State() or tuple of {(nxn),(pxn)} array_like matrices
        The system or the (A,C) matrix tuple    
        
    Returns:
    --------
    test_bool : Boolean
        Returns True if the input is Kalman observable
    
    """
    sys_flag , mats = _state_or_abcd(G,-1)
    
    if sys_flag:
        A = G.a
        C = G.c
    else:
        A , C = mats
            
    r = kalman_observability((A,C))[-1]

    if A.shape[0] > r:
        return False
        
    return True
        
# %% Linear algebra ops

def _state_or_abcd(arg,n=4):
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
        that only up to n-th matrix is tested. 
        
        To test only an A,C use n = -1
        
    Returns
    --------
    system_or_not : Boolean
        True if system and False otherwise
        
    validated_matrices: n-many 2D Numpy arrays
    
    """
    if isinstance(arg,tuple):
        system_or_not = False
        if len(arg) == n or (n == -1 and len(arg) == 2):
            z,zz = arg[0].shape
            if n == 1:
                if z != zz:
                    raise ValueError('A matrix is not square.')
                else:
                    returned_args = arg[0]
            elif n == 2:
                m = arg[1].shape[1]
                returned_args = State.validate_arguments(
                                *arg,
                                c = np.zeros((1,z)),
                                d = np.zeros((1,m))
                                )[:2]
            elif n == 3:
                m = arg[1].shape[1]
                p = arg[2].shape[0]
                returned_args = State.validate_arguments(
                                *arg,
                                d = np.zeros((p,m))
                                )[:3]
            elif n == 4:
                m = arg[1].shape[1]
                p = arg[2].shape[0]
                returned_args = State.validate_arguments(*arg)[:4]
            else:
                p = arg[1].shape[0]
                returned_args = tuple(State.validate_arguments(
                                arg[0],
                                np.zeros((z,1)),
                                arg[1],
                                np.zeros((p,1))
                                )[x] for x in [0,2])
        else:
            raise ValueError('_state_or_abcd error:\n'
                             'Not enough elements in the argument to test.'
                             'Maybe you forgot to modify the n value?')


    elif isinstance(arg,State):
            system_or_not = True
            returned_args = None
    else:
        raise TypeError('The argument is neither a tuple of matrices nor '
                        'a State() object.')

    return system_or_not , returned_args

def staircase(A,B,C,compute_T=False,form='c',invert=False):
    """
    The staircase form is used very often to assess system properties. 
    Given a state system matrix triplet A,B,C, this function computes 
    the so-called controller-Hessenberg form such that the resulting 
    system matrices have the block-form (x denoting the nonzero blocks)
    
                [x x x x x] |  [ x ]
                [x x x x x] |  [ 0 ]
                [0 x x x x] |  [ 0 ]
                [0 0 x x x] |  [ 0 ]
                [0 0 0 x x] |  [ 0 ]
                ------------|-------
                [x x x x x] |
                [x x x x x] |

    For controllability and observability, the existence of zero-rank 
    subdiagonal blocks can be checked, as opposed to forming the Kalman 
    matrix and checking the rank. Staircase method can numerically be 
    more stable since for certain matrices, A^n computations can 
    introduce large errors (for some A that have entries with varying 
    order of magnitudes). But it is also prone to numerical rank guessing
    mismatches.
    
    Notice that, if we use the pertransposed data, then we have the 
    observer form which is usually asked from the user to supply
    the data as A,B,C ==> A^T,C^T,B^T and then transpose back the result.
    This is just silly to ask the user to do that. Hence the additional 
    "form" option denoting whether it is the observer or the controller 
    form that is requested.
    
    
    Parameters
    ----------
    A,B,C : {(n,n),(n,m),(p,n)} array_like
        System Matrices to be converted
    compute_T : bool, optional
        Whether the transformation matrix T should be computed or not
    form : { 'c' , 'o' }, optional
        Determines whether the controller- or observer-Hessenberg form 
        will be computed. 
    invert : bool, optional
        Whether to select which side the B or C matrix will be compressed.
        For example, the default case returns the B matrix with (if any)
        zero rows at the bottom. invert option flips this choice either in
        B or C matrices depending on the "form" switch. 
        
    Returns
    -------

    Ah,Bh,Ch : {(n,n),(n,m),(p,n)} 2D numpy arrays
        Converted system matrices 
    T : (n,n) 2D numpy arrays
        If the boolean "compute_T" is true, returns the transformation 
        matrix such that 
        
                        [T^T * A * T | T^T * B]
                        [    C * T   |    D   ]

        is in the desired staircase form.
    k: np.array
        Array of controllable block sizes identified during block 
        diagonalization
        
    """


    if not form in {'c','o'}:
        raise ValueError('The "form" key can only take values'
                         '\"c\" or \"o\" denoting\ncontroller- or '
                         'observer-Hessenberg form.')
    if form == 'o':
        A , B , C = A.T , C.T , B.T
    
    n = A.shape[0]
    ub , sb , vb , m0 = haroldsvd(B,also_rank=True)
    cble_block_indices = np.empty((1,0))

    # Trivially  Uncontrollable Case
    # Skip the first branch of the loop by making m0 greater than n
    # such that the matrices are returned as is without any computation
    if m0 == 0:
        m0 = n + 1
        cble_block_indices = np.array([0])

    # After these, start the regular case
    if n > m0:# If it is not a square system with full rank B

        A0 = ub.T.dot(A.dot(ub))
#        print(A0)

        # Row compress B and consistent zero blocks with the reported rank 
        B0 = sb.dot(vb)
        B0[m0:,:] = 0. 
        C0 = C.dot(ub)
        cble_block_indices = np.append(cble_block_indices,m0)

        if compute_T:
            P = blockdiag(np.eye(n-ub.T.shape[0]),ub.T)

        # Since we deal with submatrices, we need to increase the
        # default tolerance to reasonably high values that are 
        # related to the original data to get exact zeros
        tol_from_A = n*sp.linalg.norm(A,1)*np.finfo(float).eps

        # Region of interest
        m = m0
        ROI_start = 0
        ROI_size = 0

        for dummy_row_counter in range(A.shape[0]):
            ROI_start += ROI_size
            ROI_size = m
#            print(ROI_start,ROI_size)
            h1,h2,h3,h4 = matrixslice(
                                A0[ROI_start:,ROI_start:],
                                (ROI_size,ROI_size)
                                )
            uh3,sh3,vh3,m = haroldsvd(h3,also_rank=True,rank_tol = tol_from_A)

            # Make sure reported rank and sh3 are consistent about zeros
            sh3[ sh3 < tol_from_A ] = 0.

            # If the resulting subblock is not full row or zero rank
            if 0 < m < h3.shape[0]:
                cble_block_indices = np.append(cble_block_indices,m)
                if compute_T:
                    P = blockdiag(np.eye(n-uh3.shape[1]),uh3.T).dot(P)
                A0[ROI_start:,ROI_start:] = np.r_[
                                    np.c_[h1,h2],
                                    np.c_[sh3.dot(vh3),uh3.T.dot(h4)]
                                    ]
                A0 = A0.dot(blockdiag(np.eye(n-uh3.shape[1]),uh3))
                # Clean up
                A0[abs(A0) < tol_from_A ] = 0.
                C0[abs(C0) < tol_from_A ] = 0.                
            elif m == h3.shape[0]:
                cble_block_indices = np.append(cble_block_indices,m)
                break
            else:
                break
        
        if invert: 
            A0 = np.fliplr(np.flipud(A0))
            B0 = np.flipud(B0)
            C0 = np.fliplr(C0)
            if compute_T:
                P = np.flipud(P)

        if form == 'o':
            A0 , B0 , C0 = A0.T , C0.T , B0.T

        if compute_T:
            return A0,B0,C0,P.T,cble_block_indices
        else:
            return A0,B0,C0,cble_block_indices

    else: # Square system B full rank ==> trivially controllable
        cble_block_indices = np.array([n])
        if form == 'o':
            A , B , C = A.T , C.T , B.T
        
        if compute_T:
            return A,B,C,np.eye(n),cble_block_indices
        else:
            return A,B,C,cble_block_indices

def canceldistance(F,G):
    """
    Given matrices F,G, computes the upper and lower bounds of 
    the perturbation needed to render the pencil [F-pI | G]
    rank deficient. It is used for assessing the controllability/
    observability degenerate distance and hence for minimality 
    assessment. 
    
    Implements the algorithm given in D.Boley SIMAX vol.11(4) 1990. 
    
    Parameters
    ----------
    F,G : {(n,n), (n,m)} array_like
        Pencil matrices to be checked for rank deficiency distance

    Returns
    -------

    upper2 : float
        Upper bound on the norm of the perturbation [dF | dG] such
        that [F + dF - pI | G + dG ] is rank deficient. 
    upper1 : float
        A theoretically softer upper bound than the upper2 for the 
        same quantity.
    lower0 : float
        Lower bound on the same quantity given in upper2
    e_f    : complex
        Indicates the eigenvalue that renders [F + dF - pI | G + dG ] 
        rank deficient i.e. equals to the p value at the closest rank
        deficiency.
    radius : float
        The perturbation with the norm bound "upper2" is located within 
        a disk in the complex plane whose center is on "e_f" and whose 
        radius is bounded by this output.
      
    """
    A = np.c_[F,G].T
    n , m = A.shape
    B = eyecolumn(n,np.s_[:m])
    D = eyecolumn(n,np.s_[m:])
    C = sp.linalg.qr(2*np.random.rand(n,n-m) - 1,mode='economic')[0]
    evals , V = sp.linalg.eig(np.c_[A,C])
    K = np.linalg.cond(V)
    X = V[:m,:]
    Y = V[m:,:]

    upp0 = [0]*n
    for x in range(n):
        upp0[x] = sp.linalg.norm(  (C-evals[x]*D).dot(Y[:,x])
                                            ) / sp.linalg.norm(X[:,x])

    f = np.argsort(upp0)[0]
    e_f = evals[f]
    upper1 = upp0[f]
    upper2 = sp.linalg.svdvals(A - e_f*B)[-1]
    lower0 = upper2/(K+1)
    radius = upper2*K
    
    return upper2 , upper1 , lower0 , e_f , radius



def minimal_realization(A,B,C,mu_tol=1e-9):
    """
    Given state matrices A,B,C computes minimal state matrices 
    such that the system is controllable and observable within the
    given tolerance mu. 
    
    Implements a basic two pass algorithm : 
     1- First distance to mode cancellation is computed then also 
     the Hessenberg form is obtained with the identified o'ble/c'ble 
     block numbers. If staircase form reports that there are no 
     cancellations but the distance is less than the tolerance, 
     distance wins and the respective mode is removed. 
    
    Uses canceldistance(), and staircase() for the aforementioned checks. 
    
    Parameters
    ----------
    A,B,C : {(n,n), (n,m), (pxn)} array_like
        System matrices to be checked for minimality
    mu_tol: float (default 1-e6)
        The sensitivity threshold for the cancellation to be compared 
        with the first default output of canceldistance() function.

    Returns
    -------

    A,B,C : {(k,k), (k,m), (pxk)} array_like
        System matrices that are identified as minimal with k states
        instead of the original n where (k <= n)
    
    """ 
    
    keep_looking = True
    run_out_of_states = False
    
    while keep_looking:
        n = A.shape[0]
        # Make sure that we still have states left
        if n == 0:
            A , B , C = [(np.empty((1,0)))]*3
            break
        
        kc = canceldistance(A,B)[0]
        ko = canceldistance(A.T,C.T)[0]

        if min(kc,ko) > mu_tol: # no cancellation
            keep_looking= False
        else:
            
            Ac,Bc,Cc,blocks_c = staircase(A,B,C)
            Ao,Bo,Co,blocks_o = staircase(A,B,C,form='o',invert=True)

            # ===============Extra Check============================
            """            
             Here kc,ko reports a possible cancellation so staircase 
             should also report fewer than n, c'ble/o'ble blocks in the 
             decomposition. If not, staircase tol should be increased. 
             Otherwise either infinite loop or uno'ble branch removes
             the system matrices
            
             Thus, we remove the last scalar or the two-by-two block
             artificially. Because we trust the cancelling distance, 
             more than our first born. The possible cases of unc'ble 
             modes are
               
               -- one real distinct eigenvalue
               -- two real identical eigenvalues 
               -- two complex conjugate eigenvalues
                        
             We don't regret this. This is sparta.
            """
            
            # If unobservability distance is closer, let it handle first
            if ko>=kc:
                if (sum(blocks_c) == n and kc <= mu_tol):
                    Ac_mod , Bc_mod , Cc_mod , kc_mod = Ac,Bc,Cc,kc
    
                    while kc_mod <= mu_tol:# Until cancel dist gets big
                        Ac_mod,Bc_mod,Cc_mod = (
                                Ac_mod[:-1,:-1],Bc_mod[:-1,:],Cc_mod[:,:-1])
                                
                        if Ac_mod.size == 0:
                            A , B , C = [(np.empty((1,0)))]*3
                            run_out_of_states = True
                            break
                        else:
                            kc_mod = canceldistance(Ac_mod,Bc_mod)[0]
    
                    kc = kc_mod
                    # Fake an iterable to fool the sum below
                    blocks_c = [sum(blocks_c)-Ac_mod.shape[0]]


            # Same with the o'ble modes
            if (sum(blocks_o) == n and ko <= mu_tol):
                Ao_mod , Bo_mod , Co_mod , ko_mod = Ao,Bo,Co,ko

                while ko_mod <= mu_tol:# Until cancel dist gets big
                    Ao_mod,Bo_mod,Co_mod = (
                            Ao_mod[1:,1:],Bo_mod[1:,:],Co_mod[:,1:])
                    
                    # If there is nothing left, break out everything
                    if Ao_mod.size == 0:
                        A , B , C = [(np.empty((1,0)))]*3
                        run_out_of_states = True
                        break
                    else:
                        ko_mod = canceldistance(Ao_mod,Bo_mod)[0]

                
                ko = ko_mod
                blocks_o = [sum(blocks_o)-Ao_mod.shape[0]]

            # ===============End of Extra Check=====================
             
            if run_out_of_states: break
             
            if sum(blocks_c) > sum(blocks_o):
                remove_from = 'o'
            elif sum(blocks_c) < sum(blocks_o):
                remove_from = 'c'
            else: # both have the same number of states to be removed
                if kc >= ko:
                    remove_from = 'o'
                else:
                    remove_from = 'c'


            if remove_from == 'c':
                l = sum(blocks_c)
                A , B , C = Ac[:l,:l] , Bc[:l,:] , Cc[:,:l]
            else:
                l = n - sum(blocks_o)
                A , B , C = Ao[l:,l:] , Bo[l:,:] , Co[:,l:]
 
    return A , B, C


def haroldsvd(D,also_rank=False,rank_tol=None):
    """
    This is a wrapper/container function of both the SVD decomposition
    and the rank computation. Since the regular rank computation is 
    implemented via SVD it doesn't make too much sense to recompute 
    the SVD if we already have the rank information. Thus instead of 
    typing two commands back to back for both the SVD and rank, we 
    return both. To reduce the clutter, the rank information is supressed
    by default. 
    
    numpy svd is a bit strange because it compresses and looses the 
    S matrix structure. From the manual, it is advised to use 
    u.dot(np.diag(s).dot(v)) for recovering the original matrix. But 
    that won't work for rectangular matrices. Hence it recreates the 
    rectangular S matrix of U,S,V triplet.

    Parameters
    ----------
    D : (m,n) array_like
        Matrix to be decomposed
    also_rank : bool, optional
        Whether the rank of the matrix should also be reported or not.
        The returned rank is computed via the definition taken from the
        official numpy.linalg.matrix_rank and appended here.
    rank_tol : {None,float} optional
        The tolerance used for deciding the numerical rank. The default 
        is set to None and uses the default definition of matrix_rank()
        from numpy.

    Returns
    -------

    U,S,V : {(m,m),(m,n),(n,n)} 2D numpy arrays
        Decomposed-form matrices
    r : integer
        If the boolean "also_rank" is true, this variable is the numerical
        rank of the matrix D

    """
    try:
        D = np.atleast_2d(np.array(D,dtype='float'))
    except TypeError:
        raise TypeError('Incompatible argument, use either list of lists'
                        'or native numpy arrays for svd.')
    except ValueError:
        raise ValueError('The argument cannot be cast as an array with'
                        '"float" entries')
            
    p,m = D.shape
    u,s,v = np.linalg.svd(D,full_matrices=True)
    diags = np.zeros((p,m))# Reallocate the s matrix of u,s,v
    for index, svalue in enumerate(s):# Repopulate the diagoanal with svds
        diags[index,index] = svalue
   
    if also_rank:# Copy the official rank computation
        if rank_tol is None:
            rank_tol = s.max() * max(p,m) * np.finfo(s.dtype).eps
        r = sum(s > rank_tol)
        return u,diags,v,r

    return u,diags,v

#TODO : type checking for both.

def ssconcat(G):
    if not isinstance(G,State):
        raise TypeError('ssconcat() works on state representations, '
        'but I found \"{0}\" object instead.'.format(type(G).__name__))
    H = np.vstack((np.hstack((G.a,G.b)),np.hstack((G.c,G.d))))
    return H

# TODO : Add slicing with respect to D matrix
def ssslice(H,n):
#    return H[:n,:n],H[:n,n:],H[n:,:n],H[n:,n:]
    return matrixslice(H,(n,n))

def matrixslice(M,M11shape):
    p , m = M11shape
    return M[:p,:m],M[:p,m:],M[p:,:m],M[p:,m:]

    
# I don't understand how this is not implemented already
#TODO : type checking.
def blockdiag(*args):
    # Get the size info of the args
    try:
        diags = tuple([m.shape for m in args if m.size > 0])
    except AttributeError:
        args = [np.atleast_2d(x) for x in args]
        diags = tuple([m.shape for m in args if m.size > 0])
    
    poppedargs = tuple([x for x in args if x.size >0])
    tot = np.zeros(tuple(map(sum,zip(*diags))))# Sum shapes for the final size
    rind,cind=(0,0)

    # Place each of them and move the index to the next diag pos
    for ind,mat in enumerate(poppedargs):
        tot[rind:rind+diags[ind][0], cind:cind+diags[ind][1]] = mat
        rind += diags[ind][0]
        cind += diags[ind][1]
    return tot


# Returns the nth column of an identity matrix as a 2D numpy array.
def eyecolumn(width,nth=0):
    return np.eye(width)[[nth]].T

# norms, where do these belong? magnets, how do they work? 
def system_norm(state_or_transfer, 
                p = np.inf, 
                validate=False, 
                verbose=False,
                why_inf=False):
    """
    Computes the system p-norm. Currently, no balancing is done on the 
    system, however in the future, a scaling of some sort will be introduced.
    Another short-coming is that while sounding general, only H2 and Hinf 
    norm are understood. 
    
    For Hinf norm, (with kind and generous help of Melina Freitag) the 
    algorithm given in:
    
    M.A. Freitag, A Spence, P. Van Dooren: Calculating the $H_\infty$-norm 
    using the implicit determinant method. SIAM J. Matrix Anal. Appl., 35(2), 
    619-635, 2014

    For H2 norm, the standard grammian definition via controllability 
    grammian can be found elsewhere is used.
    
    Parameters
    ----------
    state_or_transfer : {State,Transfer}
        System for which the norm is computed
    p : {int,Inf}
        Whether the rank of the matrix should also be reported or not.
        The returned rank is computed via the definition taken from the
        official numpy.linalg.matrix_rank and appended here.

    validate: boolean
        If applicable and if the resulting norm is finite, the result is 
        validated via other means.

    verbose: boolean
        If True, the (some) internal progress is printed out.
    
    why_inf: boolean
        Returns the reason why the result is set to infinity. Might give
        some hints when stuck. 

    Returns
    -------

    n : float
        Computed norm. In NumPy, infinity is also float-type
    omega : float
        For Hinf norm, omega is the frequency where the maximum is attained
        (technically this is a numerical approximation of the supremum).
    reason : str
        If why_inf is true, returns the string about the reason if the 
        result is infinity. Complains about the ungrateful user if the 
        result is finite.
        
    """
    if not isinstance(state_or_transfer,(State,Transfer)):
        raise('The argument should be a State or Transfer. Instead I '
              'received {0}'.format(type(state_or_transfer).__qualname__))
    if isinstance(state_or_transfer,Transfer):
        now_state = transfertostate(state_or_transfer)
    else:
        now_state = state_or_transfer
    
    if not isinstance(p,(int,float)):
        raise('The p in p-norm is not an integer or float.'
              'If you tried the string \'inf\', use Numpy.Inf instead')

    # Two norm
    if p == 2:
        # Handle trivial infinities
        if now_state._isgain:
            # If nonzero -> infinity, if zero -> zero
            if np.count_nonzero(now_state.d) > 0:
                return np.Inf
                if why_inf:
                    reason = 'The system has a non-zero feedthrough term.'
            else:
                return 0.
            
        if not now_state._isstable:
            if why_inf:
                reason = 'The system is not stable.'
            return np.Inf
            
        a , b = now_state.matrices[:2]
        x = sp.linalg.solve_sylvester(a,a.T,-b.dot(b.T))
        n = np.sqrt(np.trace(c.dot(x.dot(c.T))))

        if why_inf:
            return n,reason
        return n

    elif np.isinf(p):
        
    else:
        raise('I can only handle the cases for p=2,inf for now.')
            
            
# %% Polynomial ops    
def haroldlcm(*args,compute_multipliers=True,cleanup_threshold=1e-9):
    """
    Takes n-many 1D numpy arrays and computes the numerical 
    least common multiple polynomial. The polynomials are
    assumed to be in decreasing powers, e.g. s^2 + 5 should
    be given as numpy.array([1,0,5])
    
    Returns a numpy array holding the polynomial coefficients
    of LCM and a list, of which entries are the polynomial 
    multipliers to arrive at the LCM of each input element. 
    
    Example: 
    
    >>>> a , b = haroldlcm(*map(
                            np.array,
                            ([1,3,0,-4],[1,-4,-3,18],[1,-4,3],[1,-2,-8])
                            )
                        )
    >>>> a 
        (array([   1.,   -7.,    3.,   59.,  -68., -132.,  144.])

    >>>> b
        [array([  1., -10.,  33., -36.]),
         array([  1.,  -3.,  -6.,   8.]),
         array([  1.,  -3., -12.,  20.,  48.]),
         array([  1.,  -5.,   1.,  21., -18.])]
         
    >>>> np.convolve([1,3,0,-4],b[0]) # or haroldpolymul() for poly mult
        (array([   1.,   -7.,    3.,   59.,  -68., -132.,  144.]),

    """
    # As typical, it turns out that the minimality and c'ble subspace for 
    # this is done already (Karcanias, Mitrouli, 2004). They also have a 
    # clever extra step for the multipliers thanks to the structure of 
    # adjoint which I completely overlooked. 
    if not all([isinstance(x,type(np.array([0]))) for x in args]):
        raise TypeError('Some arguments are not numpy arrays for LCM')

    # Remove if there are constant polynomials but return their multiplier!
    poppedargs = tuple([x for x in args if x.size>1])
    # Get the index number of the ones that are popped
    poppedindex = tuple([ind for ind,x in enumerate(args) if x.size==1])
    a = blockdiag(*tuple(map(haroldcompanion,poppedargs))) # Companion A
    b = np.concatenate(tuple(map(lambda x: eyecolumn(x-1,-1),
                                 [z.size for z in poppedargs])))# Companion B
    c = blockdiag(*tuple(map(lambda x: eyecolumn(x-1,0).T,
                                 [z.size for z in poppedargs])))
    n = a.shape[0]

    # TODO: Below two lines feel like matlab programming, revisit again
    C = b
    i = 1
    # Computing full c'bility matrix is redundant we just need to see where 
    # the rank drop is (if any!). 
    # Also due matrix power, things grow too quickly.
    while np.linalg.matrix_rank(C) == C.shape[1] and i<=n:
        C = np.hstack((C,np.linalg.matrix_power(a,i).dot(b)))
        i += 1
    s,v = haroldsvd(C)[1:]
    temp = s.dot(v)
    # If not coprime we should "expect" zero rows at the bottom

    if i-1==n:# Relatively coprime
        temp2 =  np.linalg.inv(temp[:,:-1])# Every col until the last
    else:
        temp2 = blockdiag(np.linalg.inv(temp[:i-1,:i-1]),np.eye(n+1-i))
    

    lcmpoly= temp2.dot(-temp)[:i-1,-1]
    # Add monic coefficient and flip
    lcmpoly= np.append(lcmpoly,1)[::-1]
    
    if compute_multipliers:
        a_lcm = haroldcompanion(lcmpoly)
        b_lcm = np.linalg.pinv(C[:c.shape[1],:-1]).dot(b)
        c_lcm = c.dot(C[:c.shape[1],:-1])
        
        # adj(sI-A) formulas with A being a companion matrix
        # We need an array container so back to list of lists
        n_lcm = a_lcm.shape[0]
        # Create a list of lists of lists with zeros
        adjA = [[[0]*n_lcm for m in range(n_lcm)] for n in range(n_lcm)]
    
        # looping fun
        for x in range(n_lcm):
            # Diagonal terms
            adjA[x][x][:n_lcm-x] = list(lcmpoly[:n_lcm-x])
            for y in range(n_lcm):
                if y<x:  # Upper Triangular terms
                    adjA[y][x][x-y:] = adjA[x][x][:n_lcm-(x-y)]
                elif y>x:# Lower Triangular terms
                    adjA[y][x][n_lcm-y:n_lcm+1-y+x] = list(-lcmpoly[-x-1:n_lcm+1])
    
        """
        Ok, now get C_lcm * adj(sI-A_lcm) * B_lcm
    
        Since we are dealing with lists we have to fake a matrix multiplication
        with an evil hack. The reason is that, entries of adj(sI-A_lcm) are 
        polynomials and numpy doesn't have a container for such stuff hence we
        store them in Python "list" objects and manually perform elementwise
        multiplication.
            
        Middle three lines take the respective element of b vector and multiplies
        the column of list of lists. Hence we actually obtain
        
                    adj(sI-A_lcm) * blockdiag(B_lcm)
        
        The resulting row entries are added to each other to get adj(sI-A)*B_lcm
        Finally, since we now have a single column we can treat polynomial
        entries as matrix entries hence multiplied with c matrix properly. 
        
        """
        mults = c_lcm.dot(
            np.vstack(
                tuple(
                    [haroldpolyadd(*w,trimzeros=False) for w in 
                        tuple(
                            [
                              [ 
                                [b_lcm[y,0]*z for z in adjA[y][x]] 
                                  for y in range(n_lcm)
                              ] for x in range(n_lcm)
                            ]
                          )
                        ]
                      )
                    )
                  )
                      
        # If any reinsert lcm polynomial for constant polynomials  
        if not poppedindex==():
            dummyindex = 0
            dummymatrix = np.zeros((len(args),lcmpoly.size))
            for x in range(len(args)):
                if x in poppedindex:
                    dummymatrix[x,:] = lcmpoly
                    dummyindex +=1
                else:
                    dummymatrix[x,1:] = mults[x-dummyindex,:]
            mults = dummymatrix
                    
        lcmpoly[ abs(lcmpoly) < cleanup_threshold ] = 0.
        mults[ abs(mults) < cleanup_threshold ] = 0.
        mults = [ haroldtrimleftzeros(z) for z in mults ]
        return lcmpoly, mults
    else:
        return lcmpoly

def haroldgcd(*args):
    """
    Takes *args-many 1D numpy arrays and computes the numerical 
    greatest common divisor polynomial. The polynomials are
    assumed to be in decreasing powers, e.g. s^2 + 5 should
    be given as numpy.array([1,0,5])
    
    Returns a numpy array holding the polynomial coefficients
    of GCD. The GCD does not cancel scalars but returns only monic roots.
    In other words, the GCD of polynomials 2 and 2s+4 is computed
    as 1. 
    
    Example: 
    
    >>>> a = haroldgcd(*map(
                haroldpoly,
                ([-1,-1,-2,-1j,1j],[-2,-3,-4,-5],[-2]*10)
              )
            )
    >>>> print(a)
         array([ 1.,  2.])
         
    
    DISCLAIMER : It uses the LU factorization of the Sylvester matrix.
                 Use responsibly. It does not check any certificate of 
                 success by any means (maybe it will in the future).
                 I've tried the recent ERES method too. When there is a 
                 nontrivial GCD it performed satisfactorily however did 
                 not perform as well when GCD = 1 (maybe due to my 
                 implementation). Hence I've switched to matrix-based 
                 methods.

    """    


    if not all([isinstance(x,type(np.array([0]))) for x in args]):
        raise TypeError('Some arguments are not numpy arrays for GCD')

    not_1d_err_msg = ('GCD computations require explicit 1D '
                     'numpy arrays or\n2D but having one of '
                     'the dimensions being 1 e.g., (n,1) or (1,n)\narrays.')
    try:
        regular_args = [haroldtrimleftzeros(
                            np.atleast_1d(np.squeeze(x)).astype(float)
                            ) for x in args]
    except:
        raise ValueError(not_1d_err_msg)
    
    try:
        dimension_list = [x.ndim for x in regular_args]
    except AttributeError:
        raise ValueError(not_1d_err_msg)

    # do we have 2d elements? 
    if max(dimension_list) > 1:
        raise ValueError(not_1d_err_msg)
        
    degree_list = np.array([x.size - 1 for x in regular_args])
    max_degree = np.max(degree_list)
    max_degree_index = np.argmax(degree_list)
    
    try:
        # There are polynomials of lesser degree
        second_max_degree = np.max(degree_list[degree_list<max_degree])
    except ValueError:
        # all degrees are the same 
        second_max_degree = max_degree


    n , p , h = max_degree , second_max_degree , len(regular_args) - 1

    # If a single item is passed then return it back
    if h == 0:
        return regular_args[0]
    
    if n == 0:
        return np.array([1])

    if n > 0 and p==0:
        return regular_args.pop(max_degree_index)

        
    # pop out the max degree polynomial and zero pad 
    # such that we have n+m columns
    S = np.array([np.hstack((
            regular_args.pop(max_degree_index),
            np.zeros((1,p-1)).squeeze()
            ))]*p)
    
    # Shift rows to the left
    for rows in range(S.shape[0]):
        S[rows] = np.roll(S[rows],rows)

    # do the same to the remaining ones inside the regular_args
    
    for item in regular_args:
        _ = np.array([np.hstack((item,[0]*(n+p-item.size)))]*(n+p-item.size+1))
        for rows in range(_.shape[0]):
            _[rows] = np.roll(_[rows],rows)
        S = np.r_[S,_]

    rank_of_sylmat = np.linalg.matrix_rank(S)
    
    if rank_of_sylmat == min(S.shape):
        return np.array([1])
    else:
        p , l , u = sp.linalg.lu(S)
        
    u[abs(u)<1e-8] = 0
    for rows in range(u.shape[0]-1,0,-1):
        if not any(u[rows,:]):
            u = np.delete(u,rows,0)
        else:
            break

    gcdpoly = np.real(haroldtrimleftzeros(u[-1,:]))
    # make it monic
    gcdpoly /= gcdpoly[0]
    
    return gcdpoly



def haroldcompanion(somearray):
    """
    Takes an 1D numpy array or list and returns the companion matrix
    of the monic polynomial of somearray. Hence [0.5,1,2] will be first
    converted to [1,2,4]
    
    Example:
    
    >>>> haroldcompanion([2,4,6])
        array([[ 0.,  1.],
               [-3., -2.]])

    >>>> haroldcompanion([1,3])
        array([[-3.]])

    >>>> haroldcompanion([1])
        array([], dtype=float64)
    
    """
    if not isinstance(somearray,(list,type(np.array([0.])))):
        raise TypeError('Companion matrices are meant only for '
                        '1D lists or 1D Numpy arrays. I found '
                        'a \"{0}\"'.format(type(somearray).__name__))

    if len(somearray)==0:
        return np.array([])

    # regularize to flat 1D np.array
    
    somearray = np.array(somearray,dtype='float').flatten()

    ta = haroldtrimleftzeros(somearray)
    # convert to monic polynomial. 
    # Note: ta *=... syntax doesn't autoconvert to float
    ta = np.array(1/ta[0])*ta
    ta = -ta[-1:0:-1]
    n = ta.size
    
    if n == 0:# Constant polynomial
        return np.array([])

    elif n == 1:# First-order --> companion matrix is a scalar
        return np.atleast_2d(np.array(ta))

    else:# Other stuff
        return np.vstack((np.hstack((np.zeros((n-1,1)),np.eye(n-1))),ta))
                          
def haroldtrimleftzeros(somearray):
    # We trim the leftmost zero entries modeling the absent high-order terms
    # in an array, i.e., [0,0,2,3,1,0] becomes [2,3,1,0]

    arg = np.atleast_2d(somearray).astype(float).flatten()

    if arg.ndim>1:
        raise ValueError('The argument is not 1D array-like hence cannot be'
                         ' trimmed unambiguously.')
    
    if np.count_nonzero(arg) != 0:# if not all zero
        try:
            n = next(x for x,y in enumerate(arg) if y != 0.)
            return np.array(arg)[n::]
        except StopIteration:
            return np.array(arg[::])
    else:
        return np.array([0.])
        
        
def haroldpoly(rootlist):
    if isinstance(rootlist, collections.Iterable):
        r = np.array([x for x in rootlist],dtype=complex)
    else:
        raise TypeError('The argument must be something iterable,\nsuch as '
        'list, numpy array, tuple etc. I don\'t know\nwhat to do with a '
        '\"{0}\" object.'.format(type(rootlist).__name__))
    
    n = r.size
    if n == 0:
        return np.ones(1)
    else:
        p = np.array([0.+0j for x in range(n+1)],dtype=complex)
        p[0] = 1 # Monic polynomial
        p[1] = -rootlist[0] # 
        for x in range(1,n):
            p[x+1] = -p[x]*r[x]
            for y in range(x,0,-1):
                p[y] -= p[y-1] * r[x]
        return p

def haroldpolyadd(*args,trimzeros=True):
    """
    Similar to official polyadd from numpy but allows for 
    multiple args and doesn't invert the order, 
    """
    if trimzeros:
        trimmedargs = tuple(map(haroldtrimleftzeros,args))
    else:
        trimmedargs = args
        
    degs = [len(m) for m in trimmedargs]# Get the max len of args
    s = np.zeros((1,max(degs)))
    for ind,x in enumerate(trimmedargs):
        s[0,max(degs)-degs[ind]:] += np.real(x)
    return s[0]

def haroldpolymul(*args,trimzeros=True):
    """
    Simple wrapper around the scipy convolve function
    for polynomial multiplication with multiple args.
    The arguments are passed through the left zero 
    trimming function first.
    
    Example: 
    
    >>>> haroldpolymul([0,2,0],[0,0,0,1,3,3,1],[0,0.5,0.5])
    array([ 1.,  4.,  6.,  4.,  1.,  0.])
    
    
    """
    # TODO: Make sure we have 1D arrays for convolution
    # numpy convolve is too picky.

    if trimzeros:
        trimmedargs = tuple(map(haroldtrimleftzeros,args))
    else:
        trimmedargs = args
        
    p = trimmedargs[0]

    for x in trimmedargs[1:]:
        try:
            p = np.convolve(p,x)
        except ValueError:
            p = np.convolve(p.flatten(),x.flatten())


    return p
    
def haroldpolydiv(dividend,divisor):
    """
    Polynomial division wrapped around scipy deconvolve 
    function. Takes two arguments and divides the first
    by the second. 
    
    Returns, two arguments: the factor and the remainder, 
    both passed through a left zeros trimming function.
    """
    h_factor , h_remainder = map(haroldtrimleftzeros,
                                 deconvolve(dividend,divisor)
                                 )
    
    return h_factor , h_remainder

# %% Frequency Domain

def frequency_response(G,custom_grid=None,high=None,low=None,samples=None,
                       custom_logspace=None,
                       input_freq_unit='Hz',output_freq_unit='Hz'):
    """
    Computes the frequency response matrix of a State() or Transfer()
    object. Transfer matrices are converted to state representations
    before the computations. The system representations are always 
    checked for minimality and, if any, unobservable/uncontrollable 
    modes are removed.

    """
    if not isinstance(G,(State,Transfer)):
        raise ValueError('The argument should either be a State() or '
                         'Transfer() object. I have found {0}'
                         ''.format(type(G).__qualname__))
    

    for x in (input_freq_unit,output_freq_unit):
        if x not in ('Hz','rad/s'):
            raise ValueError('I can only handle "Hz" and "rad/s" as '
                             'frequency units. "{0}" is not recognized.'
                             ''.format(x))

    """ If the system has very small or zero poles/zeros then 
     the list should be shifted to meaningful region because the result
     would either be huge or unpractically small especially in logaritmic
     scale, completely useless."""     

    if G._isgain:
        samples = 2
        high = -2
        low = 2
    else:
        pz_list = np.append(G.poles,G.zeros)

        if G.SamplingSet == 'Z':
            nat_freq = np.abs(np.log(pz_list / G.SamplingPeriod))
        else:
            nat_freq = np.abs(pz_list)

        smallest_pz = np.max([np.min(nat_freq),1e-7])
        largest_pz  = np.max([np.max(nat_freq),smallest_pz+10])

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
    # or a zero head on. 
    # matlab coarseness stuff is nice but in practice leads to weirdness
    # even when granularity = 4.

    if custom_grid is None:
        if custom_logspace is None:
            high = np.ceil(np.log10(largest_pz)) + 1 if high is None else high
            low  = np.floor(np.log10(smallest_pz)) - 1 if low  is None else low
            samples = 1000 if samples is None else samples
        else:
            high , low , samples = custom_logspace
        w = np.logspace(low,high,samples)
    else:
        w = np.asarray(custom_grid,dtype='float')

    # Convert to Hz if necessary
    if not input_freq_unit == 'Hz':
        w = np.rad2deg(w)

    iw = 1j*w.flatten()

    # TODO: This has to be rewritten, currently extremely naive
    if G._isSISO:
        freq_resp_array = np.empty_like(iw,dtype='complex')
        
        if isinstance(G,State):
            Gtf = statetotransfer(G)
        freq_resp_array = (np.polyval(Gtf.num[0],iw) /
                           np.polyval(Gtf.den[0],iw)
                           )
    else:
        p , m = G.shape
        freq_resp_array = np.empty((p,m,len(iw)),dtype='complex')
        if isinstance(G,State):
            Gtf = statetotransfer(G)
        else:
            Gtf = G
        for rows in range(p):
            for cols in range(m):
                freq_resp_array[rows,cols,:] = (
                        np.polyval(Gtf.num[rows][cols][0],iw) /
                        np.polyval(Gtf.den[rows][cols][0],iw)
                        )

    return freq_resp_array , w