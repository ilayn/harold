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
from copy import deepcopy
from scipy.signal import deconvolve
from itertools import zip_longest
import collections



# %% Urgent TODOS
"""

- Switch to r_[] and c_[] notation for proper readability instead of 
    hstack, vstack if possible.
   
- Implement the resampling to the slowest Sampling Period of systems for 
    LTI arithmetic..
    - TF MIMO arithmetics requires resampling if dt of systems are 
      different. Allow for resampling and convert to the slowest. 
    
- Documentation (Need a lot of time for this)

- __repr__ of ss and tf are too rough, clean up

- Fix the bokeh business, figure out why it doesn't play well with IPython

- Stop using dB and switch to log10 ala Åström and Murray. They have a 
    point.

- Write the c2d, d2c and d2d properly. (almost done)

       Write a section in the manual why this alpha works with 3d plots of 
       Riemann sphere with a link to the original video. 
       Finish the pgfplots part --> Try to fix the /.estyle bug

- Record the discretization method in the instance such that if exists 
  the d2c understands the previous discretization method and inverts 
  according to that information. (Looks like DONE)

- tf2ss and ss2tf First rename these! Then implement duck typing. Current 
  syntax is too clunky

- mat file saving (if matlab plays nice,loading too. But not much hope)

- Convert ctrb to obsv to staircase and canceldist form. And rename!


"""

# %% Module Definitions

KnownDiscretizationMethods = ('bilinear',
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
        self._DiscretizedWith = None       
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._SamplingPeriod = False



        # Submit the numerator entry to the validator. 
        num,num_SMflag = self.validatepolymatrix(num)
            
        # Now we have both the normalized version of the numerator
        # and the detected {SISO,MIMO} flag.

        if num_SMflag == 'S':
            self._num = np.atleast_2d(num)
            self._m = 1
            self._p = 1
        elif num_SMflag == 'M':
            self._num = num
            self._p = len(self._num[0]) # should be validated already.
            self._m = len(self._num)


        if den is None:
            self._isgain = True


        if not self._isgain:
            # Submit the denominator
            den,den_SMflag = self.validatepolymatrix(den,name='Denominator')
            
            # We have two options in case of a SISO flag, either indeed 
            # SISO or it is a MIMO TF with a common denominator
    
            if den_SMflag == 'S':
                # if a single entry is given for MIMO (set by num size)
                if max(self._m,self._p) > 1:
                    self._den = [[np.array(den)]*self._m for n in range(self._p)]
                    init_vars_den_shape = (self._p,self._m)
                else:
                    self._den = np.atleast_2d(den)
                    init_vars_den_shape = (1,1)
            else:
                self._den = den
                init_vars_den_shape = (len(self._den[0]),len(self._den))
        else:
            den = np.ones((self._p,self._m)).tolist()
            self._den = []
            for den_row in den:
                    self._den += [[np.array(x) for x in den_row]]

            init_vars_den_shape = (self._p,self._m)
            


        # Now check if the num and den sizes match otherwise reject
        if not (self._p,self._m) == init_vars_den_shape:
            raise IndexError('I have a {0}x{1} shaped numerator and a '
                            '{2}x{3} shaped \ndenominator. Hence I can '
                            'not initialize this transfer \nfunction. '
                            'I secretly blame you for this.'.format(
                                        self._p,
                                        self._m,
                                        *init_vars_den_shape
                                )
                            )


        
        self._shape = (self._p,self._m)
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
        if validatepolymatrix(value) in (1,2):
            self._num = np.array(value)
            self._recalc()
        elif (len(value[0]),len(value)) == (self._m,self._p):
            self._num = value
            self._recalc()
        else:
            raise IndexError('Once created, the shape of the transfer '
                            'function \ncannot be changed. I have '
                            'received a numerator with shape {0}x{1} \nbut '
                            'the system has {2}x{3}.'.format(
                                len(value[0]),len(value),self._m,self._p
                                )
                            )
            
    @den.setter
    def den(self, value):
        if (validatepolymatrix(value,name='Denominator') in (1,2)
                                          and (self._m,self._p) == (1,1)):

            self._den = np.array(value)
            self._recalc()
        elif (len(value[0]),len(value)) == (self._m,self._p):
            self._den = value
            self._recalc()
        else:
            raise IndexError(
                    'Once created, the shape of the transfer '
                    'function \ncannot be changed. I have '
                    'received a denominator with shape {0}x{1} \nbut '
                    'the system has {2}x{3}.'.format(
                        len(value[0]),len(value),self._m,self._p
                        )
                    )


    @DiscretizedWith.setter
    def DiscretizedWith(self,value):
        if value in KnownDiscretizationMethods:
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
               

    # FIXME: Zero dynamics will fail for SISO here and a few other places
    def _recalc(self):
        print(self._den,self._num)
        if self._isgain:
            self.poles = []
            self.zeros = []
        else:
            if self._isSISO:
                self.poles = np.linalg.eigvals(haroldcompanion(self._den))
                zeros_matrix = haroldcompanion(self._num)
                if not zeros_matrix.size == 0:
                    self.zeros = np.linalg.eigvals(zeros_matrix)
                else:
                    self.zeros = np.array([])
            else:
                # Create a dummy statespace and check the zeros there
                zzz = tf2ss(self._num,self._den)
                self.zeros = tzeros(*zzz)
                self.poles = np.linalg.eigvals(zzz[0])



    # ===========================
    # tf class arithmetic methods
    # ===========================

    

    def __neg__(self):
        newnum = deepcopy(self._num)
        if not self._isSISO:
            for i in range(self._p):
                for j in range(self._m):
                    newnum[i][j] *= -1.0
        else:
            newnum = -newnum
        return tf(newnum,self._den,self._SamplingPeriod)

    def __add__(self,other):
        # SISO or MIMO switch
        if self._isSISO:
        # Handle the constant matrices, ints, floats, ss and tfs
            if isinstance(other,(int,float)):
                newnum = haroldpolyadd(other*self._den,self._num)
                return tf(newnum,self._den,self._SamplingPeriod)

            elif isinstance(other,(ss,tf)):
                if not other.shape == (1,1):
                    raise TypeError('The shapes of the systems '
                            'are not compatible for addition: '
                            '{0} vs. SISO.'.format(other.shape))

                elif isinstance(other,ss): return ss(
                    *tf2ss(self._num,self._den),
                    dt=self._SamplingPeriod) + other
                    
                elif isinstance(other,tf):
                    lcm,mults = haroldlcm(self._den,other.den)
                    newnum = haroldpolyadd(
                        np.convolve(self._num.flatten(),mults[0]),
                        np.convolve(other.num.flatten(),mults[1])
                            )

                    return tf(newnum,lcm)
            else:
                raise TypeError('I don\'t know how to add a '
                                '{0} to a transfer function '
                                '(yet).'.format(type(other).__name__))
        else:
            if isinstance(other,(int,float)):
                # ss addition is much better
                tempsys = ss(*tf2ss(self._num,self._den))
                tempsys.d += np.ones_like(tempsys.d)
                return tf(*(ss2tf(tempsys),self._SamplingPeriod))


            elif isinstance(other,(ss,tf)):
                if not other.shape == self._shape:
                    raise TypeError('The shapes of the systems '
                                'are not sdfcompatible for addition: '
                                '{0} vs. {1}.'.format(
                                                    other.shape,
                                                    self._shape
                                                    )
                                                )
                #shapes match
                tempsys_self = ss(*tf2ss(self._num,self._den),
                                      dt = self._SamplingPeriod)
                
                return tf(tempsys_self + other)
            else:
                raise TypeError('I don\'t know how to add a '
                                '{0} to a transfer function '
                                '(yet).'.format(type(other).__name__))



    def __radd__(self,other): return self + other

    def __sub__(self,other): return self + (-other)
        
    def __rsub__(self,other): return -self + other

    def __mul__(self,other):
        # SISO or MIMO switch
        if self._isSISO:
        # Handle the constant matrices, ints, floats, ss and tfs
            if isinstance(other,(int,float)):
                return tf(other*self._num,self._den,self._SamplingPeriod)

            if isinstance(other,(ss,tf)):
                if not other.shape == (1,1):
                    raise TypeError('The shapes of the systems '
                            'are not compatible for multiplication: '
                            '{0} vs. SISO.'.format(other.shape))

                if isinstance(other,ss): 
                    return other * ss(*tf2ss(self)) 
                else:
                    newnum = np.convolve(self._num,other.num)
                    newden = np.convolve(self._den,other.den)
                    return tf(newnum,newden)

            else:
                raise TypeError('I don\'t know how to multiply a '
                                '{0} with a transfer function '
                                '(yet).'.format(type(other).__name__))
        else:
            raise NotImplementedError('MIMO algebra is being implemented.')


    def __rmul__(self,other): return self * other

    # ================================================================
    # __getitem__ to provide input-output selection of a tf
    #
    # TODO: How to validate strides
    # ================================================================

#    def __getitem__(self,num_or_slice):
#        print('Lalala I"m not listening lalala')

    def __setitem__(self,*args):
        raise ValueError('To change the data of a subsystem, set directly\n'
                        'the relevant num,den or A,B,C,D attributes. '
                        'This might be\nincluded in the future though.')


    # ================================================================
    # __repr__ and __str__ to provide meaningful info about the system
    # The ascii art of matlab for tf won't be implemented.
    # Either proper image with proper superscripts or numbers.
    #
    # TODO: Approximate this to a finished product at least
    # ================================================================

    def __repr__(self):
        if self.SamplingSet=='R':
            desc_text = 'Continous-Time Transfer function with:\n'
        else:
            desc_text = ('Discrete-Time Transfer function with: '
                  'sampling time: {0:.3f} \n'.format(float(self.SamplingPeriod)))

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
        desc_text += '\n\n'+str('End of {0} object description'.format(
                                    __class__.__qualname__
                                ))
        return desc_text
        
#    def __str__(self):
#        return ''#rootprinter(self.poles)

    def validatepolymatrix(self,arg,*,name='Numerator'):
        """
        
        An internal command to validate whether given arguments to an
        ss instance are valid and compatible. I am sure I'll get comments
        on the religious Python way etc. That's not possible here because
        we are trying to support a syntax that is not a native datatype. 
    
        The intention is too funky to try/except hoops. Still I'm leaving 
        some room for being schooled. So if you can make a proper case
        I'm all ears. But I will ignore any sentence that only(!) argues 
        "Pythonic way".
        
        It also checks if the lists are 2D numpy.array'able entries.
        Otherwise it will explode somewhere further deep, leaving no 
        clue why the error happened. So better fail at the start.
        
        """
    
        # Imagine a 1x2 tf. The possible entries for num, den
        # that needs support are
        # num = [[np.array([1,2,3]),np.array([4,5,6])]] # Sane/horrible
        # num = [[[1,2,3],[4,5,6]]] # Sane/acceptable --> convert
            
    
        
    
    
        if isinstance(arg,(int,float)):# Excludes complex!
            return np.array(arg),'S'
        elif isinstance(arg,list):
            # Either 
            #        1. a simple unnested list --> SISO
            #        2. a list of lists to be np.array'd --> MIMO
    
            #------------
            # 1. Check whether all(!) elements are simple numbers
            if all([isinstance(x,(int,float)) for x in arg]):
                return np.array(arg),'S'
            #------------    
            # 2.Check first whether all(!) elements are also lists
            elif all([isinstance(x,list) for x in arg]):
                # Get the number of items in each list    
                m = [len(arg[ind]) for ind in range(len(arg))]
                # Check if the number of elements are consistent
                if max(m) == min(m):
                    try:
                        arg =  list(map(lambda x: list(map(
                            lambda y: np.asarray(y,dtype='float'),x)),arg))
                        return arg,'M'
                    except:
                        raise ValueError(# something was not floating 
                        'Something is not real scalar inside the MIMO '
                        '{0} list of lists.'.format(name))
                else:
                    raise IndexError(# element numbers of lists didn't match 
                    'MIMO {0} lists have inconsistent\n'
                    'number of entries, I\'ve found {1} '
                    'in one and {2} in another row'.format(name,max(m),min(m)))
            #------------  
            else:
                raise TypeError(# 
                '{0} starts with a list so I went in '
                'to find either \nreal scalars or more lists, but I found'
                ' other things that I won\'t mention.'.format(name))                
    
        elif (isinstance(arg,type(np.array([0.]))) # A numpy array
                    and
              arg.dtype in (float,int) # with numerical real data type
                    and
              arg.size > 0 # with at least one entry
             ):
    
            # If it is a 1D array, it classifies as a SISO entry. Because 
            # for MIMO intentions we need a LoL.
            # If it is a 2D array nxm with n,m>1 then it is a static gain
            # matrix and MIMO is assumed for now. 
        
            if arg.ndim > 1 and min(arg.shape) > 1:# e.g. np.eye(5)

                arg = np.array(arg,dtype='float') # get rid of ints
                arg_list = []
                for arg_row in arg:
                        arg_list += [[np.array(x) for x in arg_row]]
                return arg_list,'M'


            elif arg.ndim == 1 and arg.size > 1:# e.g. np.array([1,2,3])
                arg = np.array(arg,dtype='float') # get rid of ints
                return np.atleast_2d(arg),'S'

            else: # e.g. np.array(5)
                return np.atleast_2d(float(arg)),'S'
        else:
            raise TypeError(# Neither list,np.array nor scalar, reject.
            '{0} must either be a list of lists (MIMO)\n'
            'or a an unnested list (SISO). Numpy arrays or scalars' 
            'inside one-level lists such as\n[1.0] are also '
            'accepted. See the \"tf\" docstring'.format(name))


# End of Transfer Class

       
class ss:
    """
    
    ss is the one of two main system classes in harold (together with
    tf()). 
    
    A ss object can be instantiated in a straightforward manner by 
    entering 2D arrays. 
    
    >>>> G = ss([[0,1],[-4,-5]],[[0],[1]],[[1,0]],[1])
    
    
    However, the preferred way is to make everything a numpy array.
    That would skip many compatibility checks. Once created the shape 
    of the numerator and denominator cannot be changed. But compatible 
    sized arrays can be supplied and it will recalculate the pole/zero 
    locations etc. properties automatically.

    The Sampling Period can be given as a last argument or a keyword 
    with 'dt' key or changed later with the property access.
    
    >>>> G = ss([[0,1],[-4,-5]],[[0],[1]],[[1,0]],[1],0.5)
    >>>> G.SamplingSet
    'Z'
    >>>> G.SamplingPeriod
    0.5
    >>>> F = ss(1,2,3,4)
    >>>> F.SamplingSet
    'R'
    >>>> F.SamplingPeriod = 0.5
    >>>> F.SamplingSet
    'Z'
    >>>> F.SamplingPeriod
    0.5
    
    Setting  SamplingPeriod property to 'False' value to the will make 
    the system continous time again and relevant properties are reset
    to CT properties.

    Warning: Unlike matlab or other tools, a discrete time system 
    needs a specified sampling period (and possibly a discretization 
    method if applicable) because a model without a sampling period 
    doesn't make sense for analysis. If you don't care, then make up 
    a number, say, a million, since you don't care.

    
    """
    
    def __init__(self,a,b,c,d,dt=False):
        a,b,c,d = validateabcdmatrix(a,b,c,d)
        self._a , self._b , self._c , self._d = a,b,c,d
        self.SamplingPeriod = dt
        self._DiscretizedWith = None
        self._DiscretizationMatrix = None
        self._PrewarpFrequency = 0.
        self._m = self._b.shape[1]
        self._p = self._c.shape[0]

        # TODO: Add if ZERO DYNAMICS --> Empty SS!!!!
        self._isSISO = False
        self._shape = (self._p,self._m)
        if self._shape == (1,1):
            self._isSISO = True

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
        value,*_ =validateabcdmatrix(
            value,
            np.zeros_like(self._b),
            np.zeros_like(self._c),
            np.zeros_like(self._d)
            )
        self._a = value
        self._recalc()

    @b.setter
    def b(self,value):
        _,value,*_ =validateabcdmatrix(
            np.zeros_like(self._a),
            value,
            np.zeros_like(self._c),
            np.zeros_like(self._d)
            )
        self._b = value
        self._recalc()
            
    @c.setter
    def c(self,value):
        *_,value,_ =validateabcdmatrix(
            np.zeros_like(self._a),
            np.zeros_like(self._b),
            value,
            np.zeros_like(self._d)
            )
        self._c = value
        self._recalc()

    @d.setter
    def d(self,value):
        *_,value =validateabcdmatrix(
            np.zeros_like(self._a),
            np.zeros_like(self._b),
            np.zeros_like(self._c),
            value
            )
        self._d = value
#        self._recalc() # No need

        

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
        if value in KnownDiscretizationMethods:
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
           

    # FIXME: Zero dynamics will fail for SISO here and a few other places
    def _recalc(self):
            self.zeros = tzeros(self._a,self._b,self._c,self._d)
            self.poles = np.linalg.eigvals(self._a)




    # ===========================
    # ss class arithmetic methods
    # ===========================

    def __neg__(self):
        newC = deepcopy(self._c)
        newC = -1.*newC
        return ss(self._a, self._b, newC, self._d, self._SamplingPeriod)




    def __add__(self,other):
        if isinstance(other,ss):
            if self._shape == other.shape:
                adda = blkdiag(self._a,other.a)
                addb = np.vstack((self._b,other.b))
                addc = np.hstack((self._c,other.c))
                addd = self._d + other.d
                return ss(adda,addb,addc,addd)
            else:
                raise IndexError('Addition of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                self._shape,
                                                other.shape)
                                )
        if isinstance(other,tf):
            if self._shape == other.shape:
                return self + ss(*tf2ss(other))
            else:
                raise IndexError('Addition of systems requires their '
                                'shape to match but the system shapes '
                                'I got are {0} vs. {1}'.format(
                                                self._shape,
                                                other.shape)
                                )
        # Last chance                                
        if isinstance(other,(int,float)):
            addd = self._d + other*np.ones_like(self._d)
            return ss(self._a,self._b,self._c,addd)
        else:
            raise TypeError('I don\'t know how to add a '
                            '{0} to a state representation '
                            '(yet).'.format(type(other).__name__))



    
    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        pass

    def __rmul__(self,other):
        pass
    

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
#        desc_text += '\n\n'+str('End of object description %s') % __class__.__qualname__
        return desc_text



def validateabcdmatrix(a,b,c,d):
    """
    
    An internal command to validate whether given arguments to an
    ss instance are valid and compatible.
    
    It also checks if the lists are 2D numpy.array'able entries.
    Otherwise it will explode somewhere further deep, leaving no 
    clue why the error happened. So better fail at type checking.
    
    See also the docstring of "validatepolymatrix()"
    
    """
    # Start type checking
    for y,x in enumerate((a,b,c,d)):
        
        # Check for obvious choices        
        
        if not isinstance(x,(int,float,list,type(np.array([0.])))):
            raise TypeError('{0} matrix should be, regardless of the shape,'
                            ' an int, float, list or,\n'
                            'much better, a properly typed 2D Numpy '
                            'array. Instead I found a {1} object.'.format(
                                ['A','B','C','D'][y] ,
                                type(x).__name__
                                )
                            )

        if isinstance(x,list):
            if not (

                (
                 all([isinstance(z,list) for z in x]) 
                 and
                 all([all([isinstance(w,(int,float))for  w in z]) for z in x])
                )
                     
                or 
                     
                all([isinstance(y,(int,float)) for y in x])
            ):
                raise TypeError('{0} starts with a list so I went in '
                'to find either all \nreal scalars or 1D lists with '
                ' real scalar entries, but I '
                'found other things.'.format(['A','B','C','D'][y]))
            
            # Also count the elements in each list of lists
            if all([isinstance(z,list) for z in x]):
                m = [len(x[ind]) for ind in range(len(x))]
                if max(m)!=min(m):
                    raise IndexError('Rows for {0} matrix have '
                                    'inconsistent number of elements.\n'
                                    'I found {1} in one and {2} in '
                                    'another row'.format(
                                            ['A','B','C','D'][y],
                                            max(m),
                                            min(m)
                                            )
                                    )

    # Looks OK so far. Start Numpy 2D arrays and shape checking
    a,b,c,d = map(np.atleast_2d,(a,b,c,d))
    #  Here check everything is compatible
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
    
    if d.shape != (c.shape[0],b.shape[1]):
        raise ValueError('D matrix must have the same number of rows/'
                        'columns \nwith C/B matrices. I need the shape '
                        '({0[0]:d},{0[1]:d}) but got ({1[0]:d},'
                        '{1[1]:d}).'.format(
                                        (c.shape[0],b.shape[1]),
                                         d.shape
                                        )
                        )
    return a,b,c,d
            
    


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

def tzeros(A,B,C,D):
    """

    Computes the transmission zeros of a (A,B,C,D) system matrix quartet. 

    This is a straightforward implementation of the algorithm of Misra, 
    van Dooren, Varga 1994 but skipping the descriptor matrix which in 
    turn becomes Emami-Naeini,van Dooren 1979. I don't know if anyone 
    actually uses descriptor systems in practice so I removed the 
    descriptor parts to reduce the clutter. Hence, it is possible to 
    directly row/column compress the matrices without caring about the 
    upper Hessenbergness of E matrix. 


    """    
    
    n,_ = np.shape(A)
    p,m = np.shape(D)
    r = np.linalg.matrix_rank(D)
    if n < 1:
        z = np.zeros((0,1))
        
    if (p==1 and m==1 and r>0) or (r == min(p,m) and p==m):
        z = tzeros_final_compress(A,B,C,D,n,p,m)
        return z
    else:# Reduction needed
        if r == p:
            Ar,Br,Cr,Dr = (A,B,C,D)
        else:
            Ar,Br,Cr,Dr = tzeros_reduce(A,B,C,D)

        # Are we done ? 
        if p!=m: # Square and full rank. Done! Otherwise:
            Arc,Brc,Crc,Drc = tzeros_reduce(Ar.T,Cr.T,Br.T,Dr.T)
        else:
            Arc,Brc,Crc,Drc = (Ar,Br,Cr,Dr)
        
        n,_ = np.shape(Arc)
        p,m = np.shape(Drc)
        if n!=0:# Are there any state left to compute zeros for? 
            z = tzeros_final_compress(Arc,Brc,Crc,Drc,n,p,m)
        else:# No zeros --> Empty array
            z = np.zeros((0,1))
        return z
        
def tzeros_final_compress(A,B,C,D,n,p,m):
    """
    Internal command for finding the Schur form of a full rank and 
    row/column compressed C,D pair. 
    
    TODO: Clean up the numerical noise and switch to Householder 
    """     

    _,_,v = np.linalg.svd(np.hstack((D,C)),full_matrices=True)
    T = np.hstack((A,B)).dot(np.roll(np.roll(v.T,-m,axis=0),-m,axis=1))
    S = blkdiag(
            np.eye(n),
            np.zeros((p,m))
            ).dot(np.roll(np.roll(v.T,-m,axis=0),-m,axis=1))
    a,b,_,_ = sp.linalg.qz(S[:n,:n],T[:n,:n],output='complex')
    z = np.diag(b)/np.diag(a)
    # TODO : Occasionally z will include 10^15-10^16 entries instead of 
    # infinite zeros. Decide on a reasonable bound to discard.
    return z
    
def tzeros_reduce(A,B,C,D):
    while True:
        p,m = np.shape(D)
        n,_ = np.shape(A)
        t=0
        u,s,v = haroldsvd(D)
        u = np.real(u)
        Dt = s.dot(v)
        for i in np.arange(p,0,-1):
            if np.all(Dt[i-1,]==0):
                t = t+1;
                continue
            else:
                break
        if t == 0:
            Ar=A;Br=B;Cr=C;Dr=D;
            break
        Ct = u.T.dot(C)
        Ct = Ct[-t:,]
        mm = np.linalg.matrix_rank(Ct)
        vc = np.linalg.svd(Ct,full_matrices=True)[2]
        T = np.roll(vc.T,-mm,axis=1)
        Sysmat = blkdiag(T,u).T.dot(
            np.vstack((
                np.hstack((A,B)),np.hstack((C,D))
            )).dot(blkdiag(T,np.eye(m)))
            )
        Sysmat = np.delete(Sysmat,np.s_[-t:],0)
        Sysmat = np.delete(Sysmat,np.s_[n-mm:n],1)
        A = Sysmat[:n-mm,:n-mm]
        B = Sysmat[:n-mm,n-mm:]
        C = Sysmat[n-mm:,:n-mm]
        D = Sysmat[n-mm:,n-mm:]
        if A.size==0:
            break
    return A,B,C,D

# %% State <--> Transfer conversion

# Implements MIMO-compatible-tf2ss ala Varga,Sima 1981
# Compute the controllability matrix, get the svd, isolate the c'ble part
# Pertranspose the system, do the same to get the o'ble part of the c'ble part
# Then iterate over all row/cols of B and C to get SISO TFs via c(sI-A)b+d

def ss2tf(G):
    if not isinstance(G,ss):
        raise TypeError('The argument is not a state space '
                        'representation but a \"{0}\" object.\n'
                        'Hence, I\'ll pretend that it is your '
                        'fault.'.format(type(G).__name__))
    A,B,C,D = G.a,G.b,G.c,G.d
    if np.shape(A) == 0:
        raise ValueError 
    n = A.shape[0]

    if not iscontrollable(A,B):
        T,rco = ctrb(A,B)[1:]
        A = T.T.dot(A.dot(T))[:rco,:rco]
        B = T.T.dot(B)[:rco,:]
        C = C.dot(T)[:,:rco]

    if A.size == 0:
        return D,np.ones_like(D)
    
    n = A.shape[0]



    if not isobservable(G):
        S,rob = obsv(A,C)[1:]
#            _,_,S = np.linalg.svd(Cob)
        A = (S.T).dot(A.dot(S))[:rob,:rob]
        B = (S.T).dot(B)[:rob,:]
        C = C.dot(S)[:,:rob]
    if A.size == 0:
        return D,np.ones_like(D)

    p,m = C.shape[0],B.shape[1]
    n = np.shape(A)[0]
    pp = np.linalg.eigvals(A)
    
    entry_den = np.real(haroldpoly(pp))
    # Allocate some list objects for num and den entries
    num_list = [[None]*m for n in range(p)] 
    den_list = [[entry_den]*m for n in range(p)] 
    
    
    for rowind in range(p):# All rows of C
        for colind in range(m):# All columns of B

            b = B[:,colind:colind+1]
            c = C[rowind:rowind+1,:]
            # zz might contain noisy imaginary numbers but since 
            # the result should be a real polynomial, we can get 
            # away with it (on paper)

            zz = tzeros(A,b,c,np.array([[0]]))

            # For finding k of a G(s) we compute
            #        pole poly evaluated at s0
            # G(s0)*---------------------------
            #        zero poly evaluated at s0
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

    if G.SamplingSet == 'R':
        return num_list, den_list
    else:
        return num_list, den_list , G.SamplingPeriod

    #FIXME : Resulting TFs are not minimal per se. simplify them

def transfertostate(tf_or_numden):
    # mildly check if we have a transfer,state, or (num,den)
    if isinstance(tf_or_numden,tuple):
        G = Transfer(*tf_or_numden)
        num = G.num
        den = G.den
        m,p = G.NumberOfInputs,dummy_G.NumberOfOutputs
    elif isinstance(tf_or_numden,ss):
        return tf_or_numden
    else:
        try:
            num = tf_or_numden.num
            den = tf_or_numden.den
            m,p = G.NumberOfInputs,G.NumberOfOutputs
        except AttributeError: 
            raise TypeError('I\'ve checked the argument for being a' 
                   ' Transfer, a State,\nor a tuple for (num,den) but'
                   ' none of them turned out to be the\ncase. Hence'
                   ' I don\'t know how to convert this to a State object.')


    it_is_a_gain = G._isgain


    # Check if it is just a gain
    if it_is_a_gain:
        # TODO: Finish State._isgain object property
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
        if den[0] != 1.:
            d = den[0]
            num,den = num/d,den/d
            
        if len(num) < len(den):
            C = np.zeros((1,len(den)-1))
            C[0,:len(num)] = num[::-1]
            D = np.array([[0]])
        else:
            C = num - den*num[0]
            C = np.array(C[-1:0:-1],ndmin=2)
            D = np.array([[num[0]]])
            
    else:# MIMO... Implement a "Wolowich LMS-Section 4.4 (1974)"-variant

        # Extract D matrix
        D = np.zeros((p,m))

        for x in range(p):
            for y in range(m):
                
                # Possible cases (not minimality only properness !!!): 
                # 1.  3s^2+5s+3 / s^2+5s+3  Proper
                # 2.  s+1 / s^2+5s+3        Strictly proper
                # 3.  s+1 / s+1             Full cancellation
                # 4.  3   /  2              Just gains

                datanum = haroldtrimleftzeros(num[x][y])
                dataden = haroldtrimleftzeros(den[x][y])
                nn , nd = len(datanum) , len(dataden)

                if nd == 1: # Case 4 : nn should also be 1.
                    D[x,y] = datanum/dataden
                    num[x][y] = np.array([0.])

                elif nd > nn: # Case 2 : D[x,y] is trivially zero
                    pass # D[x,y] is already 0.

                else:
                    NumOrEmpty , datanum = haroldpolydiv(datanum,dataden)
                    
                    # Case 3: If all cancelled datanum is returned empty
                    if datanum.size==0:
                        D[x,y] = NumOrEmpty
                        num[x][y] = np.array([0.])
                        den[x][y] = np.array([1.])
                        
                    # Case 1: Proper case
                    else:
                        D[x,y] = NumOrEmpty
                        num[x][y] = datanum

#        for x in range(p):
#            for y in range(m):

                # Make the denominator entries monic
                if den[x][y][0] != 1.:
                    if np.abs(den[x][y][0])<1e-5:
                        print(
                          'tf2ss Warning:\n The leading coefficient '
                          'of the ({0},{1}) denominator entry is too '
                          'small (<1e-5). Expect some nonsense in the '
                          'state space matrices.'.format(x,y),end='\n')
                          
                    num[x][y] = np.array([1/den[x][y][0]])*num[x][y]
                    den[x][y] = np.array([1/den[x][y][0]])*den[x][y]

        # OK first check if the denominator is common in all entries
        if all([np.array_equal(den[x][y],den[0][0])
            for x in range(len(den)) for y in range(len(den[0]))]):

            # Nice, less work. Off to realization. Decide rows or cols?
            if p >= m:# Tall or square matrix => Right Coprime Fact.
               factorside = 'r'
            else:# Fat matrix, pertranspose the LoL => LCF.
               factorside = 'l'
               den = [list(i) for i in zip(*den)]
               num = [list(i) for i in zip(*num)]
               p,m = m,p

            d = den[0][0].size-1
            A = haroldcompanion(den[0][0])
            B = np.vstack((np.zeros((A.shape[0]-1,1)),1))
            t1 , t2 = A , B

            for x in range(m-1):
                A = blkdiag(A,t1)
                B = blkdiag(B,t2)
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
                    num[y][x] = haroldpolymul(num[y][x],mults[y],
                                                        trimzeros=False)

            coldegrees = [x.size-1 for x in den[0]]

            A = haroldcompanion(den[0][0])
            B = np.zeros((A.shape[0],1))
            B[-1] = 1.

            for x in range(1,m):
                Atemp = haroldcompanion(den[0][x])
                Btemp = np.zeros((Atemp.shape[0],1))
                Btemp[-1] = 1.
                A = blkdiag(A,Atemp)
                B = blkdiag(B,Btemp)

            n = A.shape[0]
            C = np.zeros((p,n))
            k = 0
            for y in range(m):
                for x in range(p):
                    C[x,k:k+num[x][y].size] = num[x][y][::-1]
                k += coldegrees[y] 
            
            if factorside == 'l':
                A, B, C = A.T, C.T, B.T
            
    try:# if the arg was a tf object
        if G.SamplingSet == 'R':
            return A,B,C,D
        else:
            return A,B,C,D,G.SamplingPeriod
    except AttributeError:# the arg was num,den
        return A,B,C,D

# %% Continous - Discrete Conversions

def discretize(G,dt,method='tustin',PrewarpAt = 0.,q=None):
    if not isinstance(G,(tf,ss)):
        raise TypeError('I can only convert ss or tf objects but I '
                        'found a \"{0}\" object.'.format(type(G).__name__)
                        )
    if G.SamplingSet == 'Z':
        raise TypeError('The argument is already modeled as a '
                        'discrete-time system.')

    if isinstance(G,tf):
        T = ss(*tf2ss(G))
    else:
        T = G

    args = __discretize(T,dt,method,PrewarpAt,q)

    if isinstance(G,ss):
        Gd = ss(*args)
        Gd.DiscretizedWith = method
    else:
        Gss = ss(*args)
        Gd = tf(*ss2tf(Gss))
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
                        ''.format(KnownDiscretizationMethods)
                        )
                        
    return Ad , Bd , Cd , Dd , dt


def undiscretize(G,OverrideWith = None):
    if not isinstance(G,(tf,ss)):
        raise TypeError('The argument is not transfer '
        'function or a state\nspace model.'
            )

    if G.SamplingSet == 'R':
        raise TypeError('The argument is already modeled as a '
                        'continuous time system.')


    args = __undiscretize(G)

    if isinstance(G,ss):
        Gc = ss(*args)
    else:
        Gss = ss(*args)
        Gc = tf(*ss2tf(ss(*args)))
        
    return Gc


def __undiscretize(G):

    if isinstance(G,tf):
        T = ss(*tf2ss(G))
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

    
def ctrb(G,*args):
    try:
        A = G.a
        B = G.b
    except AttributeError:
        if not isinstance(G,type(np.array([0]))):
            raise TypeError(('ctrb() expects either a state-space system or a'
                             ' numpy 2D array as a first argument.\nI got a '
                             '\"{0}\" and I don\'t know what to do with'
                             ' it.').format(type(G).__name__))
        elif not len(args)==0:
            A = G
            B = args[0]
            if not A.shape[0] == B.shape[0]:
                raise IndexError('A and B should have same number of '
                                'rows.However what I got is {0} vs. '
                                '{1}'.format(A.shape[0],B.shape[0]))
        else:
            raise ValueError('I found a matrix and assumed it was matrix A. '
                            'But I don\'t have matrix B, use either\n'
                            'ctrb(<some ss system>) or ctrb(<2d numpy array>,'
                            '<2d numpy array>) with suitable dimensions.')

        
    n = A.shape[0]
    Cc = B.copy()
    
    
    
    for i in range(1,n):# Append AB,A^2B....A^(n-1)B
        Cc = np.hstack((Cc,np.linalg.matrix_power(A,i).dot(B)))
    r = np.linalg.matrix_rank(Cc)
    T = haroldsvd(Cc)[0]

    return Cc,T,r
    
def obsv(G,*args):
    try:
        A = G.a
        C = G.c
    except AttributeError:
        if not isinstance(G,type(np.array([0]))):
            raise TypeError(('obsv() expects either a state-space system or a'
                             ' numpy 2D array as a first argument.\nI got a '
                             '\"{0}\" and I don\'t know what to do with'
                             ' it.').format(type(G).__name__))
        elif not len(args)==0:
            A = G
            C = args[0]
            if not A.shape[1] == C.shape[1]:
                raise IndexError('A and C should have same number of '
                                'columns.However what I got is {0} vs. '
                                '{1}'.format(A.shape[1],C.shape[1]))            
        else:
            raise ValueError('I found a matrix and assumed it was matrix A. '
                            'But I don\'t have matrix C, use either\n'
                            'obsv(<some ss system>) or obsv(<2d numpy array>,'
                            '<2d numpy array>) with suitable dimensions.')

        
    n = A.shape[0]
    Co = C.copy()

    for i in range(1,n):# Append CA,CA^2....CA^(n-1)
        Co = np.vstack((Co,C.dot(np.linalg.matrix_power(A,i))))
    r = np.linalg.matrix_rank(Co)
    T = haroldsvd(Co)[2].T

    return Co,T,r

    
def iscontrollable(G,B=None):
    if not B is None:
        if ctrb(G,B)[2]==G.shape[0]:
            return True
        else:
            return False
            
    elif isinstance(G,(ss,tf)):
        if ctrb(G)[2]==G.a.shape[0]:
            return True
        else:
            return False

def isobservable(G,C=None):
    if not C is None:
        if obsv(G,C)[2]==G.shape[1]:
            return True
        else:
            return False
            
    elif isinstance(G,(ss,tf)):
        if obsv(G)[2]==G.a.shape[1]:
            return True
        else:
            return False

def kalmandecomposition(G):
    """
    pass
    """    
    
#    #TODO : Type checking
#    if iscontrollable(G):
#        if isobservable(G):
#            T = np.eye(G.a.shape[0])
#            Gd = G
#        else:
#            pass
#        # How to copy the whole ss object by just changing the ABCD data? 
#    return Gd,T


        
# %% Linear algebra ops

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
            P = blkdiag(np.eye(n-ub.T.shape[0]),ub.T)

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
                    P = blkdiag(np.eye(n-uh3.shape[1]),uh3.T).dot(P)
                A0[ROI_start:,ROI_start:] = np.r_[
                                    np.c_[h1,h2],
                                    np.c_[sh3.dot(vh3),uh3.T.dot(h4)]
                                    ]
                A0 = A0.dot(blkdiag(np.eye(n-uh3.shape[1]),uh3))
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

def canceldist(F,G):
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



def minimalrealization(A,B,C,mu_tol=1e-6):
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
    
    Uses canceldist(), and staircase() for the aforementioned checks. 
    
    Parameters
    ----------
    A,B,C : {(n,n), (n,m), (pxn)} array_like
        System matrices to be checked for minimality
    mu_tol: float (default 1-e6)
        The sensitivity threshold for the cancellation to be compared 
        with the first default output of canceldist() function.

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
        
        kc = canceldist(A,B)[0]
        ko = canceldist(A.T,C.T)[0]
        
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
                        kc_mod = canceldist(Ac_mod,Bc_mod)[0]

                kc = kc_mod
                # Fake an iterable to fool the sum below
                blocks_c = [sum(blocks_c)-Acm.shape[0]]


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
                        ko_mod = canceldist(Ao_mod,Bo_mod)[0]

                
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
    if not isinstance(G,ss):
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
def blkdiag(*args):
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

def redheffer(Amatrix,Bmatrix,A22shape,B11shape):
    """
    pass
    """    
    
#    if not A22shape[::-1] == B11shape:
#        raise ValueError('The shape of (2,2) block of the first matrix '
#                         'must be compatible\nwith the (1,1) block of '
#                         'the second matrix such that A22*B11 and '
#                         'B11*A22 are well-defined.')
#                         
#    ap , am = A22shape
#    
    

# %% Polynomial ops    
def haroldlcm(*args):
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
    a = blkdiag(*tuple(map(haroldcompanion,poppedargs))) # Companion A
    b = np.concatenate(tuple(map(lambda x: eyecolumn(x-1,-1),
                                 [z.size for z in poppedargs])))# Companion B
    c = blkdiag(*tuple(map(lambda x: eyecolumn(x-1,0).T,
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
        temp2 = blkdiag(np.linalg.inv(temp[:i-1,:i-1]),np.eye(n+1-i))
    

    lcmpoly= temp2.dot(-temp)[:i-1,-1]
    # Add monic coefficient and flip
    lcmpoly= np.append(lcmpoly,1)[::-1]
    
    # TODO: Below is the multipliers of entries to be completed to LCM.
    # Decide whether this output should be optional or not
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
    
                adj(sI-A_lcm) * blkdiag(B_lcm)
    
    The resulting row entries are added to each other to get adj(sI-A)*B_lcm
    Finally, since we now have a single column we can treat polynomial
    entries as matrix entries hence multiplied with c matrix properly. 
    
    TODO: Good luck polishing this explanation...
    
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
                
    # TODO: make this bound optional
    # Remove pseudoinverse noise as if there is no tomorrow
    # If we need those entries, we have bigger problems than this.
    lcmpoly[abs(lcmpoly)<1e-9] = 0.
    mults[abs(mults)<1e-9] = 0.
    mults = [haroldtrimleftzeros(z) for z in mults]
    return lcmpoly, mults


def haroldgcd(*args):
    """
    Takes *args-many 1D numpy arrays and computes the numerical 
    greatest common divisor polynomial. The polynomials are
    assumed to be in decreasing powers, e.g. s^2 + 5 should
    be given as numpy.array([1,0,5])
    
    Returns a numpy array holding the polynomial coefficients
    of GCD. 
    
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
                 
    NOTES : I've tried the recent ERES method too. When there is a nontrivial
           GCD it performed satisfactorily however did not perform as well
           when GCD = 1 (maybe due to my implementation). Hence I've switched
           to matrix-based methods.

    """    


    if not all([isinstance(x,type(np.array([0]))) for x in args]):
        raise TypeError('Some arguments are not numpy arrays for GCD')

    
    lefttrimmed_args = list(map(haroldtrimleftzeros,args))
    deglist = np.array([len(x) for x in lefttrimmed_args])
    maxdeg = max(deglist)
    maxdegindex = next(
        ind for ind,x in enumerate(lefttrimmed_args) if len(x)==maxdeg
        )
    try:
        secondmaxdeg = max(deglist[deglist<maxdeg])
    except ValueError:# all degrees are the same?
        if not all(deglist-maxdeg):
            secondmaxdeg = maxdeg
        else:
            raise TypeError("Something is wrong with the array"
                            "lengths in GCD.")
        
            
    n,p,m = maxdeg-1,secondmaxdeg-1,len(args)-1
    
    S = np.c_[
            np.array([lefttrimmed_args.pop(maxdegindex)]*p),
            np.zeros((p,p-1))
            ]
    
    for rows in range(S.shape[0]):
        S[rows] = np.roll(S[rows],rows)

    for poly in lefttrimmed_args:

        temp = np.c_[
                 np.array([poly]*n),
                 np.zeros((n,n+p-len(poly)))
                 ]
        for rows in range(temp.shape[0]):
            temp[rows] = np.roll(temp[rows],rows)

        S = np.r_[S,temp]

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

    # Kind of normalize for indexing with at_least2d
    if any(somearray):# if not completely zero
        try:
            n = next(x for x,y in enumerate(np.atleast_2d(somearray)[0]) 
                        if y != 0.)
            return np.array(somearray[n::])
        except StopIteration:
            return np.array(somearray[::])
    else:
        return np.array([])
        
        
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
    if trimzeros:
        trimmedargs = tuple(map(haroldtrimleftzeros,args))
    else:
        trimmedargs = args
    p = trimmedargs[0]
    for x in trimmedargs[1:]:
        p = np.convolve(p,x)
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
    
# %% Plotting - Frequency Domain
