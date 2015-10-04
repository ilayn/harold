Model Objects
=============

For creating dynamic/static models, harold offers basically two options: 
A ``State()`` and a ``Transfer()`` object. The initialization of these 
objects are pretty straightforward. ::
    
    G = State([[1,2],[3,4]],[[1],[0]],[1,2],0)
    H = Transfer([1,2,3],[7,5,3])
    
Note that, in Python, the calling order of the arguments are not fixed. 
You can also use the following ::

    G = State(d=0,c=[1,2],[[1,2],[3,4]],[[1],[0]])
    H = Transfer([1,2,3],[7,5,3])


As obvious to everyone who used this syntax even in matlab's convenient
bracket semicolon notation, creating these 4 individual matrices first 
just to pass to the function becomes increasingly annoying. Instead a 
matrix slicer is available in harold::

    M = np.array([[1,2,1],[3,4,0],[1,2,0]])
    G = State(*ssslice(M,2))
    

What happened here is that, first ``ssslice()`` function took the ``M``
matrix and sliced it such that its upper left block size is 
:math:`2\times 2` then when Python encountered the ``*`` notation it 
grabbed the result of this operation as separate entities and unpacked 
the arguments as if we have provided four separate arguments. I am 
sure you will get addicted to this and then we will talk about matlab
programming a bit better informed.

.. note:: For slicing the matrix with a rectangular upper left element,
    harold also has a ``matrixslice()`` function. 

Dealing with models
-------------------
 
``State()`` models
^^^^^^^^^^^^^^^^^^

As shown above, the model creation is a straightforward enumeration of 
the involved ``A,B,C,D`` matrices in the arguments. For discrete time
models, you simply add a fifth argument, if you have not omitted any
(which is useful in some context but will be documented later),
or explicitly mentioning ``dt=<sampling period>`` as an argument. 
Example::

    G = State(-1,1,1,0,0.2)
    G = State([[1,0],[-2,-5]],[[1],[0]],[0,1],0,dt=0.01)
    
To create a static system, a gain matrix(scalar) but still having a 
state representation, then it is possible to just provide a gain matrix ::

    G = State(1)
    G = State(np.ones((5,3))
    
If there is also the discrete time property that is needed to be 
specified, then ``dt=<sampling period>`` should be provided explicitly
as harold would be confused about the second argument being the ``B``
element. ::

    G = State(1,0.001)    # Will lead to error
    G = State(1,dt=0.001) # Will work
    
``Transfer()`` models
^^^^^^^^^^^^^^^^^^^^^

As mentioned previously, numpy array syntax is strange and a bit 
verbose. Hence, it makes it difficult to type every time ``np.array``
or some other alias for creation of an array tobe used in the transfer 
representation definitions. Hence, harold actually goes a long way to 
make sense what is entered for the ``Transfer()`` initialization. 

First, it can understand even if the user enters a scalar or a Python 
list, instead of numpy arrays. It will be checked and converted if the 
input is sensible. ::

    G = Transfer(1,[1,2,3])
    G = Transfer(1,[[1,2,3]])

The second example might confuse the user since it will spit out a 
transfer representation of a :math:`1\times 3` static gain. 

What happens is that when the parses find a list of lists, it assumes
that the user is trying to create a MIMO object. Thus, it changes its 
context to make sense with the missing information. It first checks 
that numerator has a single element and thus assumes that this is a 
common numerator. Then it parses the denominator and finds only scalars
thus creates the static gain of fractions one, one half, and one third. 

Discrete time models are handled similarly. 


Context Discovery
^^^^^^^^^^^^^^^^^^

For both ``State()`` and ``Transfer()`` argument parsing, harold can 
tell what happened during the context discovery. For that, there is a
``validate_arguments()`` class method for each class. This will return
the regularized version of the input arguments and also include a flag
in case the resulting system is a static gain::

    Transfer.validate_arguments(1,[[1,2,3]],verbose=1)

will print out the following for the example we discussed above ::

    ========================================
    Handling numerator
    ========================================
    I found only a float
    ========================================
    Handling denominator
    ========================================
    I found a list
    I found a list that has only lists
    Every row has consistent number of elements
    ==================================================
    Handling raw entries are done.
    Now checking the SISO/MIMO context and regularization.
    ==================================================
    One of the MIMO flags are true
    Denominator is MIMO, Numerator is something else
    Denominator is MIMO, Numerator is SISO

    ([[array([[ 1.]]), array([[ 1.]]), array([[ 1.]])]],
     [[array([[ 1.]]), array([[ 2.]]), array([[ 3.]])]],
     (1, 3),
     True

As seen from the resulting arrays, the numerator is now 
three numpy float arrays containing the common entry. 
Both the numerator and denominator are converted to list
of lists. 

Same class method is also available for the ``State()`` class. 


Discretization Methods
----------------------

In harold, a discrete time model can keep the discretization method 
in mind such that when the occasion arises to convert back to a 
continuous model the proper method is chosen. 

Currently, the known discretization methods are given as 

======================= ========================
Method                  Aliases
----------------------- ------------------------
``bilinear``            ``tustin``
----------------------- ------------------------
``forward difference``  ``forward rectangular``

                        ``forward euler``

                        ``>>``
----------------------- ------------------------
``backward difference``  ``backward rectangular``

                         ``backward euler``

                         ``<<``
----------------------- ------------------------
``zoh``
----------------------- ------------------------
``lft``
======================= ========================

.. todo:: Explain these methods!!
