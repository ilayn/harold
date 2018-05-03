System Representations
======================

For creating dynamic/static models, ``harold`` currently offers two options: 
A ``State()`` and a ``Transfer()`` object for representing systems with 
state models and transfer matrices. 

Creating models
----------------
 
State models
^^^^^^^^^^^^^^^^^^

The initialization of these objects are pretty straightforward. Note that you 
can skip ``D`` term and it will be assumed to be zero::
    
    >>> G = State([[1,2],[3,4]],[[1],[0]],[1,2])
    >>> G = State([[1,2],[3,4]],[[1],[0]],[1,2], dt = 0.1)

In the second example, we create a discrete-time model but to avoid the clash with
a ``D`` term we add the ``dt`` keyword explicitly. If there was also a nonzero 
feedthrough element then we can also skip that too::

    >>> G = State([[1,2],[3,4]],[[1],[0]],[1,2],1,0.1)

As obvious to everyone who used this syntax even in matlab's convenient bracket
semicolon notation, creating four individual matrices everytime just to pass
to the function becomes increasingly annoying. Instead a matrix slicer is 
available in ``harold`` ::

    >>> M = np.array([[1,2,1],[3,4,0],[1,2,0]])
    >>> G = State(*matrix_slice(M,corner shape=(2,2), corner='nw'))

As shown above, the model creation is a straightforward enumeration of 
the involved ``A,B,C,D`` matrices in the arguments. For discrete time
models, you simply add a fifth argument, if you have not omitted any
or explicitly mentioning ``dt=<sampling period>`` as an argument. 
    
To create a static models, just provide an array ::

    >>> G = State(1)
    >>> G = State(np.ones((5,3)), dt=0.5)
    
Here again, ``dt=<sampling period>`` should be provided explicitly as it would
be confused about the second argument being the ``B`` element. ::

    >>> G = State(1,0.001)    # Will lead to error
    >>> G = State(1,dt=0.001) # Will work

If the model is discretized we can also check ::

    >>> G.SamplingPeriod  # returns the sampling period
    >>> G.SamplingSet     # returns 'Z' for discrete-time, 'R' otherwise
    >>> G.DiscretizedWith # returns the discretization method if applicable

These make sure that the discretization remembers how it got there in 
the first place if harold is used. Or if the model is already given 
as a discrete time model, the method can be set such that ``undiscretize``
can use the correct method. 

.. autoclass:: harold.State
    :members:

Transfer models
^^^^^^^^^^^^^^^^^^^^^

Similarly, Transfer models can be created via ::

    >>> H = Transfer([1,2,3],[7,5,3])

As mentioned previously, numpy array syntax is strange and a bit 
verbose. Hence, it makes it difficult to type every time ``np.array``
or some other alias for creation of an array tobe used in the transfer 
representation definitions. Hence, harold actually goes a long way to 
make sense what is entered for the ``Transfer()`` initialization. 

First, it can understand even if the user enters a scalar or a Python 
list, instead of numpy arrays. It will be checked and converted if the 
input is sensible. ::

    >>> G = Transfer(1,[1,2,3])
    >>> G = Transfer(1,[[1,2,3]])

The second example might confuse the user since it will spit out a 
transfer representation of a 1x3 static gain. 

What happens is that when the parses find a list of lists, it assumes
that the user is trying to create a MIMO object. Thus, it changes its 
context to make sense with the missing information. It first checks 
that numerator has a single element and thus assumes that this is a 
common numerator. Then it parses the denominator and finds only scalars
thus creates the static gain of fractions one, one half, and one third. 

Discrete time models are handled similarly. 


.. autoclass:: harold.Transfer
    :members:
	
Model Arithmetic
-------------------

Both ``Transfer`` and ``State`` instances support basic model arithmetic. 
You can add/multiply/subtract models that are compatible (division is 
a completely different story hence omitted). Again, `harold` tries its
best to explain what went wrong. Let's take the same discrete time 
model and set another random MIMO model ``H`` with 3 states, ::

    >>> G = State(-1, [1, 2] , [[1], [1]], dt=0.01)
    >>> H = State(*matrix_slice(np.random.rand(5, 6), (3, 3)))
    >>> G*H
    ValueError: The sampling periods don't match so I cannot multiply these systems.

Even when both are forced to be continuous ::

    >>> G = State(-1, [1, 2] , [[1], [1]])
    >>> H = State(*matrix_slice(np.random.rand(5, 6), (3, 3)))
    >>> H*G
    ValueError: Shapes are not compatible for multiplication. Model shapes are (2, 3) and (2, 2)
    
Notice the system sizes are repeated in the error message hence we 
don't need to constantly check which part is the culprit for both
systems. For `Transfer` models, another useful property is recognition of 
common poles when doing simple addition/subtraction. For example, ::

	>>> G = Transfer([1,1],[1,2])
	>>> H = Transfer([1],[1,3,2])
	
	>>> F = G+H
	>>> F.polynomials #check num,den
	(array([[ 1.,  2.,  2.]]), array([[ 1.,  3.,  2.]]))
	
As you can see the common terms are identified such that the model order does
not increase artificially for numerically well-conditioned expressions.
Minimality is not guaranteed.

.. note:: This is not the case for ``State`` instances that is state matrices
    are directly augmented without any cancellation checks.


Context Discovery
------------------

For both ``State()`` and ``Transfer()`` argument parsing, ``harold`` can 
tell what happened during the context discovery. For that, there is a
``validate_arguments()`` class method for each class. This will return
the regularized version of the input arguments and also include a flag
in case the resulting system is a static gain::

    >>> Transfer.validate_arguments(1,[[1,2,3]],verbose=1)

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
    In the MIMO context and proper entries, I've found
    scalar denominator entries hence flagging as a static gain.
    Out[2]: 
    ([[array([[1.]]), array([[1.]]), array([[1.]])]],
     [[array([[1.]]), array([[2.]]), array([[3.]])]],
     (1, 3),
     True)

As seen from the resulting arrays, the numerator is now three numpy float 
arrays containing the common entry. This is because the denominator is given
as a list of lists which is taken as MIMO intention. Both the numerator and
denominator are converted to list of lists. 

This method can also be used to verify whether a certain input is a valid
argument for creating model objects hence the name. 

Same class method is also available for the ``State()`` class. 
