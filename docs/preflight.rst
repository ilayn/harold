Getting Started  
===============

The user is assumed to have some acquientance with Python, NumPy, and SciPy in
order to go through the tutorial. However, it is strongly recommended to get
versed in these impressive tools regardless.
    
.. note :: It was actually 2014 that Python developers included a symbol for matrix
    multiplication that is the ``@`` character (again almost everything else is
    reserved and the regular ``*`` is meant for element-wise multiplication).

    Another point that might annoy the users is the complex number syntax. In Python,
    you are obliged to use the letter ``j`` for the complex unit but cannot use ``i``.


.. admonition :: Tip : An almost exhaustive NumPy cheat sheet
    :class: admonition hint

    The following link is actually one of the first hits on any search engine
    but here it is for completeness. Please have some time spared to check out
    the differences between numpy and matlab syntax. It might even teach you
    a thing or two about matlab. 
    
    `Click here for \"Numpy for matlab users\" <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_


Now assuming that you have mastered the art of finding your way through
gazillion of blogs, filtering StackOverflow nerd anger, decrypting the
documentation of Numpy, let's start doing some familiar things in ``harold``.

Initializing harold
-------------------

Once harold is installed you can import it via one of the following ways ::

    import harold
    import harold as har
    from harold import *

While the 3rd option is frowned upon by programmers, for working engineers it
might be the most convenient option since the functions would not require a top
level namespace. In other words, the first two options would require the
functions to be used with a prefix such as ::

    harold.frequency_response(G)
    
or ::
    
    har.frequency_response(G)

.. note :: Almost all proper programming languages involve the concept of 
    **namespaces** (guess which one doesn't). This concept actually makes it
    possible to attach all the function names to a particular name family which
    is represented by the dot notation e.g., :: 

        mypackage.myfunction.myattribute = 45

    One obvious reason for this is that separate name families avoid name clashes
    which is a nightmare in matlab if you have two folders on the path and both have
    the variants of a function with the same name. You can never be sure which 
    one matlab will read if you decide to call this function from somewhere else. 

    Hence you need to assess the risk whether there will be name clashes if you use
    the 3rd option before you import all the names to the main namespace. This will
    scan the harold library and import every object whose name doesn't start with 
    ``_`` or ``__``. For interactive notebooks, this is pretty convenient if you
    are not importing libraries that have similar function names (in turn, you can
    never be sure).

Here is the boilerplate code to start with::

    import numpy as np
    import scipy as sp
    from harold import *

You can of course extend this to your liking with other packages. Now, let's
invesatigate how we can build dynamic models and do some control stuff.

System Representations
======================

For creating dynamic/static models, ``harold`` currently offers two options: 
A ``State()`` and a ``Transfer()`` object for representing systems with 
state models and transfer matrices. 

Creating models
----------------
 
``State()`` models
^^^^^^^^^^^^^^^^^^

The initialization of these objects are pretty straightforward. Note that you 
can skip ``D`` term and it will be assumed to be zero::
    
    G = State([[1,2],[3,4]],[[1],[0]],[1,2])
    G = State([[1,2],[3,4]],[[1],[0]],[1,2], dt = 0.1)

In the second example, we create a discrete-time model but to avoid the clash with
a ``D`` term we add the ``dt`` keyword explicitly. If there was also a nonzero 
feedthrough element then we can also skip that too::

    G = State([[1,2],[3,4]],[[1],[0]],[1,2],1,0.1)

As obvious to everyone who used this syntax even in matlab's convenient bracket
semicolon notation, creating four individual matrices everytime just to pass
to the function becomes increasingly annoying. Instead a matrix slicer is 
available in ``harold`` ::

    M = np.array([[1,2,1],[3,4,0],[1,2,0]])
    G = State(*matrix_slice(M,corner shape=(2,2), corner='nw'))

As shown above, the model creation is a straightforward enumeration of 
the involved ``A,B,C,D`` matrices in the arguments. For discrete time
models, you simply add a fifth argument, if you have not omitted any
or explicitly mentioning ``dt=<sampling period>`` as an argument. 
    
To create a static models, just provide an array ::

    G = State(1)
    G = State(np.ones((5,3)), dt=0.5)
    
Here again, ``dt=<sampling period>`` should be provided explicitly as it would
be confused about the second argument being the ``B`` element. ::

    G = State(1,0.001)    # Will lead to error
    G = State(1,dt=0.001) # Will work

If the model is discretized we can also check ::

    G.SamplingPeriod  # returns the sampling period
    G.SamplingSet     # returns 'Z' for discrete-time, 'R' otherwise
    G.DiscretizedWith # returns the discretization method if applicable

	
These make sure that the discretization remembers how it got there in 
the first place if harold is used. Or if the model is already given 
as a discrete time model, the method can be set such that ``undiscretize``
can use the correct method. 

``Transfer()`` models
^^^^^^^^^^^^^^^^^^^^^

    H = Transfer([1,2,3],[7,5,3])

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

After the initialization of the models, we can see the model properties ::

    G.NumberOfOutputs # returns the number of rows of numerator
    G.NumberOfOutputs # returns the number of cols of numerator
    G.shape           # returns a tuple of (# of outs,# of ins)
    
    G.polynomials     # returns the polynomials ala tfdata
    G.poles           # returns the poles 
    G.zeros           # returns the zeros
    G.num,G.den       # returns the individual polynomials
    
If the model is discretized we can also check ::

    G.SamplingPeriod  # returns the sampling period
    G.SamplingSet     # returns 'Z' for discrete-time, 'R' otherwise
    G.DiscretizedWith # returns the discretization method if applicable
    


.. autoclass:: Transfer
    :members:
	

	
	
Model Arithmetic
-------------------

Both ``Transfer`` and ``State`` instances support basic model arithmetic. 
You can add/multiply/subtract models that are compatible (division is 
a completely different story hence omitted). Again, `harold` tries its
best to explain what went wrong. Let's take the same discrete time 
SISO system and set another random MIMO model ``H`` with 3 states::

    G = State(-1,1,1,dt=0.01)
	H = State(*matrix_slice(np.random.rand(5,6),(3,3)))
	G*H

	TypeError: The sampling periods don't match so I cannot multiply these 
	systems. If you still want to multiply them asif they are compatible, 
	carry the data to a compatible system model and then multiply.
	
Even after making ``G`` a continous time system ::
	
	G.SamplingPeriod = 0.
	G*H
	
	IndexError: Multiplication of systems requires their shape to match but 
	the system shapes I got are (1, 1) vs. (2, 3)

Notice the system sizes are repeated in the error message hence we 
don't need to constantly check which part is the culprit for both
systems. 

For `Transfer` models, another useful property is recognition of 
common poles when doing simple addition/subtraction. For example, ::

	G = Transfer([1,1],[1,2])
	H = Transfer([1],[1,3,2])
	
	F = G+H
	F.polynomials #check num,den
	(array([[ 1.,  2.,  2.]]), array([[ 1.,  3.,  2.]]))
	
As you can see the cancellations are performed at the computations such that 
the model order does not increase artificially. 

.. note:: This is currently not the case for ``State`` instances that is
    to say the state matrices are directly augmented without any cancellation
    checks. This is probably going to change in the future. 


Context Discovery
------------------

For both ``State()`` and ``Transfer()`` argument parsing, ``harold`` can 
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
	In the MIMO context and proper entries, I've found
	scalar denominator entries hence flagging as a static gain.
	Out[7]: 
	([[array([[ 1.]]), array([[ 1.]]), array([[ 1.]])]],
	 [[array([[ 1.]]), array([[ 2.]]), array([[ 3.]])]],
	 (1, 3),
	 True)

As seen from the resulting arrays, the numerator is now 
three numpy float arrays containing the common entry. 
Both the numerator and denominator are converted to list
of lists. 

This method can also be used to verify whether a certain input
is a valid argument for creating model objects hence the name. 

Same class method is also available for the ``State()`` class. 
