Getting Started  
===============

The user is assumed to have some acquientance with Python, NumPy, and SciPy in
order to go through the tutorial. However, it is strongly recommended to get
versed in these impressive tools regardless. ::
    
It was actually 2014 that Python developers included a symbol for matrix
multiplication that is the ``@`` character (again almost everything else is
reserved and the regular ``*`` is meant for element-wise multiplication).

Another point that might annoy the users is the complex number syntax. In Python,
you are obliged to use the letter ``j`` for the complex unit but cannot use ``i``.

An almost exhaustive cheat sheet for recovering matlab users
-------------------------------------------------------------

The following link is actually one of the first hits on any search engine but here 
it is for completeness. Please have some time spared to check out the 
differences between numpy and matlab syntax. It might even teach you 
a thing or two about matlab. 

.. note :: `Click here for \"Numpy for matlab users\" <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_

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
    is represented by the dot notation e.g., 
    ``mypackage.myfunction.myattribute = 45`` etc. 

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
do some control stuff.