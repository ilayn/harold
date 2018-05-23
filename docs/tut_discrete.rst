
Discretization Methods
=======================

Usage
-----

In harold, a discrete time model can keep the discretization method 
in mind such that when the occasion arises to convert back to a 
continuous model the proper method is chosen. For example suppose
we discretized a state model via Zero-order-hold::

    >>> # Take an array and form the corresponding companion matrix 
    >>> G = State(haroldcompanion([1,2,3,4]),eyecolumn(3,2),eyecolumn(3,0).T)
    >>> F = discretize(G,0.01,method='zoh') # default method is 'tustin'
    
Now if we actually check the properties of ``F`` we can see what it 
actually keeps::

    >>> F.DiscretizedWith # returns 'zoh'
    >>> F.SamplingPeriod  # returns 0.01
    >>> F.SamplingSet     # returns 'Z'
    
Currently, the known discretization methods are given as 

======================= ========================
Method                  Aliases
----------------------- ------------------------
``bilinear``            ``tustin``

                        ``trapezoidal``
----------------------- ------------------------
``forward difference``  ``forward rectangular``

                        ``forward euler``

                        ``>>``
----------------------- ------------------------
``backward difference``  ``backward rectangular``

                         ``backward euler``

                         ``<<``
----------------------- ------------------------
``lft``
----------------------- ------------------------
``zoh``
----------------------- ------------------------
``foh``
======================= ========================

Hence, if a model with ``DiscretizedWith`` property set to 
something else then ``None``, then the continuous time conversion
actually takes it into account, and uses that method on the way
back. As an example::

    >>> H = Transfer([1.,0,3],[1,3,5])
    >>> G = discretize(H,method='zoh',dt=0.01)
    >>> F1 = undiscretize(G)
    >>> F2 = undiscretize(G,use_method='tustin')
    >>> F2.polynomials
    (array([[ 1.01499975,  0.01020009,  3.00002499]]),
     array([[ 1.        ,  3.00015   ,  5.00004165]]))
    >>> F1.polynomials
    (array([[  1.00000000e+00,   5.39124301e-13,   3.00000000e+00]]),
     array([[ 1.,  3.,  5.]]))

We can clearly see the artifacts of the mismatched conversion methods 
even in this simple example.

For Tustin method, prewarp frequency correction is also implemented which can
also be used during the undiscretization.

Functions
---------

.. py:currentmodule:: harold    
.. autofunction:: discretize
.. autofunction:: undiscretize 