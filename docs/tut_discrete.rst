
Discretization Methods
=======================


In harold, a discrete time model can keep the discretization method 
in mind such that when the occasion arises to convert back to a 
continuous model the proper method is chosen. For example suppose
we discretized a state model via Zero-order-hold::

    # Take a polynomial and make the corresponding companion matrix 
    G = State(haroldcompanion([1,2,3,4]),eyecolumn(3,2),eyecolumn(3,0).T)
    F = discretize(G,0.01,method='zoh') # default method is 'tustin'
    
Now if we actually check the properties of ``F`` we can see what it 
actually keeps::

    F.DiscretizedWith # returns 'zoh'
    F.SamplingPeriod  # returns 0.01
    F.SamplingSet     # returns 'Z'
    
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
======================= ========================

Hence, if a model with ``DiscretizedWith`` property set to 
something else then ``None``, then the continuous time conversion
actually takes it into account, and uses that method on the way
back. 



.. todo:: Explain these methods and of course ``undiscretize()``
    stuff more!!