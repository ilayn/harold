Matrix Equations
================

.. todo:: Finish this section and add Riccati, hinfsyn etc.

The Lyapunov equations and Algebraic Riccati equations play 
central role in model-based control design algorithms. Recently,
LMI based methods, IQC synthesis in particular demands much more 
precise and more importantly better-conditioned solutions. Hence, 
there is a need to tweak the details of such solvers however the 
typical software uses closed-source code and the internals are not 
possible to modify/observe. Therefore, ``harold`` provides some 
subset of the common solvers, even when there is a version in scipy
as a native code. 



Lyapunov Equation 
-------------------

The Lyapunov equations are defined as follows,

.. math::

    \begin{align}
    X A + A^T X + Y &= 0 \tag{1} \\\\
    A^T X A - X + Y &= 0 \tag{1'} \\\\
    E^T X A + A^T X E + Y &= 0 \tag{2} \\\\
    A^T X A - E^T X E + Y &= 0   \tag{2'}
    \end{align}


.. py:currentmodule:: harold    
.. autofunction:: lyapunov_eq_solver


