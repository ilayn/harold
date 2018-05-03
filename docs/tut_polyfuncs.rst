Polynomial Operations
======================

To be able to do usable transfer functions, some care
should be given to polynomial operations. The least common
multiples and greatest common divisors are of the utmost
importance. 

Thus, ``harold`` offers a system-theory inspired method 
to compute the LCM and GCD with a modified version of 
[#f1]_. I would appreciate ill-conditioned examples and/or
fails or much better methods. 

.. py:currentmodule:: harold    
.. autofunction:: haroldlcm
.. autofunction:: haroldgcd

.. [#f1] N. Karcanias, M. Mitrouli, `System theoretic based 
    characterisation and computation of the least common 
    multiple of a set of polynomials`, Linear Algebra and its Applications
    381, 2004, `DOI <http://doi:10.1016/j.laa.2003.11.009>`__

