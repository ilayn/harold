Controllability and Observability
==================================


Kalman-tests
-------------

Typical checks for controllability and observability are included in 
the functions listed below.

.. warning:: Note that these operations are numerically potentially
    ill-conditioned due to the increasing powers in the tests. Hence
    here included for completeness. However, for numerical purposes
    ``harold`` does not use these (see next section). 

.. py:currentmodule:: harold    
.. autofunction:: controllability_matrix
.. autofunction:: observability_matrix
.. autofunction:: kalman_decomposition
.. autofunction:: is_kalman_controllable
.. autofunction:: is_kalman_observable


Cancellation Distance
---------------------

Instead of checking a numerically ill-conditioned `rank` property, 
we use the metric of how small a perturbation is needed for the 
pencil :math:`\begin{bmatrix} \lambda I - A &B \end{bmatrix}`. If 
this quantity is less than a threshold we assume cancellation. 

Minimal realization for state representations uses this metric. The
method is implemented as described in [#f1]_.

.. autofunction:: cancellation_distance


.. [#f1] D. Boley, `Estimating the Sensitivity of the Algebraic Structure 
    of Pencils with Simple Eigenvalue Estimates`, SIMAX 11-4 (1990),
    `DOI <http://dx.doi.org/10.1137/0611046>`__