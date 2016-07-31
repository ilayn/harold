Model Properties
=================

.. todo:: Populate these with more text/examples

.. py:currentmodule:: harold    
.. autofunction:: minimal_realization
.. autofunction:: transmission_zeros
.. autofunction:: staircase
.. autofunction:: system_norm


Auxilliary Functions
====================

These are either functions that were necessary but not available
or a certain functionality was missing in them in their native form. 
Hence the prefix ``harold``. Over time, as ``scipy`` keeps adding 
functionalities (hopefully), they would become obsolete. 

.. autofunction:: haroldsvd
.. autofunction:: haroldker
.. autofunction:: matrix_slice
.. autofunction:: e_i
.. autofunction:: haroldtrimleftzeros
.. autofunction:: haroldcompanion


The following function is particularly useful if there is a 
significant amount of copy/pasting to matlab. Hence this small
function when used inside a ``print`` prints a numpy array 
in matlab syntax. 

.. autofunction:: matrix_printer

