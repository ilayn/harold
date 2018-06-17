.. _function-ref:

==================
Function reference
==================

.. automodule:: harold
    :no-members:
    :no-inherited-members:

.. currentmodule:: harold

System creation
===============
.. autosummary::
    :toctree: generated/

    State
    Transfer
    random_state_model

Frequency domain simulation
===========================

.. autosummary::
    :toctree: generated/

    frequency_response
    bode_plot
    nyquist_plot

Model Functions
===============

.. autosummary::
    :toctree: generated/

    system_norm

Kalman tests
------------

.. autosummary::
    :toctree: generated/

    controllability_matrix
    observability_matrix
    is_kalman_controllable
    is_kalman_observable
    
Time domain simulation
======================

.. autosummary::
    :toctree: generated/

    simulate_linear_system
    simulate_step_response
    simulate_impulse_response
    impulse_response_plot
    step_response_plot

Model simplification tools
==========================
.. autosummary::
    :toctree: generated/

    feedback
    minimal_realization
    hessenberg_realization
    staircase

Polynomial Operations
======================

.. autosummary::
    :toctree: generated/

    haroldgcd
    haroldlcm
    haroldpolyadd
    haroldpolymul
    haroldcompanion
    haroldker

Auxillary Functions
===================

.. autosummary::
    :toctree: generated/

    e_i
    concatenate_state_matrices
    haroldsvd
    