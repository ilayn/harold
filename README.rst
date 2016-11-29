|Travis-CI| |GitHub license| |Join the chat at https://gitter.im/ilayn/harold|

harold
======

A (yet another) systems and controls module for Python3. See ``LICENSE.md``
file for details (spoiler alert it's MIT Licensed).

Disclaimer
----------

This is a work in progress. It's just almost about to have the first release,
version 0.1. So, please adjust the expectations accordingly.

Introduction
============

harold library is written with the ambition of becoming a daily work-horse of
a control engineer/student/researcher with complete access to the source
code and also with full rights in terms of open source.

The common licensing problems where there are more users than the available
number of licenses or, in a commercial setting, exporting commercial code
with the product, or building a product using commercial software and so on
are naturally avoided.

In short, the purpose of this library is to disclose the development of
control algorithms. harold fully supports the mantra of reproducible research
and thus aims to provide the means of accessible and transparent computational
development tools.

harold also follows the Pythonic way of naming things and hence deviates from
the matlab way of working. This might add some learning effort on the user
but hopefully the function names are self-explanatory enough to ease the pain.

harold currently comes with two representation models: State and Transfer.
All is based around such object classes and the rest is just (lots of)
functions.

To demonstrate some examples, you create a state space or transfer function
representation via the following syntax::

    G = State([[0,1],[-2,-1]],[[0],[1]],[[1,0]]) # you can skip D if zero
    H = Transfer([1,2,1],[1,4,5,6])
    J = State(5)            # Gain models are supported
    K = Transfer(3, dt=0.1) # in discrete time too.

If we type `H` we get::

    Continous-Time Transfer function with:
     1 input(s) and 1 output(s)

      Poles(real)    Poles(imag)    Zeros(real)    Zeros(imag)
    -------------  -------------  -------------  -------------
             -0.5        1.32288             -1              0
             -0.5       -1.32288             -1              0
             -3          0

    End of Transfer object description


This might be better represented in the future versions. Or you can implement
your own infostring because you might be computing the damping of the
zeros/poles all the time. Important to note is the matrix syntax of NumPy
is a little awkward and we don't have much to change that now. However,
in turn, `harold` tries to be as forgiving as possible expecting the least
formal representation of arrays.

For example, for MIMO Transfer representations you can simply give a common
numerator or denominator and it will be completed internally to a matching
size, e.g.,::

    G = Transfer(1,[ [ [1,1],[1,2,1] ],[ [2,3],[1,3,3,1] ] ])

Notice how we had to increase the nesting of the bracket level to make it
contain arrays for a 2x2 system. This is, say in matlab, is done via curly
braces but for Python, brackets are natural for a list syntax.

We are at it, here is a Bode Plot of a 348-state Clamped Beam model
from the SLICOT collections::

    import scipy.io as sio # needed to read .mat files
    M = sio.loadmat('beam.mat')
    A = M['A'].todense()
    B = M['B']
    C = M['C']
    G = State(A,B,C)
    bode_plot(G)

.. image:: https://cloud.githubusercontent.com/assets/1303842/20697360/dfd9800c-b5f8-11e6-8f98-79d1964ec701.png

By the way, you can't convert this model to a Transfer model (including in
matlab etc.) because of the numerical error build-up. Thus, harold tries to
stay with whatever is given to it.

What can I do with `harold`?
----------------------------

Pretty much everything that is related to control systems
number-crunching as long as I can keep up with the implementation. The
testing phase is coming to an end, however, certain functionality is
(hopefully) in place:

  - State, Transfer representations ✓
  - Pole/zero computations ✓
  - Minimal realizations ✓
  - Controllabe/Observable Hessenberg Forms ✓
  - Pole/Zero cancellation distance computations ✓
  - Discretizations/Undiscretizations ✓
  - Frequency Response calculations ✓
  - LTI Algebra (add,mul,sub,feedback) ✓ (except feedback and series)
  - Polynomial Operations ✓
  - Riccati and Lyapunov equation solvers ✓ (Riccati solver is moved to SciPy)
  - H2- and HInfinity norms of systems ✓
  - Bode, Nyquist static plots ✓


The coded-but-not-tested parts are the essential synthesis techniques

  - Pole placement (uses the scipy versions)
  - LQR/LQG (the functionality is there but needs to be wrapped in a function)

Still waiting to be coded

  - Step, Impulse responses
  - Hinf, H2 controllers
  - Lead, Lag and notch filters with semi-recommendations about where to place them

and so on.

For example, you would like to have a row compressed, observable Hessenberg
form such that the rows of C matrix is compressed on the right end. You might
wonder why you would need such a representation. Welcome to the weird world
of control theory!::

    M = np.array([[-6.5,  0.5,  6.5, -6.5,  0. ,  1. ,  0. ],
                  [-0.5, -5.5, -5.5,  5.5,  2. ,  1. ,  2. ],
                  [-0.5,  0.5,  0.5, -6.5,  3. ,  4. ,  3. ],
                  [-0.5,  0.5, -5.5, -0.5,  3. ,  2. ,  3. ],
                  [ 1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    # Slice the matrix into 4 pieces with south west piece has (1,4) shape
    A , B , C , D = matrix_slice(M, (1, 4), corner='sw')
    a,b,c,k = staircase(A, B, C, form='o', invert=True)

    a
    array([[ 0.        , -6.        , -7.48528137,  0.        ],
           [-6.        ,  0.        ,  9.48528137,  0.        ],
           [ 0.        ,  0.        , -6.        ,  0.        ],
           [ 0.        ,  0.        ,  1.41421356, -6.        ]])

    b
    array([[ 3.        ,  2.        ,  3.        ],
           [ 3.        ,  4.        ,  3.        ],
           [ 1.41421356,  0.        ,  1.41421356],
           [ 1.41421356,  1.41421356,  1.41421356]])

    c
    array([[ 0.        ,  0.        ,  0.        ,  1.41421356]])

    k
    array([ 1.,  1.])

Here `k` is the size of the block that is identified as observable at each
step of the staircase. We can deduce that two of the modes are already
unobservable since the upper left 2x2 block does not interact with the lower
right two modes (A21 block is identically zero). Let's check the minimality
indeed::

    a,b,c = minimalrealization(A,B,C)

    a
    array([[-6.        ,  0.        ],
           [ 1.41421356, -6.        ]])

    b
    array([[ 1.41421356,  0.        ,  1.41421356],
           [ 1.41421356,  1.41421356,  1.41421356]])

    c
    array([[ 0.        ,  1.41421356]])

which gives a 2x2 system as we have expected.


Suppose you have some polynomials and would like to compute the LCM/GCD. Then
you can go about it via::

    a , b = haroldlcm([1,3,0,-4], [1,-4,-3,18], [1,-4,3], [1,-2,-8])

which returns::

    a
    (array([   1.,   -7.,    3.,   59.,  -68., -132.,  144.])

    b
    [array([  1., -10.,  33., -36.]),
     array([  1.,  -3.,  -6.,   8.]),
     array([  1.,  -3., -12.,  20.,  48.]),
     array([  1.,  -5.,   1.,  21., -18.])]

Here `a` is the least common multiple and `b` is the array of polynomials
that are needed to be multiplied by the original polynomials (in the order
of appearance) to obtain the LCM.

Another point-of-interest is the interactive plots that are promising. That
would hopefully minimize the right-click mania that follows almost every
plotting command in every commercial software for Bode, Nyquist, Sensitivity,
Coherence and others.


What about ...?
===============

Yes, yes, LMIs are coming. I have to learn cvxpy a bit faster. Other stuff
you need to let me know what the need is.

Documentation
=============

See the `Sphinx documentation`_ .

Useful Links
============

1- There is already an almost-matured control toolbox which is led by
Richard Murray et al. (`click for the Github page`_ ) and it can perform
already most of the essential tasks. Hence, if you want to have
something that resembles the basics of matlab control toolbox give it a
try. However, it is mostly limited to SISO tools and also relies on
SLICOT library which can lead to licensing problems for nontrivial tasks.

2- By the way, if you are interested in robust control you would
probably appreciate the `Skogestad-Python`_ project.

Help Wanted!
============

harold is built on rainy days and boring evenings. If you are desperately
missing out a feature, don't be shy and contact. User-feedback has higher
priority over the general development. Or shout out in the Gitter chatroom.

Or if you want to jump into development, PR submissions are more than welcome.

Contact
--------

If you have questions/comments feel free to shoot one to
``harold.of.python@gmail.com``

.. _click for the Github page: https://github.com/python-control/python-control
.. _Sphinx documentation: http://harold.readthedocs.org/en/latest/
.. _Skogestad-Python: https://github.com/alchemyst/Skogestad-Python

.. |GitHub license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/ilayn/harold/blob/master/LICENSE
.. |Join the chat at https://gitter.im/ilayn/harold| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/ilayn/harold?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Travis-CI| image:: https://travis-ci.org/ilayn/harold.svg?branch=master
    :target: https://travis-ci.org/ilayn/harold
