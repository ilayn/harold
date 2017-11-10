|Travis-CI| |License| |Join the chat at Gitter| |Coveralls| |ReadTheDocs|

harold
======

A control and systems enginnering package for Python3.

Introduction
============

This package is written with the ambition of becoming a daily work-horse of
a control engineer/student/researcher with complete access to the source code
with full rights (see ``LICENSE`` file) while still working inside a
full-fledged programming language. This means working in any medium that
supports Python and its scientific packages NumPy and SciPy.

``harold`` fully supports the mantra of reproducible research and thus aims
to provide the means of accessible and transparent computational development
tools.

Usage
=====

``harold`` currently comes with two representation models: State and Transfer.
All is based around such object classes and the rest is just (lots of)
functions. You create a state or transfer representation via the following
syntax::

    >>> G = State([[0,1],[-2,-1]],[[0],[1]],[[1,0]]) # you can skip D if zero!
    >>> H = Transfer([1,2,1],[1,4,5,6])
    >>> J = State(5)            # Static Gain models are supported
    >>> K = Transfer([[1,2], [3,4]], dt=0.1) # MIMO and/or in discrete time too.

If we type ``H`` we get::

    Continous-Time Transfer function with:
     1 input(s) and 1 output(s)

      Poles(real)    Poles(imag)    Zeros(real)    Zeros(imag)
    -------------  -------------  -------------  -------------
             -0.5        1.32288             -1              0
             -0.5       -1.32288             -1              0
             -3          0

    End of Transfer object description

Thus, instead of seeing some strange ASCII-art of what you already have typed a
second ago, you can arrange whatever you wish to see from a system
representation. You can implement your own infostring because you might
be computing the damping of the zeros/poles all the time and that might
be the only detail you wist to see from the models.

Since NumPy syntax is a bit more laborious for entering arrays, in turn,
``harold`` tries to be flexible about the input arguments. Here is a Bode Plot
of a 348-state Clamped Beam model from the SLICOT collections::

    >>> import scipy.io as sio # needed to read .mat files
    >>> M = sio.loadmat('beam.mat')
    >>> A = M['A'].todense()
    >>> B = M['B']
    >>> C = M['C']
    >>> G = State(A,B,C)
    >>> bode_plot(G, use_db=True) # Default, 10^x is used in the mag plots

.. image:: https://user-images.githubusercontent.com/1303842/32674179-7bba5f54-c652-11e7-91bf-3d113188a8fb.PNG

This model is numerically quite ill-conditioned to be converted to a Transfer
model because of the numerical error build-up. Thus, harold tries to
stay with whatever representation model given to it if an algorithm exists for
that particular representation.

What can I do with ``harold`` ?
-------------------------------

In theory, pretty much everything that is related to control systems
number-crunching (as long as I can keep up with the implementation):

  - State, Transfer representations ✓
  - Pole/zero computations ✓
  - Minimal realizations ✓
  - Controllable/Observable Hessenberg Forms ✓
  - Pole/Zero cancellation distance computations ✓
  - Discretizations/Undiscretizations ✓
  - Frequency Response calculations ✓
  - LTI Algebra (add,mul,sub,feedback) ✓
  - Polynomial Operations ✓
  - Riccati and Lyapunov equation solvers ✓ (Riccati solver is moved to SciPy)
  - H₂- and H∞-norms of systems ✓
  - Bode, Nyquist static plots ✓
  - Pole placement (already available in scipy) ✓
  - LQ design (LQR/LQRY/DLQR/DLQRY) ✓

Still on the pipeline:

  - Step, Impulse responses
  - Riccati-based H∞, H₂ controllers
  - Lead, Lag and notch filters with semi-recommendations about where to place them


For example, you would like to have a row compressed, observable Hessenberg
form such that the rows of C matrix is compressed on the right end. You might
wonder why you would need such a representation. Welcome to the weird world
of control theory!::

    >>> M = np.array([[-6.5,  0.5,  6.5, -6.5,  0. ,  1. ,  0. ],
                      [-0.5, -5.5, -5.5,  5.5,  2. ,  1. ,  2. ],
                      [-0.5,  0.5,  0.5, -6.5,  3. ,  4. ,  3. ],
                      [-0.5,  0.5, -5.5, -0.5,  3. ,  2. ,  3. ],
                      [ 1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> # Slice the matrix into 4 pieces with south west piece has (1,4) shape
    >>> A, B, C, D = matrix_slice(M, (1, 4), corner='sw')
    >>> a,b,c,k = staircase(A, B, C, form='o', invert=True)
    >>> a
    array([[ 0.        , -6.        , -7.48528137,  0.        ],
           [-6.        ,  0.        ,  9.48528137,  0.        ],
           [ 0.        ,  0.        , -6.        ,  0.        ],
           [ 0.        ,  0.        ,  1.41421356, -6.        ]])
    >>> b
    array([[ 3.        ,  2.        ,  3.        ],
           [ 3.        ,  4.        ,  3.        ],
           [ 1.41421356,  0.        ,  1.41421356],
           [ 1.41421356,  1.41421356,  1.41421356]])
    >>> c
    array([[ 0.        ,  0.        ,  0.        ,  1.41421356]])
    >>> k
    array([ 1.,  1.])

Here ``k`` is the size of the block that is identified as observable at each
step of the staircase. We can deduce that two of the modes are already
unobservable since the upper left 2x2 block does not interact with the lower
right two modes since A(2,1) block is identically zero. Let's check the
minimality indeed::

    >>> a,b,c = minimal_realization(A,B,C)

    >>> a
    array([[-6.        ,  0.        ],
           [ 1.41421356, -6.        ]])

    >>> b
    array([[ 1.41421356,  0.        ,  1.41421356],
           [ 1.41421356,  1.41421356,  1.41421356]])

    >>> c
    array([[ 0.        ,  1.41421356]])

which gives a 2x2 system as we have suspected (might have also been
uncontrollable).

In terms of auxillary functions which are used also internally for ``Transfer``
object manipulations too. Suppose you have bunch of polynomials and would like
to compute the LCM or GCD of them. Then you can go about it via::

    >>> a , b = haroldlcm([1,3,0,-4], [1,-4,-3,18], [1,-4,3], [1,-2,-8])

which returns::

    >>> a
    (array([   1.,   -7.,    3.,   59.,  -68., -132.,  144.])

    >>> b
    [array([  1., -10.,  33., -36.]),
     array([  1.,  -3.,  -6.,   8.]),
     array([  1.,  -3., -12.,  20.,  48.]),
     array([  1.,  -5.,   1.,  21., -18.])]

Here ``a`` is the least common multiple and ``b`` is the array of polynomials
that are needed to be multiplied by the original polynomials (in the order
of appearance) to obtain the LCM.

Another point-of-interest is the interactive plots that are promising. That
would hopefully minimize the right-click mania that follows almost every
plotting command in every commercial software for Bode, Nyquist, Sensitivity,
Coherence and others.

What about ...?
===============

Yes, yes, LMIs are coming. I have to learn ``cvxpy`` a bit faster. Other stuff
you need to let me know what the need is.

Documentation
=============

See the `Sphinx documentation`_ .

Useful Links
============

1- There is already an almost-matured control toolbox which is led by
Richard Murray et al. (`click for the Github page`_ ) and it can perform
already most of the essential tasks. Hence, if you want to have
something that resembles the basics of matlab control toolbox, you should give
it a try. However, it is somewhat limited to SISO tools and also relies on
SLICOT library which can lead to installation hassle and/or licensing
problems for nontrivial tasks.

2- Instead, you are interested in robust control you probably would appreciate
the `Skogestad-Python`_ project.

Help Wanted!
============

``harold`` is built on rainy days and boring evenings. If you are missing out
a feature, don't be shy and contact. User-feedback has higher priority over
the general development or shout out in the Gitter chatroom.

Or if you want to jump into development, PR submissions are more than welcome.

Contact
--------

If you have questions/comments feel free to shoot one to
``harold.of.python@gmail.com``

.. _click for the Github page: https://github.com/python-control/python-control
.. _Sphinx documentation: http://harold.readthedocs.org/en/latest/
.. _Skogestad-Python: https://github.com/alchemyst/Skogestad-Python

.. |License| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/ilayn/harold/blob/master/LICENSE
.. |Join the chat at Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/ilayn/harold?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Travis-CI| image:: https://travis-ci.org/ilayn/harold.svg?branch=master
    :target: https://travis-ci.org/ilayn/harold
.. |Coveralls| image:: https://coveralls.io/repos/github/ilayn/harold/badge.svg?branch=master
    :target: https://coveralls.io/github/ilayn/harold?branch=master
.. |ReadTheDocs| image:: https://readthedocs.org/projects/harold/badge/?version=latest
    :target: http://harold.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
