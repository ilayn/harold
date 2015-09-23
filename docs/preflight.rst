Getting Started
===============

Using Python and other open-source tools are great only if one
gets acquainted with the ecosystem sufficiently. Otherwise, usually
an online search frenzy and then the frustration is around the corner. 

Hence, before we move to the harold documentation, maybe a general 
information about the related concepts can soften the fall. 

Python and its scientific stack
-------------------------------

For the newcomer, a slightly distorted story of the tools required to work 
with harold might save some headaches. To emphasize, **the story is roughly
correct but precisely informative** !

Since the late 70s, there has been an enormous effort went into the numerical
computing. For reasons that are really not relevant now, Fortran (yes that old
weirdo language) is still considered to be one of the fastest if not the fastest
runtime performance platform. On top of this performance, people with great 
numerical algebra expertise had been optimizing these implementations since then.

The result is now known as BLAS, LAPACK and other key libraries optimized to the
bone with almost literally manipulating the memory addresses manually. And what
happens is that many commercial software suites actually somehow find a way to 
utilize this performance even though they are not coding in Fortran directly. 
You have to appreciate how generous the authors were and why I'm humbly replicating
their style with the MIT license. 

Hence, without knowingly, you have been mostly using compiled Fortran code with
various frontends including matlab and other software. However, this high performance
doesn't come for free. The price to pay is to be extremely verse with low-level 
operations to benefit from these tools.  Hence, every language/software somehow designs 
a front end that communicates with both the user, understands the context, say
a matrix turns out to be triangular, then prepares the data in a very strict 
format and sends to these libraries. In turn, picks up the result, shuffles the
output format and converts it to something that the user can utilize further.

In Python, this front end is called the scientific stack, that is NumPy and SciPy. 
These involve C and Fortran bindings to low-level tools and wraps them with the 
typical easy-to-use Python syntax. 

That's why, if you are not the faint-hearted user and installing everything by 
yourself, you have to provide a compiler that is capable of providing the compiled
versions of these libraries. You might have noticed that some sources ship precompiled
flavors of NumPy, SciPy to ease the pain of building the libraries from scratch. 
While being useful, if the compiler that precompiled that binaries do not match
your system, the result is often a glorious crash. 

.. note:: The compiler problem is almost-universally a Windows problem since it doesn't 
come with a proper compiler and the unfortunate users (including me) who have 
no idea even what a compiler is, have to find a compiler that crashes depending on 
the weather conditions of the installation date. See `this question and its answer 
<http://stackoverflow.com/questions/2676763/>`_ to appreciate the situation. 

For this reason, some groups have come up with precompiled and matched version 
packages that are ready to be installed without bothering the users such as 
Anaconda, pythonxy etc. See the `installation page of Scientific stack 
<http://www.scipy.org/install.html>`_ and cross your fingers. 

Python and its strange syntax
-----------------------------

