Introduction
=============

This is the documentation for the ``harold`` control systems toolbox. ``harold``
is a package for Python3 designed to be completely open-source in order to
serve the mantra *reproducible research*.

The main goal of ``harold`` is to provide accessible computation algorithms for
common control engineering tasks. For the academic use, the functions are 
documented as much as possible with proper citations and inline comments to 
demonstrate the strength and weaknesses(if any) for future improvements or 
debugging convenience. More importantly, if an algorithm is flawed or lacking
performance, the user can directly modify since everything under the hood is
visible to the user.

Prerequisites
-------------

``harold`` works with Python with version ``>= 3.6``.  It depends on the
packages ``numpy``, ``scipy>=1.0.0``, ``tabulate``, and ``matplotlib``.

Though ``harold`` can be made to work on Python 3.5 too, by removing, mostly,
f-strings and related details, Python 3.6 is recommended to work with anyways.

Not only it would bring a lot of extras but also keeyword handling behavior is
better and especially Windows systems would benefit from Unicode handling.

Installation
------------

Installing harold is a straightforward package installation: you can install 
the most recent ``harold`` version using `pip`_::

    >>> pip install harold

.. _pip: http://pypi.python.org/pypi/pip

If you have cloned the project from the GitHub repository for the latest
development version then you can also install locally via ::

    >>> python setup.py install

which will install the ``dev`` version. To generate the documentation locally,
you will need Sphinx ``>=1.7.4`` and ``cloud-sptheme>=1.9.4``. Then change the
directory to ``../docs/`` and then run ::

    >>> make html
    
on a terminal/command prompt.

Development
-----------

The official development lives on 
`harold GitHub repository <https://github.com/ilayn/harold>`_. Please open an
issue or submit your pull requests there. Feedback is always welcome. You can
also leave a message on the `Gitter chatroom <https://gitter.im/ilayn/harold>`_

Please let the developers know about the problems you have encountered or
features that you feel missing. That would help the roadmap greatly.

Some tips for Python beginners
------------------------------------

In order to benefit from ``harold`` some acquaintance with Python, NumPy, and
SciPy is necessary. However, it is strongly recommended to get versed in these
impressive tools in any case.
    
.. note :: It was actually 2014 that Python language developers reserved a symbol
    for matrix multiplication that is the ``@`` character (again almost everything
    else is reserved and the regular ``*`` is meant for element-wise
    multiplication in NumPy).

    Another point that might annoy the new users is the complex number syntax.
    In Python, you are obliged to use the letter ``j`` for the complex unit but
    cannot use ``i``.

.. admonition :: Tip : An almost exhaustive NumPy cheat sheet
    :class: admonition hint

    The following link is actually one of the first hits on any search engine
    but here it is for completeness. Please have some time spared to check out
    the differences between numpy and matlab syntax. It might even teach you
    a thing or two about matlab. 
    
    `Click here for \"Numpy for matlab users\" <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_

Once harold is installed you can import it via one of the following ways ::

    >>> import harold
    >>> import harold as har
    >>> from harold import *

While the 3rd option is frowned upon by programmers, for working engineers it
might be the most convenient option since the functions would not require a top
level namespace. In other words, the first two options would require the
functions to be used with a prefix such as ::

    >>> harold.frequency_response(G)
    
or ::
    
    >>> har.frequency_response(G)

.. note :: Almost all proper programming languages involve the concept of 
    **namespaces** (guess which one doesn't). This concept actually makes it
    possible to attach all the function names to a particular name family which
    is represented by the dot notation e.g., :: 

        >>> mypackage.myfunction.myattribute = 45

    One obvious reason for this is that separate name families avoid name clashes
    which is a nightmare in matlab if you have two folders on the path and both have
    the variants of a function with the same name. You can never be sure which 
    one matlab will read if you decide to call this function from somewhere else. 

    Hence you need to assess the risk whether there will be name clashes if you use
    the 3rd option before you import all the names to the main namespace. This will
    scan the harold library and import every object whose name doesn't start with 
    ``_`` or ``__``. For interactive notebooks, this is pretty convenient if you
    are not importing libraries that have similar function names (in turn, you can
    never be sure).
