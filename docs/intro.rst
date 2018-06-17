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


In order to benefit from ``harold`` some acquaintance with NumPy and SciPy is
necessary. However, it is strongly recommended to get versed in these
impressive tools in any case.
    
.. admonition :: Tip : An almost exhaustive NumPy cheat sheet
    :class: admonition hint

    The following link is actually one of the first hits on any search engine
    but here it is for completeness. Please have some time spared to check out
    the differences between numpy and matlab syntax. It might even teach you
    a thing or two about matlab. 
    
    `Click here for \"Numpy for matlab users\" <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_

Development
-----------

The official development lives on 
`harold GitHub repository <https://github.com/ilayn/harold>`_. Please open an
issue or submit your pull requests there. Feedback is always welcome. You can
also leave a message on the `Gitter chatroom <https://gitter.im/ilayn/harold>`_

Please let the developers know about the problems you have encountered or
features that you feel missing. That would help the roadmap greatly.

