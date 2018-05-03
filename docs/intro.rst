Introduction
============

This is the documentation for the harold control systems toolbox. ``harold``
is a package for Python3 designed to be completely open-source in order to
serve the mantra *reproducible research*.

The main goal of harold is to provide accessible computation algorithms for
common control engineering tasks. For the academic use, the functions are 
documented as much as possible with proper citations and inline comments to 
demonstrate the strength and weaknesses(if any) for future improvements or 
debugging convenience. More importantly, if an algorithm is flawed, the user can
directly modify since everything under the hood is visible to the user.

Prerequisites
-------------

harold works with Python >= 3.6.  It uses the packages ``numpy``, ``scipy``,
``tabulate``, and ``matplotlib``. 

Installation
------------

Installing harold is a straightforward package installation: you can install 
the most recent harold version using `pip`_::

    >>> pip install harold

.. _pip: http://pypi.python.org/pypi/pip

Development
-----------

The official development is located on 
`harold GitHub repository <https://github.com/ilayn/harold>`_. 
