|Travis-CI| |License| |Gitter| |Coveralls| |ReadTheDocs| |Downloads|

harold
======

A control systems package for Python3 (>=v3.6 required).

Introduction
============

This package is written with the ambition of becoming a daily work-horse of
a control engineer/student/researcher with complete access to the source code
with full rights (see ``LICENSE`` file) while still working inside a
full-fledged programming language. This allows for working in any medium that
supports Python and its scientific packages NumPy and SciPy.

``harold`` fully supports the mantra of reproducible research and thus aims
to provide the means of accessible and transparent computational development
tools.

Usage
=====

A brief tutorial about the basics can be found under the notebooks folder.

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

2- You can also use the tools available in SciPy ``signal`` module for basics
of LTI system manipulations.

3- Instead, if you are interested in robust control you probably would
appreciate the `Skogestad-Python`_ project.

Help Wanted!
============

``harold`` is built on rainy days and boring evenings unless you hire me
directly for a specific tool. If you are missing out a feature, don't be shy
and contact me. User-feedback has higher priority over the general development.

Bug reports and PR submissions are more than welcome!

Contact
--------

If you have questions/comments feel free to shoot one to
``harold.of.python@gmail.com`` or join the Gitter chatroom.

.. _click for the Github page: https://github.com/python-control/python-control
.. _Sphinx documentation: http://harold.readthedocs.org/en/latest/
.. _Skogestad-Python: https://github.com/alchemyst/Skogestad-Python

.. |License| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/ilayn/harold/blob/master/LICENSE
.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/ilayn/harold?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Travis-CI| image:: https://travis-ci.org/ilayn/harold.svg?branch=master
    :target: https://travis-ci.org/ilayn/harold
.. |Coveralls| image:: https://coveralls.io/repos/github/ilayn/harold/badge.svg?branch=master
    :target: https://coveralls.io/github/ilayn/harold?branch=master
.. |ReadTheDocs| image:: https://readthedocs.org/projects/harold/badge/?version=latest
    :target: http://harold.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Downloads| image:: http://pepy.tech/badge/harold
    :target: http://pepy.tech/count/harold
    :alt: Download Counts
