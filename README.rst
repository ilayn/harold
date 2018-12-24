|Travis-CI| |License| |Gitter| |Coveralls| |ReadTheDocs| |Downloads|

harold
======

A control systems package for Python>=3.6.

Introduction
============

This package is written with the ambition of providing a full-fledged control
systems software that serves a control engineer/student/researcher with complete
access to the source code with permissive rights (see ``LICENSE`` file). 
Moreover, via working with a proper high-level computer programming language
many proprietary software obstacles are avoided and users can incorporate this
package into their workflow in any way they see fit.

Quick Reference and Documentation
---------------------------------

The documentation is online at `ReadTheDocs`_. A brief tutorial about the basics
can be found under the notebooks folder to see ``harold`` in action.

Roadmap
-------

The items that are in the pipeline and what possibly lies ahead is enumerated
in our `roadmap <https://github.com/ilayn/harold/wiki/harold-roadmap>`_.

Useful Links
------------

- There is already an almost-matured control toolbox which is led by
  Richard Murray et al. (`click for the Github page`_) and it can perform
  already most of the essential tasks. Hence, if you want to have
  something that resembles the basics of matlab control toolbox, you should give
  it a try. However, it is somewhat limited to SISO tools and also relies on
  SLICOT library which can lead to installation hassle and/or licensing
  problems for nontrivial tasks.

- You can also use the tools available in SciPy ``signal`` module for basics
  of LTI system manipulations. SciPy is a powerful all-purpose scientific
  package. This makes it extremely useful however admittedly every discipline
  has a limited presence hence the limited functionality. If you are looking
  for a quick LTI system manipulation and don't want to install yet another
  package, then it might be the tool for you.

- Instead, if you are interested in robust control you probably would
  appreciate the `Skogestad-Python`_ project. They are replicating the
  code parts of the now-classic book completely in Python. Awesome!

Help Wanted!
------------

If you are missing out a feature, or found a bug, get in contact. Such
reports and PR submissions are more than welcome!

Contact
--------

If you have questions/comments feel free to shoot one to
``harold.of.python@gmail.com`` or join the Gitter chatroom.

.. _click for the Github page: https://github.com/python-control/python-control
.. _ReadTheDocs: http://harold.readthedocs.org/en/latest/
.. _Skogestad-Python: https://github.com/alchemyst/Skogestad-Python

.. |License| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/ilayn/harold/blob/master/LICENSE
.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/ilayn/harold?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Travis-CI| image:: https://travis-ci.com/ilayn/harold.svg?branch=master
    :target: https://travis-ci.com/ilayn/harold
.. |Coveralls| image:: https://coveralls.io/repos/github/ilayn/harold/badge.svg?branch=master
    :target: https://coveralls.io/github/ilayn/harold?branch=master
.. |ReadTheDocs| image:: https://readthedocs.org/projects/harold/badge/?version=latest
    :target: http://harold.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Downloads| image:: http://pepy.tech/badge/harold
    :target: http://pepy.tech/count/harold
    :alt: Download Counts
