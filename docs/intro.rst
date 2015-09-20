Introduction
============

This is the documentation for the harold control systems toolbox.
harold is a library for Python3 that is designed to be completely open-source
to serve the mantra *reproducible research*.

The main goal of harold is to provide accessible computation algorithms for
common control engineering tasks. For the academic use, the functions are 
documented as much as possible with proper citations and inline comments to 
demonstrate the strength and weaknesses(if any) for future improvements or 
debugging convenience. More importantly, if you think the developer goofed 
with a certain detailed, you can directly modify and fix your code as 
everything under the hood is visible to the user. 

For the industrial use, the most important aspect is its license. With proper
citation, it is possible to use in a commercial context. Secondary advantages 
are ease of context creation, for example, if you have an identification 
experiment, it is possible to identify from which point in the loop the 
measurements are taken, and harold will try to fill the gaps for you (this 
is currently being tested on a real experiment). 

Prerequisites
-------------

harold works with Python >= 3.4.  It uses the packages NumPy, SciPy, `tabulate`
and also relies on `itertools`, `collections`, and `copy` modules. It uses 
Bokeh as the main graphical framework for plots and so on. However, currently
for the terminal use and simpler manipulation a `matplotlib` frontend is 
experimented with. 


Installation
------------

Installing harold is a straightforward package installation: you can install 
the most recent harold version using `easy_install`_ or `pip`_::

    easy_install harold
    pip install harold

.. _easy_install: http://peak.telecommunity.com/DevCenter/EasyInstall
.. _pip: http://pypi.python.org/pypi/pip

Development
-----------

The official development is located on 
`harold GitHub repository <https://github.com/ilayn/harold>`_. 
