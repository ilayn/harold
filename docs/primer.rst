A primer to harold
==================

.. todo:: Finish this as soon as possible

A humble advice to Python beginners
-----------------------------------

Since Python is a programming language and not a front-end software, the 
code we write usually needs to be executed somehow. Without going into 
the details, Python code needs to be interpreted (similar to matlab) 
as opposed to compiling C code. 

The native way of doing this is simply typing ``python`` on the command 
line and making it a Python interpreter. However, this is hardly ever 
useful for any practical purposes, let alone resuming or reproducing 
previous work. Having said that, you don't need a scary Visual blabla 
suite that looks like an airplane cockpit either. There are many options 
to make your life easier.


  1. The first option is simply working with an editor, e.g., `Spyder`_ 
     (I had a very positive experience with it), Eclipse, PyCharm, vim, 
     emacs so on. 
  2. Using the recent and very very powerful `Jupyter`_ (previously known 
    as IPython) which converts your web browser into a mathematica like 
    environment with explicit cells but you can embed even Youtube videos. 
    Moreover, it also works on the command window too which is not limited 
    to Python, but as the name implies Julia, Python, R and so on. 

I would strongly recommend Jupyter notebook option. It also makes sharing 
your work with others extremely easy. Please follow the link to Jupyter 
and install accordingly to your liking. 

  .. _Spyder : https://pythonhosted.org/spyder/
  .. _Jupyter : http://jupyter.org

Initializing harold
-------------------

Once you have managed to make Jupyter work you will have to import harold 
as a library. And when you do you have to access the function names properly 
depending on how you imported harold. 


This point is a pretty confusing and a source for heated arguments but I'll 
just cut to the chase. Almost all proper programming languages involve the 
concept of **namespaces** (you guess correctly, matlab doesn't have this). 
Sticking to the part that is relevant for us, this concept actually makes 
it possible to attach all the function names to a particular name family 
which is represented by the dot notation e.g., 
``mypackage.myfunction.myattribute = 45`` etc. 

One obvious reason for this is that separate name families avoid name 
clashes which is a nightmare in matlab if you have two folders on the 
path and both have the variants of the same function. You can never be 
sure which one matlab is going to read if you decide to call this 
function from somewhere else. 

Long story short when you import harold you can simply write on top of 
your notebook ::

    import harold

then it is possible to access the functions with the ``harold.`` prefix 
such as, say, for frequency response calculations::

    harold.frequency_response(G)

Alternatively, you can use an abbreviation for the package name ::

    import harold as har

and then you can access the functions with ``har.`` prefix. Lastly, 
there is another way which is, as is for almost everything involving 
professional programmers, another battlefield. You can basically decide 
to skip the namespace and import all functions with their original name 
to the parent namespace::

    from harold import *

This will scan the harold library and import every object whose name 
doesn't start with ``_`` or ``__``. For interactive notebooks, this 
is pretty convenient if you are not importing libraries that have 
similar function names (in turn, you can never be sure). 

Conclusion, if you don't have any worries about name clashes use the 
last syntax. The typical first cell of the notebook is the importing 
declarations. Here is the boilerplate code to start with::

    import numpy as np
    import scipy as sp
    from harold import *

You can of course extend this to your liking with your own packages. 
Finally, let's do some control stuff
