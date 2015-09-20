Frequently Asked Questions
==========================

This page answers some of the often asked questions about harold.

Is it really free? 
------------------

Yes. Please search "MIT Expat" license or see a summary in `TL;DR Legal
<https://tldrlegal.com/license/mit-license>`_. 

How fast is it?
---------------

Currently in its unoptimized state, it is pretty close to matlab speed, due to 
being lighter and simpler. However, this will change when the structural 
correctness is established. Note that, control engineering computations rarely
require speed. And when they do, they have to be really, really fast, much faster
than what these software offer. Hence, I have left the optimization of the 
code until the Numba, Cython etc. debate settles. Plus, it is much simpler 
to do parallelized tasks in Python. So more speed-ups are around the corner 
waiting for me to read more documentation. 

The algorithmic speed is usually head-to-head with other software, since I also 
used the new published methods as much as I can understand them. 


Why is deciBell not the default unit?
-------------------------------------

Because it does not serve to the general audience. And as a unit, it is just a 
gain away from the logarithmic plot. So there is no added value. But for the sake 
of completeness, there are some keyword options available to select it. 


Why is there no Python 2 support?
---------------------------------

Python 2 is legacy code. Get over it. 


Why is it called harold?
------------------------

Originally, it was planned to be called as `pykant` which is a wordplay
between the Dutch words "spicy" and "Py(thon) side". But it also has a 
pretty strong slang side to it too in english. So that didn't fly.

Then, after accidentally reading the wildly off-target manuscript of 
David A. Mindell [#f1]_, I was saddened by witnessing yet another instance of 
the math snobbery that has poisoned the control engineering field in the 
last decades; I wanted something that resembles the good ol' days of automation. 

In this specific manuscript Mindell attempts to, in a quite skilful 
tongue-in-cheek fashion, belittle the invention of negative feedback amplifier 
by Black to a mere coincidence involving a layman that is S.H. Black. Because
Black was too generous about the scale of his achievements, he cannot be the actual
inventor because then the classical *über-humble genius* storyline that needs to 
follow would not follow. Mindell, later finds his heroes;  it should have been the 
smart PhD floor of Bell Labs that created all the smaller mountains (which is indeed 
true for other things). Quoting from his introduction, *"As it turns out, Black did 
not understand as much about feedback as he later recalled."*, Mindell later 
demonstrates clearly that himself too, does not understand feedback theory that much 
and fails to identify his anachronistic focus and hindsight bias. Consequently, 
he misinterprets these milestones and thus chooses the easiest way out by polishing 
the elitist trophies and dismisses a truly gigantic discovery based on its inventor's 
personality. 

Hence, the name `harold` to pay some tribute to the real people as opposed to 
the pantheon of stainless characters.


.. rubric:: Footnotes 

.. [#f1] D.A. Mindell, "Opening Black's Box: Rethinking Feedback's Myth of Origin", Technology and Culture, Volume 41, Number 3, pp. 405-434, July 2000. Accessed: 17 February 2015