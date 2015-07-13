[![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg?style=plastic)](https://github.com/ilayn/harold/blob/master/LICENSE) [![Join the chat at https://gitter.im/ilayn/harold](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ilayn/harold?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
# harold
A systems and controls toolbox for Python3. MIT licensed. See `LICENSE.md` file for details.


## Features

The main feature of `harold` is that it is written in pure Python3. Other than Numpy/Scipy, there are no additional libraries outside Python numerically. Thus, everything is under the hood and accessible. This, combined with the typical readable Python syntax, gives the full transparency to the control engineers and mathematicians how a certain functionality is implemented. Moreover, that makes it much easier to spot mistakes/quirks/outdated algorithms. 

Currently, in matlab and other places, many  fundamental algorithms of model-based control algorithms come from a Fortran library called [SLICOT](http://slicot.org/) with an opaque license. As needed to wrap around this library, I tried but couldn't install [Slycot](https://github.com/jgoppert/Slycot) (actually drove me nuts) on Windows and hence I've started to code everything in MIMO context with a possibly unrealistic up-to-production mindset.

`harold` also has a few more features worth mentioning:

  - The function names are verbose and (hopefully) understandable. 
  - Programmatically, only two classes exist `Transfer` and `State` and the rest is number crunching. 
  - Interactive plots suitable for Ipython (if I can fix a nasty last problem) hence way better-looking and quite less-painful Bode, Nyquist plots, and of course, in case people are still being tortured by them, root loci. 
  - Numerical polynomial operations (as opposed to symbolic) such as Least Common Multiples, Greatest Common Divisors etc. which is tested via pathological examples found in academic papers and to some extent practically on strange transfer matrices. 
  - Matrix-pencil based subroutines, Hessenberg forms and transmission zero computations (this one I've tested as much as I can). Agrees with matlab, if not better resolution, and typically much faster for reasons I don't know yet.
  - Better control of internal tolerances that affects minimal realizations, rank tolerances and so on. 
  - When a model is discretized, it remembers the method used (or set later) and also continous/discrete, discrete/discrete type of interconnections recognize the slowest sampling time system and resamples remaining ones. 
  - Helper functions that makes it easy to slice, concatenate, manipulate system data thanks to Python/Numpy flexibility. 

## Useful Links


 1- There is already an almost-matured control toolbox which is led by Richard Murray et al. ([click for the Github page](https://github.com/python-control/python-control) ) and it can perform already most of the essential tasks. Hence, if you want to have something that resembles the basics of matlab control toolbox give it a try. 

`python-control` emulates the matlab syntax and usage (for a good reason considering the matlab users) which is something I really wish to leave behind completely. Over the years I truly hated that syntax but that's just me. Just to name a few, argument hopping thanks to `nargin,nargout` stuff, inconsistent (sometimes just stupid) error handling, weird amalgam of structs and cells usage... 


  2- By the way, if you are interested in robust control you would probably appreciate the  [Skogestad-Python](https://github.com/alchemyst/Skogestad-Python) project. 


### Contact

If you have a question/comment feel free to shoot one to `harold.of.python@gmail.com`
