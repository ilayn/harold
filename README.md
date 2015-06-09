# harold [![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg?style=plastic)](https://github.com/ilayn/harold/blob/master/LICENSE) [![Join the chat at https://gitter.im/ilayn/harold](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ilayn/harold?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A systems and controls toolbox for Python3. MIT licensed. See `LICENSE.md` file for details.


## Features

The main feature of `harold` is that it is written in pure Python3. It certainly relies on Numpy and Scipy and these libraries utilize many well-known C/Fortran packages. However, other than Numpy/Scipy, there are no additional libraries outside Python. Thus, everything is under the hood and accessible. This, combined with the typical readable Python syntax, gives the full transparency to the control engineers and mathematicians how a certain functionality is implemented. Moreover, that makes it much easier to spot mistakes/quirks/outdated algorithms. Currently, many algorithms that are fundamental to the basic functionality of model-based control algorithms come from a Fortran library called [SLICOT](http://slicot.org/). I don't think anybody should be forced to read Fortran code unless there is a significant performance consideration. Instead the algorithms are described in excellent papers (unfortunately, unlike many theoretical papers) and can be simply read out from them. I couldn't install [Slycot](https://github.com/jgoppert/Slycot) (actually drove me nuts) on Windows and hence I've started to code everything in MIMO context with a possibly unrealistic up-to-production mindset.

I'm currently implementing a subset of the features of the control toolboxes that are found on matlab, mathematica, sage, octave, scilab and others. Though the main purpose of this attempt was to scratch my own itches and teach myself Python with a running project example, I was wondering why there is no standalone easy-access compilation of numerical code available in control engineering. Hence, I started to look for a roadblock that would make this hypothetical reason obvious and so far there is none other than the religious usage of matlab universally in everywhere. However, I would like to invite any confident control theoretician to write down how to find the transmission zeros of a state space system without cheating. So far I've asked more than a hundred people including proffessors and post-docs, and only two people managed to describe it and they did it only conceptually. 

`harold` also has a few more features worth mentioning:

  - The function names are verbose and (hopefully) understandable. Note that, matlab naming habits are prehistoric: back then file names had to comply with now-ridiculous naming limitations such as [8.3 filenames](http://en.wikipedia.org/wiki/8.3_filename). And because every function, roughly speaking, has to be an `.m` file, we get these strange names. But newly created functionalities don't suffer from this. That is the main reason for most of the cryptic command names combined with pretty normal names.  Among my favorites are `butter`, `ncfmr` and `ellip`. Amazing. Thanks to Python, in case you don't like the naming you can basically assign any name to any function such that 

          eig(A)                                           # <--- This line,
          my_favorite_eigenvalue_finding_function_2 = eig  #      after this assignment, is the same with 
          my_favorite_eigenvalue_finding_function_2(A)     # <--- this line with same functionality.

  - Programmatically, only two classes exist `Transfer` and `State` and the rest is number crunching. This is admittedly a personal taste, however, I would like to invite you to think about `zpk`'s and `frd`'s of matlab. `harold` also has facilities to contain identification measurements and logging and so on. (still on paper, see the section **A word** below) 
  - Interactive plots suitable for Ipython (if I can fix a nasty last problem) hence way better-looking and quite less-painful Bode, Nyquist plots, and of course, in case people are still being tortured by them, root loci. 
  - Numerical polynomial operations (as opposed to symbolic) such as Least Common Multiples, Greatest Common Divisors etc. which is tested via pathological examples found in academic papers and to some extent practically on strange transfer matrices. Partially due to the papers age, partially due to computers capabilities, it is almost always competitive, if not usually better, with regards to the results reported by the very same papers.
  - Matrix-pencil based subroutines, Hessenberg forms and transmission zero computations (this one I've tested as much as I can). Agrees with matlab, if not better resolution, and typically much faster for reasons I don't know yet.
  - Better control of internal tolerances that affects minimal realizations, rank tolerances and so on. 
  - When a model is discretized, it remembers the method used (or set later) and also continous/discrete, discrete/discrete type of interconnections recognize the slowest sampling time system and resamples remaining ones. 
  - Helper functions that makes it easy to slice, concatenate, manipulate system data thanks to Python/Numpy flexibility. 



## A word 

This repository will be both the master and the development branch until a proper release candidate with documentation is created. Currently, it is just constantly being updated until a few simple functionalities plotting and academic synthesis techniques such as LQR, pole placement are included. 

Burning issues are the documentation and enabling the `nosetest` framework and also I'm mostly talking to myself hence this needs to go on PyPI.

## Useful Links


 1- There is already a almost-matured control toolbox which is led by Richard Murray et al. ([click for the Github page](https://github.com/python-control/python-control) ) and it can perform already most of the essential tasks. Hence, if you want to have something that resembles the basics of matlab control toolbox give it a try. 

`python-control` emulates the matlab syntax and usage (for a good reason considering the matlab users) which is something I really wish to leave behind completely. Over the years I truly hated that syntax but that's just me. Just to name a few, argument hopping thanks to `nargin,nargout` stuff, inconsistent (sometimes just stupid) error handling, weird amalgam of structs and cells usage... 


  2- By the way, if you are interested in robust control you would probably appreciate the  [Skogestad-Python](https://github.com/alchemyst/Skogestad-Python) project. 


### Contact

If you have a question/comment feel free to shoot one to `harold.of.python@gmail.com`
