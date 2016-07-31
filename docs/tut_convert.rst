Conversion of Dynamic Models
============================

.. warning:: The naming convention of these functions has not been 
    settled yet. So some underscores might go in and the names can 
    change until the release. 


From State() to Transfer()
--------------------------
Suppose we are given a ``State()`` model and we would like 
to obtain the equivalent ``Transfer()`` model. The conversion
is just done straightforwardly by iterating over the columns/rows
of the input/output matrices and forming the state models 
:math:`(A,b_i,c_j,d_{ij})`. Then, computing the transmission 
zeros, poles, and the gain of each of these systems gives the 
result.

Let's take the conversion example from 
`Control Tutorials <http://ctms.engin.umich.edu/CTMS/index.php?aux=Extras_Conversions>`_ .

We first define the state model::

    G = State([[0,    1  ,   0   ,0],
               [0,-0.1818,2.6727 ,0],
               [0,    0  ,   0   ,1],
               [0, -4.545,31.1818,0]],
              [[0],[1.8182],[0],[4.5455]], # B matrix
              eyecolumn(4,[0,2]).T # the first and third column of 4x4 identity matrix
            )

Then we use the conversion function ::

    F = statetotransfer(G) 

If we check the result, we get ::

    F.num

    [[array([[  1.8182    ,   0.        , -44.54599091]])],
     [array([[ 4.5455   , -7.4373471,  0.       ]])]]
    
    F.den
    
    [[array([[  1. ,   0.1818 , -31.1818 ,   6.47857026 ,  -0.  ]])],
     [array([[  1. ,   0.1818 , -31.1818 ,   6.47857026 ,  -0.  ]])]]    


Sometimes, there is no need to actually create a full blown ``Transfer``
instance but just the polynomials are needed. For that, an extra keyword
option is sufficient and intermediate variables are avoided::

    n , d = statetotransfer(G,output='polynomials')
    
Then, ``n , d`` variables hold the numerator and polynomial entries
instead of a model. 


.. warning:: The minimality of the resulting transfer functions are not
    guaranteed. Something along these lines is being implemented but in 
    general, this problem is numerically very rich with strange pathological
    cases hence a post-inspection is always a good idea. 

    

From ``Transfer()`` to ``State()``
----------------------------------

Similar to the state to transfer model conversion, ``transfertostate()``
converts the ``Transfer()`` models to ``State`` models::

    H = transfertostate(F)

The same convenience is also defined for individual matrix output instead
of a full state model::

    aa,bb,cc,dd = transfertostate(F,output='matrices')

If we actually check the resulting state model matrices, ::

    concatenate_state_matrices(H)# Combines the state matrices into a larger matrix

    array([[  0.        ,   1.        ,   0.        ,   0.        ,   0.        ],
           [  0.        ,   0.        ,   1.        ,   0.        ,   0.        ],
           [  0.        ,   0.        ,   0.        ,   1.        ,   0.        ],
           [  0.        ,  -6.47857026,  31.1818    ,  -0.1818    ,   1.        ],
           [  1.8182    ,   0.        , -44.54599091,   0.        ,   0.        ],
           [  4.5455    ,  -7.4373471 ,   0.        ,   0.        ,   0.        ]])

we can get a hint about the methodology from the companion matrix structure. 
For SISO models, the procedure is given elsewhere in most textbooks. For the 
MIMO models, the procedure is a variant of the method given in [#f1]_

The minimality of the resulting model is guaranteed for the SISO and single
row/column models. If the model has more inputs **and** ouputs than one then
depending on the shape, either columnwise or rowwise common denominators are
computed. Obviously, this does not guarantee the resulting model minimality
however at least prevents the dummy pole zero build-up as often encountered 
in practice. 

.. note:: I am experimenting with a directed graph approach to this problem,
    but if someone has a better idea to factorize the common elements faster 
    than this, I would be happy to include it. 


.. note:: For both conversion functions, the discretization information is 
    preserved on the resulting model. 

.. [#f1] W.A. Wolowich, *Linear Multivariable Systems*, Springer, 1974 (Section 4.4). 
