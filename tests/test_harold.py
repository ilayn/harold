# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 03:54:39 2015

@author: ilayn
"""

from harold import *
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises 
import numpy.testing as npt

# %% State and Transfer class tests

def test_Transfer_Instantiations():
    assert_raises(TypeError,Transfer)
    # Double list is MIMO, num is SISO --> Static!
    G = Transfer(1,[[1,2,1]])
    assert_equal(len(G.num),1)
    assert_equal(len(G.num[0]),3)
    assert_equal(G.shape,(1,3))
    assert G._isgain
    
    G = Transfer([[1]],[1,2,1])
    assert G._isSISO
    assert_equal(G.num.shape,(1,1))
    assert_equal(G.den.shape,(1,3))
    
    G = Transfer([[1,2]],[1,2,1])
    assert not G._isSISO
    assert_equal(len(G.num[0]),2)
    assert_equal(len(G.den[0]),2)
    assert_equal(G.shape,(1,2))

    num = [[np.array([1,2]),1],[np.array([1,2]),0]]
    den = [1,4,4]
    G = Transfer(num,den)
    assert_equal(len(G.num),2)
    assert_equal(len(G.num[0]),2)
    assert_equal(G.shape,(2,2))
    
    G = Transfer(1)
    assert G._isSISO
    assert_equal(G.num,np.array([[1]]))
    assert_equal(G.den,np.array([[1]]))

    G = Transfer(None,1)
    assert G._isSISO    
    assert_equal(G.num,np.array([[1]]))
    assert_equal(G.den,np.array([[1]]))


    G = Transfer(np.random.rand(3,2))
    assert not G._isSISO
    assert_equal(G.shape,(3,2))
    assert G.poles.size==0
    
    assert_raises(IndexError,Transfer,np.ones((3,2)),[[[1,2],[1,1]]])

def test_Transfer_algebra():

    G = Transfer([[1,[1,1]]],[[[1,2,1],[1,1]]])
    H = Transfer([[[1,3]],[1]],[1,2,1])
    F = G*H
    npt.assert_almost_equal(F.num,np.array([[1,3,4]]))
    npt.assert_almost_equal(F.den,np.array([[1,4,6,4,1]]))
    F = H*G
    npt.assert_almost_equal(F.num[0][0],np.array([[1,3]]))
    npt.assert_almost_equal(F.num[0][1],np.array([[1,4,3]]))
    npt.assert_almost_equal(F.num[1][0],np.array([[1]]))
    npt.assert_almost_equal(F.num[1][1],np.array([[1,1]]))
    
    npt.assert_almost_equal(F.den[0][0],np.array([[1,4,6,4,1]]))
    npt.assert_almost_equal(F.den[0][1],np.array([[1,3,3,1]]))
    npt.assert_almost_equal(F.den[1][0],F.den[0][0])
    npt.assert_almost_equal(F.den[1][1],F.den[0][1])
    
    G = Transfer([[1,[1,1]]],[[[1,2,1],[1,1]]])
    F = - G
    npt.assert_almost_equal(G.num[0][0],-F.num[0][0])
    npt.assert_almost_equal(G.num[0][1],-F.num[0][1])
    H = F + G
    for x in range(2):
        npt.assert_array_equal(H.num[0][x],np.array([[0]]))
        npt.assert_array_equal(H.den[0][x],np.array([[1]]))
        
    G = Transfer(1,[1,2,3])
    F = 5 + G
    npt.assert_almost_equal(F.num,np.array([[5,10,16.]]))
    npt.assert_almost_equal(F.den,G.den)
    F = G + 3
    npt.assert_almost_equal(F.num,np.array([[3,6,10.]]))
    npt.assert_almost_equal(F.den,G.den)

    F = F * 5
    npt.assert_almost_equal(F.num,np.array([[15,30,50]]))
    npt.assert_almost_equal(F.den,G.den)
    
    F *= 0.4
    npt.assert_almost_equal(F.num,np.array([[6,12,20]]))
    npt.assert_almost_equal(F.den,G.den)
    
    num1 = [[[1., 2.], [0., 3.], [2., -1.]],
            [[1.], [4., 0.], [1., -4., 3.]]]
    den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
            [[3., 0., .0], [2., -1., -1.], [1.,0,0,4]]]
    num2 = [[[0,0,0,-1], [2.], [-1., -1.]],
            [[1., 2.], [-1., -2.], [4.]]]
    den2 = [[[-1.], [1., 2., 3.], [-1., -1.]],
            [[-4., -3., 2.], [0., 1.], [1., 0.]]]    

    G = Transfer(num1,den1)
    assert_raises(ValueError,Transfer,num2,den2)
    den2[1][1] = [2,-1,-1]
    F = Transfer(num2,den2)
    H = G + F
    # Flatten list of lists via sum( , []) trick
    
    Hnum = [ np.array([[-1.,  5/3  ,10/3]]),
             np.array([[ 5.,  6.   ,  9.]]),
             np.array([[ 1.,  0.5  , -0.5]]),
             np.array([[ 1.,  3.   ,  0.75 , -0.5]]),
             np.array([[ 3., -2.]]),
             np.array([[ 5., -4.   ,  3.   ,  16.]])
           ]
 
    Hden = [ np.array([[ 1., -2/3  , -4/3]]),
             np.array([[ 1.,  2.       ,  3.       ,  0.   ,  0.]]),
             np.array([[ 1.,  0.5      , -0.5]]),
             np.array([[ 1.,  0.75     , -0.5      ,  0.   ,  0.]]),
             np.array([[ 1., -0.5      , -0.5]]),
             np.array([[ 1.,  0.       ,  0.       ,  4.   ,  0.]]) 
           ]
    Hnum_computed = sum(H.num,[])
    Hden_computed = sum(H.den,[])
    for x in range(np.multiply(*H.shape)):
        npt.assert_almost_equal(Hnum[x],Hnum_computed[x])
        npt.assert_almost_equal(Hden[x],Hden_computed[x])    
    

def test_State_Instantiations():
    assert_raises(TypeError,State)
    G = State(5)
    assert G.a.size == 0
    assert G._isSISO
    assert G._isgain
    assert_equal(G.d,np.array([[5.]]))
    
    G = State(np.eye(2))
    assert_equal(G.shape,(2,2))
    assert G._isgain
    assert not G._isSISO
    
    assert_raises(ValueError,State,
                  np.eye(2),
                  np.array([[1],[2],[3]]),
                  np.array([1,2]),
                  0)

    assert_raises(ValueError,State,
                  np.eye(2),
                  np.array([[1],[2]]),
                  np.array([1,2,3]),
                  0)
    assert_raises(ValueError,State,
                  np.eye(2),
                  np.array([[1],[2]]),
                  np.array([1,2]),
                  np.array([0,0]))


def test_State_algebra():
    static_siso_state = State(5)
    static_mimo_state = State(2.0*np.eye(3))
    dynamic_siso_state = State(haroldcompanion([1,3,3,1]),
                               eyecolumn(3,-1),
                               eyecolumn(3,1).T,
                               0)

    dynamic_mimo_state = State(haroldcompanion([1,3,3,1]),
                               eyecolumn(3,[1,2]),
                               np.eye(3),
                               np.zeros((3,2)))
                               
    dynamic_square_state = State(haroldcompanion([1,3,3,1]),
                               np.eye(3),
                               np.eye(3),
                               np.zeros((3,3)))

    assert_raises(IndexError,dynamic_siso_state.__mul__,static_mimo_state)
    assert_raises(IndexError,dynamic_siso_state.__add__,static_mimo_state)
    assert_raises(IndexError,static_mimo_state.__add__,dynamic_mimo_state)

    F = static_mimo_state * dynamic_mimo_state
    
    npt.assert_almost_equal(F.c,np.eye(3)*2.0)
    npt.assert_almost_equal((dynamic_square_state + static_mimo_state).d,
                            2*np.eye(3))
    
                         



# %% LinAlg Tests 
def test_blockdiag():
    
    a = blockdiag([1,2],[3],[[4],[5]],[[6,7],[8,9]]) - np.array([
                                                        [1,2,0,0,0,0.],
                                                        [0,0,3,0,0,0],
                                                        [0,0,0,4,0,0],
                                                        [0,0,0,5,0,0],
                                                        [0,0,0,0,6,7],
                                                        [0,0,0,0,8,9]
                                                        ])
    b = blockdiag(1,[1],[[1]]) - np.eye(3) 
    
    npt.assert_allclose(a,np.zeros((6,6)))
    npt.assert_allclose(b,np.zeros((3,3)))
    

def test_haroldsvd():
    
    blkdiag_mat = blockdiag(*tuple([10 ** x for x in range(-4,5)]))
    shuffler = np.linalg.qr(np.random.rand(9,9),mode='complete')[0]
    testmat = np.linalg.solve(shuffler,blkdiag_mat).dot(shuffler)
    u,s,v,r = haroldsvd(testmat,also_rank=True)
    
    npt.assert_allclose(s,np.flipud(np.fliplr(blkdiag_mat)))
    assert_equal(r,9)
    
    r = haroldsvd(testmat,also_rank=True,rank_tol=1.00001e-1)[-1]
    assert_equal(r,5)

# %% Polynomial Tests
def test_haroldgcd():
    a = np.array([1,3,2])
    b = np.array([1,4,6,4,1])
    c = np.array([0,1,1])
    d = np.array([])
    e = np.eye(2)
    f = np.array(1)

