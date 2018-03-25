from harold import (State, Transfer,
                    simulate_linear_system, simulate_step_response)

from numpy import array, eye, ones, zeros, arange
from numpy.testing import assert_allclose, assert_raises, assert_equal


def test_simulate_linear_system_check_x0():
    t = [0, 0.01, 0.02]
    u = [1, 2, 3]
    G = State(5)
    # No IC for static models
    assert_raises(ValueError, simulate_linear_system, G, u, t, x0=1)
    # No IC for Transfer
    G = Transfer(1, [1, 2])
    assert_raises(ValueError, simulate_linear_system, G, u, t, x0=1)
    # x0 is not 1D
    G = State(eye(3), ones([3, 1]), ones([1, 3]), dt=0.01)
    assert_raises(ValueError, simulate_linear_system, G, u, t, x0=ones([5, 5]))
    # x0 isnot compatible with number of states
    assert_raises(ValueError, simulate_linear_system, G, u, t, x0=arange(4))


def test_simulate_linear_system_check_u_and_t():
    G = Transfer(1, [1, 2])
    u = [1, 2, 3]
    t = [0, 1]
    # u and t are not compatible
    assert_raises(ValueError, simulate_linear_system, G, u, t)
    # continuous model needs a valid t
    assert_raises(ValueError, simulate_linear_system, G, u)
    # Time array is not 1D
    t = ones([3, 2])
    assert_raises(ValueError, simulate_linear_system, G, u, t)
    # t is not evenly spaced
    t = [1, 2, 4]
    G = Transfer(1, [1, 1, 1], dt=0.01)
    # t increment is not equal to dt of the model
    t = [0, 0.1, 0.2]
    assert_raises(ValueError, simulate_linear_system, G, u, t)
    # u is empty
    u = []
    t = [0, 0.01, 0.02]
    assert_raises(ValueError, simulate_linear_system, G, u, t)
    # u has more input channels than required
    u = ones([3, 2])
    assert_raises(ValueError, simulate_linear_system, G, u, t)


def test_simulate_linear_system_response():
    G = Transfer(1, [1, 1, 1], dt=0.01)
    u = [1, 2, 3]
    # G is discrete
    y, tout = simulate_linear_system(G, u)
    assert_allclose(y, array([[0, 0, 1.]]).T)
    assert_allclose(tout, array([0, 0.01, 0.02]))
    # G is continuous and static
    mat = 4*eye(5) - 7*ones(5)
    G = State(mat)
    u = ones([10, 5])
    t = arange(10)
    y, tout = simulate_linear_system(G, u, t)
    assert_allclose(tout, arange(10))
    assert_allclose(y - u @ mat.T, zeros([10, 5]))


def test_simulate_step_response_response():
    G = State(4)
    yout, tout = simulate_step_response(G)
    assert_allclose(yout, 4*ones(len(yout))[:, None])
    G = State(4, dt=0.01)
    y2, t2 = simulate_step_response(G)
    assert_allclose(y2, 4*ones(len(y2))[:, None])
    assert_equal(len(yout), len(y2))
