from harold import State, Transfer, frequency_response
from numpy import eye
from numpy.random import rand, seed


# TODO: There is not much tested here, only success
def test_frequency_response():
    seed(1234)
    # SISO, only integrators
    G = Transfer([1, 0], [1, 0, 0, 0])
    _, _ = frequency_response(G)
    G = Transfer([1, -1], [1, -2, 1], dt=0.1)
    _, _ = frequency_response(G)
    # Static Gains
    G = Transfer(5, dt=0.5)
    _, _ = frequency_response(G)
    G = Transfer(eye(5))
    _, _ = frequency_response(G)
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    _, _ = frequency_response(G)
    G = State(a, b, c, dt=1)
    _, _ = frequency_response(G)
