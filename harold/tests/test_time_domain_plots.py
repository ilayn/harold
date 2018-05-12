from harold import State, Transfer, step_response_plot, impulse_response_plot

from numpy import eye
from numpy.random import rand, seed


def test_step_response_plot_shape():
    seed(1234)
    f = step_response_plot(Transfer(5, dt=0.5))
    assert f.shape == (1, 1)
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = step_response_plot(G)
    assert f.shape == G.shape


def test_impulse_response_plot_shape():
    seed(1234)
    f = impulse_response_plot(Transfer(5, dt=0.5))
    assert f.shape == (1, 1)
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = impulse_response_plot(G)
    assert f.shape == G.shape
