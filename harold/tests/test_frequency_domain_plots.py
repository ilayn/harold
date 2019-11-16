from harold import State, Transfer, bode_plot, nyquist_plot
from numpy import eye
from numpy.random import rand, seed


def test_bode_plot_shape():
    seed(1234)
    # SISO
    f = bode_plot(Transfer(5, dt=0.5))
    assert f._gridspecs[0].get_geometry() == (2, 1)
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = bode_plot(G)
    assert f._gridspecs[0].get_geometry() == (G.shape[0]*2, G.shape[1])


def test_nyquist_plot_shape():
    seed(1234)
    # SISO
    H = Transfer(5, dt=0.5)
    f = nyquist_plot(H)
    assert f._gridspecs[0].get_geometry() == (1, 1)
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = nyquist_plot(G)
    assert f._gridspecs[0].get_geometry() == G.shape
