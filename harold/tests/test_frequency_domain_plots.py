from harold import State, Transfer, bode_plot, nyquist_plot

from numpy import eye
from numpy.random import rand, seed
from numpy.testing import assert_equal


def test_bode_plot_shape():
    seed(1234)
    # SISO
    assert_equal(len(bode_plot(Transfer(5, dt=0.5)).axes), 2)
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = bode_plot(G)
    prop = f.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    assert_equal(prop.get_geometry(), (G.shape[0]*2, G.shape[1]))


def test_nyquist_plot_shape():
    seed(1234)
    # SISO
    assert_equal(len(nyquist_plot(Transfer(eye(2), dt=0.5)).axes), 4)
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = nyquist_plot(G)
    prop = f.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    assert_equal(prop.get_geometry(), G.shape)
