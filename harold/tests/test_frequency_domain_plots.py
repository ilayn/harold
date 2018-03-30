from harold import State, Transfer, bode_plot, nyquist_plot
from numpy import eye
from numpy.random import rand, seed
from numpy.testing import assert_equal


def test_bode_plot_shape():
    seed(1234)
    # SISO
    f = bode_plot(Transfer(5, dt=0.5)).axes
    assert_equal(len(f), 2)
    f.clear()
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = bode_plot(G)
    prop = f.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    assert_equal(prop.get_geometry(), (G.shape[0]*2, G.shape[1]))
    f.clear()


def test_nyquist_plot_shape():
    seed(1234)
    # SISO
    H = Transfer(5, dt=0.5)
    f = nyquist_plot(H)
    assert_equal(len(f.axes), 1)
    f.clear()
    # MIMO
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = nyquist_plot(G)
    prop = f.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    assert_equal(prop.get_geometry(), G.shape)
    f.clear()
