from harold import State, Transfer, step_response_plot, impulse_response_plot

from numpy import eye
from numpy.random import rand, seed
from numpy.linalg import solve
from numpy.testing import assert_equal


def test_step_response_plot_shape():
    seed(1234)
    assert_equal(len(step_response_plot(Transfer(5, dt=0.5)).axes), 1)
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = step_response_plot(G)
    # Ridiculous API
    prop = f.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    assert_equal(prop.get_geometry(), G.shape)
    a = solve(a-eye(5), (a+eye(5)))  # Bilinear map to stabilize just in case
    G = State(a, b, c)


def test_impulse_response_plot_shape():
    seed(1234)
    assert_equal(len(impulse_response_plot(Transfer(5, dt=0.5)).axes), 1)
    a, b, c = -3*eye(5) + rand(5, 5), rand(5, 3), rand(4, 5)
    G = State(a, b, c)
    f = impulse_response_plot(G)
    prop = f.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    assert_equal(prop.get_geometry(), G.shape)
    a = solve(a-eye(5), (a+eye(5)))  # Bilinear map to stabilize just in case
    G = State(a, b, c)
