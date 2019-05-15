from harold import State, Transfer, frequency_response
from numpy import eye, abs
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


def test_frequency_response_zero_peak_precision():
    H = Transfer([0.05634, -0.00093524, -0.00093524, 0.05634],
                 [1., -0.7631, -0.1919863, 0.5434631],
                 dt=1)
    rrr, www = frequency_response(H)
    assert abs(rrr[(0.1 < www) & (www < 0.2)]).min() < 1e-6


def test_frequency_response_all_integrators():
    J = Transfer([0.2, 0, 0.2], [1, -2, 1], 0.001)
    rrr, www = frequency_response(J)
    assert abs(rrr[(200 < www) & (www < 300)]).min() < 1e-6

    J = Transfer([0.2, 0, 0.2], [1, 0, 0])
    rrr, www = frequency_response(J)
    assert abs(rrr[(0.1 < www) & (www < 0.2)]).min() < 1e-6


def test_frequency_response_freq_points_close_pole_zero():
    J = Transfer([1, 0, 10], [1, 0, 10.4])
    rrr, www = frequency_response(J)
    assert www.size < 100
