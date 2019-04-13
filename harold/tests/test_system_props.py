from harold import (Transfer, system_norm, transfer_to_state,
                    controllability_indices)
from test_static_ctrl_design import byersnash_A_B_test_pairs

from numpy import sqrt, array, isinf
from numpy.testing import assert_almost_equal, assert_equal
from pytest import raises as assert_raises


def test_system_norm_args():
    G = Transfer([100], [1, 10, 100])
    assert_raises(ValueError, system_norm, G, p='a')
    assert_raises(ValueError, system_norm, G, p=2j)
    assert_raises(ValueError, system_norm, G, p=3)


def test_system_norm_simple():
    G = Transfer([100], [1, 10, 100])
    assert_almost_equal(system_norm(G), 1.1547, decimal=5)
    assert_almost_equal(system_norm(G, p=2), 2.2360679)

    F = transfer_to_state(G)
    assert_almost_equal(system_norm(F), 1.1547, decimal=5)
    assert_almost_equal(system_norm(F, p=2), 2.2360679)


def test_system_norm_hinf_max_at_0():
    G = Transfer([1, 2], [1, 1])*array([1, -3])
    assert_almost_equal(system_norm(G), 2*sqrt(10))
    # Nonzero feedthrough 2-norm is inf
    assert isinf(system_norm(G, 2))

    # Now transpose the system for the tall matrix case
    G = Transfer([1, 2], [1, 1])*array([[1], [-3]])
    assert_almost_equal(system_norm(G), 2*sqrt(10))
    # Nonzero feedthrough 2-norm is inf
    assert isinf(system_norm(G, 2))


def test_controllability_indices():
    test_cont_ind = [[2, 2], [3, 2], [2, 2], [1, 2], [2, 3], [1, 3],
                     [2, 3, 3], [2, 1, 1], [2, 2], [2, 2], [1, 3]]
    example_gen = byersnash_A_B_test_pairs()
    for ind, (A, B) in enumerate(example_gen):
        indices = controllability_indices(A, B)
        assert_equal(indices, test_cont_ind[ind])
