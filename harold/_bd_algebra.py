
import numpy as np
from scipy.linalg import svdvals, solve
from harold._classes import (State, Transfer, transfer_to_state,
                             state_to_transfer)
from harold._aux_linalg import matrix_slice
from harold._arg_utils import _check_for_int_float_array, _check_equal_dts

__all__ = ['feedback']


def feedback(G, H, negative=True):
    """
    Feedback interconnection of two models in the following configuration ::

                        +     ┌───────┐
                   u ────O────┤   G   ├────┬───> y
                         │s   └───────┘    │
                         │                 │
                         │    ┌───────┐    │
                         └────┤   H   ├────┘
                              └───────┘

    In this configuration nonmatching shapes will not be broadcasted.

    The feedback sign `s` in the block diagram is determined by the keyword
    `negative` and by default it is negative feedback hence `True`.

    All models are converted to State type and then the loop is closed. The
    resulting closed loop model is not necessarily minimal.

    Parameters
    ----------
    G : {State, Transfer, ndarray}
        The model that is on the forward path
    H : {State, Transfer, ndarray}
        The model that is on the feedback path
    negative : bool, optional
        The default is True

    Returns
    -------
    CL : {State, Transfer, ndarray}
        The closed loop model

    """
    # REWORK: P/Z cancellations. See feedback(tf([2,2],[1,1]),tf(1,[1,0,0,0]])
    # TODO : implement LFT interconnections to do this more properly.

    g, h, conv_tf = _check_for_feedback_ic(G, H)
    dt = g.SamplingPeriod
    sign = -1. if negative else 1.
    # In the following case
    #
    #                              ┌───────┐
    #                        +     │ A │ B │
    #                   r ────O────┤ ──┼── ├────┬───> y
    #                         │-   │ C │ D │    │
    #                         │    └───────┘    │
    #                         │                 │
    #                         │    ┌───────┐    │
    #                         │    │ K │ L │    │
    #                         └────┤ ──┼── ├────┘
    #                          u   │ M │ N │
    #                              └───────┘
    # In this configuration the closed loop system is given by
    #    ┌          ┐    ┌    ┐
    #    │ A  0  B  │    │  B │      ┌             ┐
    #    │ LC K  LD │ ∓  │ LD │ Δ⁻¹  │ NC   M   ND │
    #    │ C  0  D  │    │  D │      └             ┘
    #    └          ┘    └    ┘
    # and also Δ = (I ± ND). If posive feedback is desired then

    # Depending on static models certain channels are removed. If both are
    # static models then in boils down to D ∓ DΔ⁻¹ND (or simply (I ± DN)⁻¹D)
    (p1, m1), n1, n2 = g.shape, g.NumberOfStates,  h.NumberOfStates
    (a, b, c, d), (k, l, m, n) = g.matrices, h.matrices
    delta = np.eye(m1) - n @ d

    # Since models are typically at most up to O(10^2) SVD is not the most
    # time consuming step thus we check for well-posedness via svdvals instead
    # of tedious dgecon
    wpvals = svdvals(delta)
    if np.min(wpvals) <= m1 * np.spacing(10.):
        raise ValueError('The feedback interconnection is not well-posed. In'
                         f' particular, (I{"+" if negative else "-"}D₂D₁) is '
                         'not reliably invertible.')

    # if both are static
    if g._isgain and h._isgain:
        if np.any(d) and np.any(n):
            mat = d + sign * (d @ solve(delta, n @ d))

        return Transfer(mat, dt=dt) if conv_tf else State(mat, dt=dt)
    # if both are dynamic
    elif (not g._isgain) and (not h._isgain):
        acl = np.block([[a, np.zeros([n1, n2]), b],
                        [l@c, k, l@d],
                        [c, np.zeros([p1, n2]), d]])
        bcl = np.block([[b], [l@d], [d]])
        ncl = np.block([n@c, m, n@d])
        Mcl = acl + sign * bcl @ solve(delta, ncl)

        G = State(*matrix_slice(Mcl, (n1 + n2, n1 + n2)), dt=dt)
        if conv_tf:
            G = state_to_transfer(G)

        return G
    # one of them is static
    else:
        if g._isgain:
            acl = np.block([[k, l@d],
                            [np.zeros([p1, n2]), d]])
            bcl = np.block([[l@d], [d]])
            ncl = np.block([m, n@d])
            Mcl = acl + sign * bcl @ solve(delta, ncl)
            s = n2
        else:
            acl = np.block([[a, b],
                            [c, d]])
            bcl = np.block([[b], [d]])
            ncl = np.block([n@c, n@d])
            Mcl = acl + sign * bcl @ solve(delta, ncl)
            s = n1
        G = State(*matrix_slice(Mcl, (s, s)), dt=dt)
        if conv_tf:
            G = state_to_transfer(G)

        return G


def _check_for_feedback_ic(G, H):
    """
    A helper function to sanitize the inputs to the feedback() function

    Paramaters
    ----------
    G, H : {State, Transfer, ndarray, int, float}
        Models on the feedforward and feedback paths

    Returns
    -------
    g, h : {State}
        Sanitized models
    tf_convert : bool
        If, initially, both models are Transfer representations, then this flag
        is set to True and the returned object is a Transfer. Otherwise State
        always have higher precedence.

    """
    # Get the boring model comparison and shape checks out of the way.
    flag_sys = [isinstance(G, (State, Transfer)),
                isinstance(H, (State, Transfer))]
    tf_convert = False

    # Both are either ndarrays ints floats or rejected
    if not np.any(flag_sys):
            g, h = [State(_check_for_int_float_array(x)) for x in ([G, H])]

    # both are system models
    elif np.all(flag_sys):
        _check_equal_dts(G, H)
        flag_tf = [isinstance(G, Transfer), isinstance(H, Transfer)]

        # both transfer
        if np.all(flag_tf):
            g, h = transfer_to_state(G), transfer_to_state(H)
            tf_convert = True
        # one is a transfer
        elif np.logical_xor(*flag_tf):
            if flag_tf[0]:
                g, h = transfer_to_state(G), H
            else:
                g, h = G, transfer_to_state(H)
        # both state
        else:
            g, h = G, H

    else:
        # one of them is a model the other one is checked
        if flag_sys[0]:
            # If G is a model then check H
            h = _check_for_int_float_array(H)
            h = State(h, dt=G.SamplingPeriod)
            tf_convert = True if isinstance(G, Transfer) else False
            g = G if isinstance(G, State) else transfer_to_state(G)
        else:
            g = _check_for_int_float_array(G)
            g = State(g, dt=H.SamplingPeriod)
            tf_convert = True if isinstance(H, Transfer) else False
            h = H if isinstance(H, State) else transfer_to_state(H)

    (p1, m1), (p2, m2) = g.shape, h.shape
    if p1 != m2 or m1 != p2:
        raise ValueError('Forward path model should have equal number of'
                         ' inputs/outputs to the feedback path outputs/inputs.'
                         f' G has the shape {g.shape} but H has {h.shape}.')

    return g, h, tf_convert
