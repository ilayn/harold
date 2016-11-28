"""
The MIT License (MIT)

Copyright (c) 2016 Ilhan Polat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
from ._frequency_domain import frequency_response
from ._classes import Transfer, State, transfer_to_state
from ._solvers import lyapunov_eq_solver
from ._system_funcs import minimal_realization
from scipy.linalg import solve, eigvals

__all__ = ['system_norm']


def system_norm(state_or_transfer,
                p=np.inf,
                validate=False,
                verbose=False,
                max_iter_limit=100,
                hinf_tolerance=1e-10,
                eig_tolerance=1e-12
                ):
    """
    Computes the system p-norm. Currently, no balancing is done on the
    system, however in the future, a scaling of some sort will be introduced.
    Another short-coming is that while sounding general, only
    :math:`\\mathcal{H}_2` and :math:`\\mathcal{H}_\\infty`
    norm are understood.

    For :math:`\\mathcal{H}_2` norm, the standard grammian definition via
    controllability grammian, that can be found elsewhere is used.

    Currently, the :math:`\\mathcal{H}_\\infty` norm is computed via
    so-called Boyd-Balakhrishnan-Bruinsma-Steinbuch algorithm (See e.g. [2]).

    However, (with kind and generous help of Melina Freitag) the algorithm
    given in [1] is being implemented and depending on the speed benefit
    might be replaced as the default.

    [1] M.A. Freitag, A Spence, P. Van Dooren: Calculating the
    :math:`\\mathcal{H}_\\infty`-norm using the implicit determinant method.
    SIAM J. Matrix Anal. Appl., 35(2), 619-635, 2014

    [2] N.A. Bruinsma, M. Steinbuch: Fast Computation of
    :math:`\\mathcal{H}_\\infty`-norm of transfer function. System and Control
    Letters, 14, 1990

    Parameters
    ----------
    state_or_transfer : {State,Transfer}
        System for which the norm is computed
    p : {int,Inf}
        Whether the rank of the matrix should also be reported or not.
        The returned rank is computed via the definition taken from the
        official numpy.linalg.matrix_rank and appended here.

    validate: boolean
        If applicable and if the resulting norm is finite, the result is
        validated via other means.

    verbose: boolean
        If True, the (some) internal progress is printed out.

    max_iter_limit: int
        Stops the iteration after max_iter_limit many times spinning the
        loop. Very unlikely but might be required for pathological examples.

    hinf_tolerance: float
        Convergence check value such that when the progress is below this
        tolerance the result is accepted as *converged*.

    eig_tolerance: float
        The algorithm relies on checking the eigenvalues of the Hamiltonian
        being on the imaginary axis or not. This value is the threshold
        such that the absolute real value of the eigenvalues smaller than
        this value will be accepted as pure imaginary eigenvalues.

    Returns
    -------
    n : float
        Computed norm. In NumPy, infinity is also float-type

    omega : float
        For Hinf norm, omega is the frequency where the maximum is attained
        (technically this is a numerical approximation of the supremum).

    """
    if not isinstance(state_or_transfer, (State, Transfer)):
        raise TypeError('The argument should be a State or Transfer. Instead '
                        'I received {0}'.format(type(
                                    state_or_transfer).__qualname__))

    if isinstance(state_or_transfer, Transfer):
        now_state = transfer_to_state(state_or_transfer)
    else:
        now_state = state_or_transfer

    if not isinstance(p, (int, float)):
        raise('The p in p-norm is not an integer or float.'
              'If you tried the string \'inf\', use Numpy.Inf instead')

    # Two norm
    if p == 2:
        # Handle trivial infinities
        if now_state._isgain:
            # If nonzero -> infinity, if zero -> zero
            if np.count_nonzero(now_state.d) > 0:
                return np.Inf
            else:
                return 0.

        if not now_state._isstable:
            return np.Inf

        if now_state.SamplingSet == 'R':
            a, b, c = now_state.matrices[:3]
            x = lyapunov_eq_solver(a, b.dot(b.T))
            return np.sqrt(np.trace(c.dot(x.dot(c.T))))
        else:
            a, b, c, d = now_state.matrices
            x = lyapunov_eq_solver(a, b.dot(b.T), form='d')
            return np.sqrt(np.trace(c.dot(x.dot(c.T))+d.dot(d.T)))

    elif np.isinf(p):
        if not now_state._isstable:
            return np.Inf, None

        a, b, c, d = now_state.matrices

        # Initial gamma0 guess
        # Get the max of the largest svd of either
        #   - feedthrough matrix
        #   - G(iw) response at the pole with smallest damping
        #   - G(iw) at w = 0

        # We only need the svd vals hence call numpy svd
        lb1 = np.max(np.linalg.svd(d, compute_uv=False))

        # Formula (4.3) given in Bruinsma, Steinbuch Sys.Cont.Let. (1990)
        if any(np.abs(np.imag(now_state.poles) > 1e-5)):
            low_damp_freq = np.abs(now_state.poles[np.argmax([
                                      np.abs(np.imag(x)/np.real(x)/np.abs(x)
                                             ) for x in now_state.poles])])
        else:
            low_damp_freq = np.min(np.abs(now_state.poles))

        f, w = frequency_response(now_state, custom_grid=[0, low_damp_freq])
        if now_state._isSISO:
            lb2 = np.max(np.abs(f))
        else:
            lb2 = [np.linalg.svd(f[:, :, x], compute_uv=0) for x in range(2)]
            lb2 = np.max(lb2)

        # Finally
        gamma_lb = np.max([lb1, lb2])

        # Constant part of the Hamiltonian to shave off a few flops
        H_of_gam_const = np.c_[np.r_[a, np.zeros_like(a)],
                               np.r_[c.T.dot(c), a.T]]
        H_of_gam_lfact = np.r_[b, c.T.dot(d)]
        H_of_gam_rfact = np.c_[-d.T.dot(c), b.T]

        # Start a for loop with a definite end !
        for x in range(max_iter_limit):

            # (Step b1)
            test_gamma = gamma_lb * (1 + 2*hinf_tolerance)

            # (Step b2)
            R_of_gam = d.T.dot(d) - test_gamma**2 * np.eye(d.shape[1])
            # TODO : It might be good to implement the result of Benner et al.
            # for the Hamiltonian later
            try:
                eigs_of_H = eigvals(H_of_gam_const - H_of_gam_lfact.dot(
                                    solve(R_of_gam, H_of_gam_rfact)))
            except np.linalg.linalg.LinAlgError as err:
                if 'singular matrix' == str(err):
                    at, bt, ct = minimal_realization(a, b, c)
                    if at.size == 0:
                        return lb1, None
                    else:
                        raise ValueError('The A matrix is/looks like stable '
                                         'but somehow I managed to screw it '
                                         'up. Please send an insulting mail '
                                         'to ilhan together with this example'
                                         '.')

            # (Step b3)
            #
            if all(np.abs(np.real(eigs_of_H)) > eig_tolerance):
                gamma_ub = test_gamma
                break
            else:
                # Pick up the ones with the tiny or zero real parts
                # but leave out the euclidian ball around the origin
                mix_imag_eigs = eigs_of_H[np.abs(
                            np.real(eigs_of_H)) < eig_tolerance]
                imag_eigs = np.unique(np.round(np.abs(
                              mix_imag_eigs[
                                  np.abs(np.imag(eigs_of_H)) > eig_tolerance]),
                                  decimals=7))

                m_i = [np.average(imag_eigs[x], imag_eigs[x+1])
                       for x in range(len(imag_eigs))]

                # TODO : Clean up this mess above with imag_eigs etc.
                # TODO : Still needs five times speed-up

                f, w = frequency_response(now_state, custom_grid=m_i)
                if now_state._isSISO:
                    gamma_lb = np.max(np.abs(f))
                else:
                    gamma_lb = [np.linalg.svd(
                            f[::x], compute_uv=0) for x in range(len(m_i))]
                    gamma_lb = np.max(lb2)

        return np.mean([gamma_lb, gamma_ub])

        #           [ A     0  ]  _ [  B  ] * [        -1 ] * [-C.T*D].T
        #  H(gam) = [C.T*C  A.T]    [C.T*D]   [R_of_gam   ]   [   B  ]

    else:
        raise('I can only handle the cases for p=2,inf for now.')
