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
from numpy import count_nonzero, block
from ._frequency_domain import frequency_response
from ._classes import Transfer, transfer_to_state
from ._solvers import lyapunov_eq_solver
from ._system_funcs import _minimal_realization_state
from scipy.linalg import solve, eigvals, svdvals, LinAlgError
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['system_norm']


def system_norm(G, p=np.inf, max_iter_limit=100, hinf_tol=1e-6, eig_tol=1e-12):
    """
    Computes the system p-norm. Currently, no balancing is done on the
    system, however in the future, a scaling of some sort will be introduced.
    Another short-coming is that while sounding general, only
    :math:`\\mathcal{H}_2` and :math:`\\mathcal{H}_\\infty`
    norm are understood.

    For :math:`\\mathcal{H}_2` norm, the standard grammian definition via
    controllability grammian, that can be found elsewhere is used.

    Currently, the :math:`\\mathcal{H}_\\infty` norm is computed via
    so-called Boyd-Balakhrishnan-Bruinsma-Steinbuch algorithm [2]_.

    [2] N.A. Bruinsma, M. Steinbuch: Fast Computation of
    :math:`\\mathcal{H}_\\infty`-norm of transfer function. System and Control
    Letters, 14, 1990

    Parameters
    ----------
    G : {State,Transfer}
        System for which the norm is computed
    p : {int,Inf}
        Whether the rank of the matrix should also be reported or not.
        The returned rank is computed via the definition taken from the
        official numpy.linalg.matrix_rank and appended here.
    max_iter_limit: int
        Stops the iteration after max_iter_limit many times spinning the
        loop. Very unlikely but might be required for pathological examples.
    hinf_tol: float
        When the progress is below this tolerance the result is accepted
        as *converged*.
    eig_tol: float
        The algorithm relies on checking the eigenvalues of the Hamiltonian
        being on the imaginary axis or not. This value is the threshold
        such that the absolute real value of the eigenvalues smaller than
        this value will be accepted as pure imaginary eigenvalues.

    Returns
    -------
    n : float
        Resulting p-norm
    """
    _check_for_state_or_transfer(G)

    if not isinstance(p, (int, float)):
        raise ValueError('The p in p-norm is not an integer or float. If you'
                         ' tried the string \'inf\', use np.Inf instead')

    T = transfer_to_state(G) if isinstance(G, Transfer) else G
    a, b, c, d = T.matrices

    # 2- norm
    if p == 2:
        # Handle trivial infinities
        if T._isgain:
            # If nonzero -> infinity, if zero -> zero
            if count_nonzero(T.d) > 0:
                return np.Inf
            else:
                return 0.

        if not T._isstable:
            return np.Inf

        if T.SamplingSet == 'R':
            x = lyapunov_eq_solver(a.T, b @ b.T)
            return np.sqrt(np.trace(c @ x @ c.T))
        else:
            x = lyapunov_eq_solver(a.T, b @ b.T, form='d')
            return np.sqrt(np.trace(c @ x @ c.T + d @ d.T))
    # ∞-norm
    elif np.isinf(p):
        if not T._isstable:
            return np.Inf

        # Initial gamma0 guess
        # Get the max of the largest svd of either
        #   - feedthrough matrix
        #   - G(iw) response at the pole with smallest damping
        #   - G(iw) at w = 0

        # Formula (4.3) given in Bruinsma, Steinbuch Sys.Cont.Let. (1990)

        if any(np.imag(T.poles)):
            J = [np.abs(np.imag(x)/np.real(x)/np.abs(x)) for x in T.poles]
            ind = np.argmax(J)
            low_damp_freq = np.abs(T.poles[ind])
        else:
            low_damp_freq = np.min(np.abs(T.poles))

        f, w = frequency_response(T, w=[0, low_damp_freq], w_unit='rad/s',
                                  output_unit='rad/s')
        if T._isSISO:
            lb = np.max(np.abs(f))
        else:
            # Only evaluated at two frequencies, 0 and wb
            lb = np.max([svdvals(f[:, :, x]) for x in range(2)])

        # Finally
        gamma_lb = np.max([lb, np.max(svdvals(d))])

        # Start a for loop with a definite end!
        for x in range(max_iter_limit):

            # (Step b1)
            test_gamma = gamma_lb * (1 + 2*hinf_tol)

            # (Step b2)
            R = d.T @ d - test_gamma**2 * np.eye(d.shape[1])
            S = d @ d.T - test_gamma**2 * np.eye(d.shape[0])
            # TODO : It might be good to implement the result of Benner et al.
            # for the Hamiltonian later
            try:
                mat = block([[a - b @ solve(R, d.T) @ c,
                              -test_gamma * b @ solve(R, b.T)],
                             [test_gamma * c.T @ solve(S, c),
                              -(a - b @ solve(R, d.T) @ c).T]])
                eigs_of_H = eigvals(mat)
            except LinAlgError as err:
                if 'singular matrix' == str(err):
                    at, bt, ct = _minimal_realization_state(a, b, c)
                    if at.size == 0:
                        return lb, None
                    else:
                        raise ValueError('The A matrix is/looks like stable '
                                         'but somehow I managed to screw it '
                                         'up. Please send an insulting mail '
                                         'to ilhan together with this example'
                                         '.')

            # (Step b3)
            #
            if all(np.abs(np.real(eigs_of_H)) > eig_tol):
                gamma_ub = test_gamma
                break
            else:
                w_i = np.sort(np.unique(np.imag(eigs_of_H)))
                m_i = (w_i[1:] + w_i[:-1]) / 2

                # TODO : Still needs five times speed-up

                f, w = frequency_response(T, w=m_i, w_unit='rad/s',
                                          output_unit='rad/s')
                if T._isSISO:
                    gamma_lb = np.max(np.abs(f))
                else:
                    gamma_lb = np.max([svdvals(f[:, :, x]) for x
                                       in range(len(m_i))])

        return (gamma_lb + gamma_ub)/2

    else:
        raise ValueError('I can only handle 2- and ∞-norms for now.')
