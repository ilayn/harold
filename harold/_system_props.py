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
from scipy.linalg import solve, norm, eigvals

from ._frequency_domain import frequency_response
from ._classes import Transfer, transfer_to_state
from ._solvers import lyapunov_eq_solver
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['system_norm']


def system_norm(G, p=np.inf, hinf_tol=1e-6, eig_tol=1e-8):
    """
    Computes the system p-norm. Currently, no balancing is done on the
    system, however in the future, a scaling of some sort will be introduced.
    Currently, only H₂ and H∞-norm are understood.

    For H₂-norm, the standard grammian definition via controllability grammian,
    that can be found elsewhere is used.

    Parameters
    ----------
    G : {State,Transfer}
        System for which the norm is computed
    p : {int,np.inf}
        The norm type; `np.inf` for H∞- and `2` for H2-norm
    hinf_tol: float
        When the progress is below this tolerance the result is accepted
        as converged.
    eig_tol: float
        The algorithm relies on checking the eigenvalues of the Hamiltonian
        being on the imaginary axis or not. This value is the threshold
        such that the absolute real value of the eigenvalues smaller than
        this value will be accepted as pure imaginary eigenvalues.

    Returns
    -------
    n : float
        Resulting p-norm

    Notes
    -----
    The H∞ norm is computed via the so-called BBBS algorithm ([1]_, [2]_).

    .. [1] N.A. Bruinsma, M. Steinbuch: Fast Computation of H∞-norm of
        transfer function. System and Control Letters, 14, 1990.

    .. [2] S. Boyd and V. Balakrishnan. A regularity result for the singular
           values of a transfer matrix and a quadratically convergent
           algorithm for computing its L∞-norm. System and Control Letters,
           1990.

    """

    # TODO: Try the corrections given in arXiv:1707.02497

    _check_for_state_or_transfer(G)

    if p not in (2, np.inf):
        raise ValueError('The p in p-norm is not 2 or infinity. If you'
                         ' tried the string \'inf\', use "np.inf" instead')

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

        if any(T.poles.imag):
            J = [np.abs(x.imag/x.real/np.abs(x)) for x in T.poles]
            ind = np.argmax(J)
            low_damp_fr = np.abs(T.poles[ind])
        else:
            low_damp_fr = np.min(np.abs(T.poles))

        f, w = frequency_response(T, w=[0, low_damp_fr], w_unit='rad/s',
                                  output_unit='rad/s')
        if T._isSISO:
            lb = np.max(np.abs(f))
        else:
            # Only evaluated at two frequencies, 0 and wb
            lb = np.max(norm(f, ord=2, axis=2))

        # Finally
        gamma_lb = np.max([lb, norm(d, ord=2)])

        # Start a for loop with a definite end! Convergence is quartic!!
        for x in range(51):

            # (Step b1)
            test_gamma = gamma_lb * (1 + 2*np.sqrt(np.spacing(1.)))

            # (Step b2)
            R = d.T @ d - test_gamma**2 * np.eye(d.shape[1])
            S = d @ d.T - test_gamma**2 * np.eye(d.shape[0])
            # TODO : Implement the result of Benner for the Hamiltonian later
            mat = block([[a - b @ solve(R, d.T) @ c,
                          -test_gamma * b @ solve(R, b.T)],
                         [test_gamma * c.T @ solve(S, c),
                          -(a - b @ solve(R, d.T) @ c).T]])
            eigs_of_H = eigvals(mat)

            # (Step b3)
            im_eigs = eigs_of_H[np.abs(eigs_of_H.real) <= eig_tol]
            # If none left break
            if im_eigs.size == 0:
                gamma_ub = test_gamma
                break
            else:
                # Take the ones with positive imag part
                w_i = np.sort(np.unique(np.abs(im_eigs.imag)))
                # Evaluate the cubic interpolant

                m_i = (w_i[1:] + w_i[:-1]) / 2
                f, w = frequency_response(T, w=m_i, w_unit='rad/s',
                                          output_unit='rad/s')
                if T._isSISO:
                    gamma_lb = np.max(np.abs(f))
                else:
                    gamma_lb = np.max(norm(f, ord=2, axis=2))

        return (gamma_lb + gamma_ub)/2
