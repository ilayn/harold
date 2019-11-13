import numpy as np
from numpy import block
from numpy.linalg.linalg import _makearray
from scipy.linalg import solve, norm, eigvals, qr

from ._array_validators import _assert_2d, _assert_square, _assert_finite
from ._frequency_domain import frequency_response
from ._classes import Transfer, transfer_to_state
from ._solvers import lyapunov_eq_solver
from ._arg_utils import _check_for_state_or_transfer

__all__ = ['system_norm', 'controllability_indices']


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
        :doi:`10.1016/0167-6911(90)90049-Z`

    .. [2] S. Boyd and V. Balakrishnan. A regularity result for the singular
           values of a transfer matrix and a quadratically convergent
           algorithm for computing its L∞-norm. System and Control Letters,
           1990. :doi:`10.1016/0167-6911(90)90037-U`

    """

    # Tried the corrections given in arXiv:1707.02497, couldn't get the gains
    # mentioned in the paper.

    _check_for_state_or_transfer(G)

    if p not in (2, np.inf):
        raise ValueError('The p in p-norm is not 2 or infinity. If you'
                         ' tried the string \'inf\', use "np.inf" instead')

    T = transfer_to_state(G) if isinstance(G, Transfer) else G
    a, b, c, d = T.matrices

    # 2-norm
    if p == 2:
        # Handle trivial infinities
        if not np.allclose(T.d, np.zeros_like(T.d)) or (not T._isstable):
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
            lb = np.max(norm(f, ord=2, axis=(0, 1)))

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
                    gamma_lb = np.max(norm(f, ord=2, axis=(0, 1)))

        return (gamma_lb + gamma_ub)/2


def controllability_indices(A, B, tol=None):
    """Computes the controllability indices for a controllable pair (A, B)

    Controllability indices are defined as the maximum number of independent
    columns per input column of B in the following sense: consider the Kalman
    controllability matrix (widely known as Krylov sequence) ::

        C = [B, AB, ..., A^(n-1)B]

    We start testing (starting from the left-most) every column of this matrix
    whether it is a linear combination of the previous columns. Obviously,
    since C is (n x nm), there are many ways to pick a linearly independent
    subset. We select a version from [1]_. If a new column is dependent
    we delete that column and keep doing this until we are left with a
    full-rank square matrix (this is guaranteed if (A, B) is controllable).

    Then at some point, we are necessarily left with columns that are obtained
    from different input columns ::

        ̅C = [b₁,b₂,b₃...,Ab₁,Ab₃,...,A²b₁,A²b₃,...,A⁽ᵏ⁻¹⁾b₃,...]

    For example, it seems like Ab₂ is deleted due to dependence on the previous
    columns to its left. It can be shown that the higher powers will also be
    dependent and can be removed too. By reordering these columns, we combine
    the terms that involve each bⱼ ::

        ̅C = [b₁,Ab₁,A²b₁,b₂,b₃,Ab₃,A²b₃,...,A⁽ᵏ⁻¹⁾b₃,...]

    The controllability index associated with each input column is defined as
    the number of columns per bⱼ appearing in the resulting matrix. Here, 3
    for first input 1 for second and so on.

    If B is not full rank then the index will be returned as 0 as that column
    bⱼ will be dropped too.

    Parameters
    ----------
    A : ndarray
        2D (n, n) real-valued array
    B : ndarray
        2D (n, m) real-valued array
    tol : float
        Tolerance value for the Arnoldi iteration to decide linear dependence.
        By default it is `sqrt(eps)*n²`

    Returns
    -------
    ind : ndarray
        1D array that holds the computed controllability indices. The sum of
        the values add up to `n` if (A, B) is controllable.

    Notes
    -----
    Though internally not used, this function can also be used as a
    controllability/observability test by summing up the resulting indices and
    comparing to `n`.

    References
    ----------
    .. [1] : W.M. Wonham, "Linear Multivariable Control: A Geometric Approach",
        3rd edition, 1985, Springer, ISBN:9780387960715
    """
    a, _ = _makearray(A)
    b, _ = _makearray(B)
    _assert_2d(a, b)
    _assert_finite(a, b)
    _assert_square(a)

    n, m = b.shape

    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should have the same number of rows")

    # FIXME: Tolerance is a bit arbitrary for now!!
    tol = np.sqrt(np.spacing(1.))*n**2 if tol is None else tol

    # Will be populated below
    remaining_cols = np.arange(m)
    indices = np.zeros(m, dtype=int)

    # Get the orthonormal columns of b first
    q, r, p = qr(b, mode='economic', pivoting=True)
    rank_b = sum(np.abs(np.diag(r)) > max(m, n)*np.spacing(norm(b, 2)))

    remaining_cols = remaining_cols[p][:rank_b].tolist()
    q = q[:, :rank_b]
    indices[remaining_cols] += 1

    w = np.empty((n, 1), dtype=float)
    # Start the iteration - at most n-1 spins
    for k in range(1, n):
        # prepare new A @ Q test vectors
        q_bank = a @ q[:, -len(remaining_cols):]
        for ind, col in enumerate(remaining_cols.copy()):
            w[:] = q_bank[:, [ind]]

            for reorthogonalization in range(2):
                w -= ((q.T @ w).T * q).sum(axis=1, keepdims=True)

            normw = norm(w)
            if normw <= tol:
                remaining_cols.remove(col)
                continue
            else:
                q = np.append(q, w/normw, axis=1)
                indices[col] += 1

        if len(remaining_cols) == 0:
            break

    return indices
