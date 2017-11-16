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
from copy import deepcopy
import numpy as np
from numpy.linalg import cond, eig, norm
from scipy.linalg import svdvals, qr, block_diag
from ._aux_linalg import haroldsvd, matrix_slice, e_i
from ._classes import State, Transfer
from ._arg_utils import _check_for_state_or_transfer


__all__ = ['staircase', 'cancellation_distance', 'minimal_realization']


# TODO : Too much matlab-ish coding, clean up !!
def staircase(A, B, C,
              compute_T=False, form='c', invert=False, block_indices=False):
    """
    The staircase form is used very often to assess system properties.
    Given a state system matrix triplet ``A``, ``B``, ``C``, this function
    computes the so-called controller/observer-Hessenberg form such that the
    resulting system matrices have the block-form (x denoting the possibly
    nonzero blocks) ::
                                [x x x x x|x]
                                [x x x x x|0]
                                [0 x x x x|0]
                                [0 0 x x x|0]
                                [0 0 0 x x|0]
                                [---------|-]
                                [x x x x x|x]
                                [x x x x x|x]

    For controllability and observability, the existence of zero-rank
    subdiagonal blocks can be checked, as opposed to forming the Kalman
    matrix and checking the rank. Staircase method can numerically be
    more stable since for certain matrices, A^n computations can
    introduce large errors (for some A that have entries with varying
    order of magnitudes). But it is also prone to numerical rank guessing
    mismatches.
    Notice that, if we use the pertransposed data, then we have the
    observer form which is usually asked from the user to supply
    the data as :math:`A,B,C \Rightarrow A^T,C^T,B^T` and then transpose
    back the result. Instead, the additional ``form`` option denoting
    whether it is the observer or the controller form that is requested.

    Parameters
    ----------
    A : (n, n) array_like
        State array
    B : (n, m) array_like
        Input array
    C : (p, n) array_like
        Output array
    compute_T : bool, optional
        Whether the transformation matrix T should be computed or not
    form : str, optional
        Determines whether the controller- or observer-Hessenberg form
        will be computed via ``'c'`` or ``'o'`` values.
    invert : bool, optional
        Whether to select which side the B or C matrix will be compressed.
        For example, the default case returns the B matrix with (if any)
        zero rows at the bottom. invert option flips this choice either in
        B or C matrices depending on the "form" switch.
    block_indices : bool, optional

    Returns
    -------
    Ah : (n, n) array_like
        Resulting State array
    Bh : (n, m) array_like
        Resulting Input array
    Ch : (p, n) array_like
        Resulting Output array
    T : (n,n) 2D numpy array
        If the boolean ``compute_T`` is true, returns the transformation
        matrix such that ::

            [ T⁻¹AT | T⁻¹B ]   [ Ah | Bh ]
            [-------|------] = [----|----]
            [   CT  |      ]   [ Ch |    ]

        is in the desired staircase form.
    k: Numpy array
        If the boolean ``block_indices`` is true, returns the array
        of controllable/observable block sizes identified by the algorithm
        during elimination.

    """

    if form not in {'c', 'o'}:
        raise ValueError('The "form" key can only take values'
                         '\"c\" or \"o\" denoting\ncontroller- or '
                         'observer-Hessenberg form.')
    if form == 'o':
        A, B, C = A.T, C.T, B.T

    n = A.shape[0]
    ub, sb, vb, m0 = haroldsvd(B, also_rank=True)
    cble_block_indices = np.empty((1, 0))

    # Trivially  Uncontrollable Case
    # Skip the first branch of the loop by making m0 greater than n
    # such that the matrices are returned as is without any computation
    if m0 == 0:
        m0 = n + 1
        cble_block_indices = np.array([0])

    # After these, start the regular case
    if n > m0:  # If it is not a square system with full rank B

        A0 = ub.T.dot(A.dot(ub))

        # Row compress B and consistent zero blocks with the reported rank
        B0 = sb.dot(vb)
        B0[m0:, :] = 0.
        C0 = C.dot(ub)
        cble_block_indices = np.append(cble_block_indices, m0)

        if compute_T:
            P = block_diag(np.eye(n-ub.T.shape[0]), ub.T)

        # Since we deal with submatrices, we need to increase the
        # default tolerance to reasonably high values that are
        # related to the original data to get exact zeros
        tol_from_A = n*norm(A, 1)*np.spacing(1.)

        # Region of interest
        m = m0
        ROI_start = 0
        ROI_size = 0

        for _ in range(A.shape[0]):
            ROI_start += ROI_size
            ROI_size = m
            h1, h2, h3, h4 = matrix_slice(A0[ROI_start:, ROI_start:],
                                          (ROI_size, ROI_size))
            uh3, sh3, vh3, m = haroldsvd(h3, also_rank=True,
                                         rank_tol=tol_from_A)

            # Make sure reported rank and sh3 are consistent about zeros
            sh3[sh3 < tol_from_A] = 0.

            # If the resulting subblock is not full row or zero rank
            if 0 < m < h3.shape[0]:
                cble_block_indices = np.append(cble_block_indices, m)
                if compute_T:
                    P = block_diag(np.eye(n-uh3.shape[1]), uh3.T).dot(P)
                A0[ROI_start:, ROI_start:] = np.r_[np.c_[h1, h2],
                                                   np.c_[sh3.dot(vh3),
                                                         uh3.T.dot(h4)]]
                A0 = A0.dot(block_diag(np.eye(n-uh3.shape[1]), uh3))
                C0 = C0.dot(block_diag(np.eye(n-uh3.shape[1]), uh3))
                # Clean up
                A0[abs(A0) < tol_from_A] = 0.
                C0[abs(C0) < tol_from_A] = 0.
            elif m == h3.shape[0]:
                cble_block_indices = np.append(cble_block_indices, m)
                break
            else:
                break

        if invert:
            A0 = np.fliplr(np.flipud(A0))
            B0 = np.flipud(B0)
            C0 = np.fliplr(C0)
            if compute_T:
                P = np.flipud(P)

        if form == 'o':
            A0, B0, C0 = A0.T, C0.T, B0.T

        if compute_T:
            if block_indices:
                return A0, B0, C0, P.T, cble_block_indices
            else:
                return A0, B0, C0, P.T
        else:
            if block_indices:
                return A0, B0, C0, cble_block_indices
            else:
                return A0, B0, C0

    else:  # Square system B full rank ==> trivially controllable
        cble_block_indices = np.array([n])
        if form == 'o':
            A, B, C = A.T, C.T, B.T

        if compute_T:
            if block_indices:
                return A, B, C, np.eye(n), cble_block_indices
            else:
                return A, B, C, np.eye(n)
        else:
            if block_indices:
                return A, B, C, cble_block_indices
            else:
                return A, B, C


def cancellation_distance(F, G):
    """
    Given matrices :math:`F,G`, computes the upper and lower bounds of
    the perturbation needed to render the pencil [F-pI | G]` rank deficient.
    It is used for assessing the controllability/observability degenerate
    distance and hence for minimality assessment.

    Parameters
    ----------
    A : (n, n) array_like
        Square array
    B : (n, m) array_like
        Input array

    Returns
    -------
    upper2 : float
        Upper bound on the norm of the perturbation ``[dF | dG]`` such
        that ``[F+dF-pI | G+dG]` is rank deficient for some ``p``.
    upper1 : float
        A theoretically softer upper bound than ``upper2`` for the same
        norm.
    lower0 : float
        Lower bound on the norm given in ``upper2``
    e_f : complex
        Indicates the eigenvalue that renders ``[F+dF-pI | G+dG ]`` rank
        deficient
    radius : float
        The perturbation with the norm bound ``upper2`` is located within
        a disk in the complex plane whose center is on ``e_f`` and whose
        radius is bounded by this output.

    Notes
    -----
    Implements the algorithm given in D.Boley SIMAX vol.11(4) 1990.

    """
    if not np.equal(*F.shape):
        raise ValueError('F input must be a square array.')
    if F.shape[0] != G.shape[0]:
        raise ValueError('F and G inputs must have the same number of rows.')

    A = np.c_[F, G].T
    n, m = A.shape
    B = e_i(n, np.s_[:m])
    D = e_i(n, np.s_[m:])
    C = qr(2*np.random.rand(n, n-m) - 1, mode='economic')[0]
    evals, V = eig(np.c_[A, C])
    K = cond(V)
    X = V[:m, :]
    Y = V[m:, :]

    upp0 = [0]*n
    for x in range(n):
        upp0[x] = norm((C-evals[x]*D).dot(Y[:, x])) / norm(X[:, x])

    f = np.argsort(upp0)[0]
    e_f = evals[f]
    upper1 = upp0[f]
    upper2 = svdvals(A - e_f*B)[-1]
    lower0 = upper2/(K+1)
    radius = upper2*K

    return upper2, upper1, lower0, e_f, radius


def minimal_realization(G, tol=1e-6):
    """
    Given system realization G, this computes minimal realization such that
    if a State representation is given then the returned representation is
    controllable and observable within the given tolerance ``tol``. If
    a Transfer representation is given, then the fractions are simplified
    in the representation entries.

    Parameters
    ----------
    G : State, Transfer
        System representation to be checked for minimality
    tol: float
        The sensitivity threshold for the cancellation.

    Returns
    -------
    G_min : State, Transfer
        Minimal realization of the input `G`

    Notes
    -----
    For State() inputs the alogrithm uses ``cancellation_distance()`` and
    ``staircase()`` for the tests. A basic two pass algorithm performs:

        1- First distance to mode cancellation is computed then also
        the Hessenberg form is obtained with the identified o'ble/c'ble
        block numbers.

        2- If staircase form reports that there are no cancellations but the
        distance is less than the tolerance, distance wins and the
        corresponding mode is removed.

    For Transfer() inputs, every entry of the representation is checked for
    pole/zero cancellations and ``tol`` is used to decide for the decision
    precision.
    """
    _check_for_state_or_transfer(G)

    if isinstance(G, State):
        if G._isgain:
            return State(G.to_array)
        else:
            A, B, C, D = G.matrices
            Am, Bm, Cm = _minimal_realization_state(A, B, C, tol=tol)
            if Am.size > 0:
                return State(Am, Bm, Cm, D, dt=G.SamplingPeriod)
            else:
                return State(D, dt=G.SamplingPeriod)
    else:
        if G._isgain:
            return Transfer(G.to_array)
        else:
            num, den = G.polynomials
            numm, denm = _minimal_realization_transfer(num, den, tol=tol)
            return Transfer(numm, denm, dt=G.SamplingPeriod)


def _minimal_realization_state(A, B, C, tol=1e-6):
    keep_looking = True
    run_out_of_states = False

    while keep_looking:
        n = A.shape[0]
        # Make sure that we still have states left
        if n == 0:
            A, B, C = [(np.empty((1, 0)))]*3
            break

        kc = cancellation_distance(A, B)[0]
        ko = cancellation_distance(A.T, C.T)[0]

        if min(kc, ko) > tol:  # no cancellation
            keep_looking = False
        else:

            Ac, Bc, Cc, blocks_c = staircase(A, B, C, block_indices=True)
            Ao, Bo, Co, blocks_o = staircase(A, B, C, form='o', invert=True,
                                             block_indices=True)

            # ===============Extra Check============================
            """
             Here kc,ko reports a possible cancellation so staircase
             should also report fewer than n, c'ble/o'ble blocks in the
             decomposition. If not, staircase tol should be increased.
             Otherwise either infinite loop or uno'ble branch removes
             the system matrices

             Thus, we remove the last scalar or the two-by-two block
             artificially. Because we trust the cancelling distance,
             more than our first born. The possible cases of unc'ble
             modes are

               -- one real distinct eigenvalue
               -- two real identical eigenvalues
               -- two complex conjugate eigenvalues

             We don't regret this. This is sparta.
            """

            # If unobservability distance is closer, let it handle first
            if ko >= kc:
                if (sum(blocks_c) == n and kc <= tol):
                    Ac_mod, Bc_mod, Cc_mod, kc_mod = Ac, Bc, Cc, kc

                    while kc_mod <= tol:  # Until cancel dist gets big
                        Ac_mod, Bc_mod, Cc_mod = (Ac_mod[:-1, :-1],
                                                  Bc_mod[:-1, :],
                                                  Cc_mod[:, :-1])

                        if Ac_mod.size == 0:
                            A, B, C = [(np.empty((1, 0)))]*3
                            run_out_of_states = True
                            break
                        else:
                            kc_mod = cancellation_distance(Ac_mod, Bc_mod)[0]

                    kc = kc_mod
                    # Fake an iterable to fool the sum below
                    blocks_c = [sum(blocks_c)-Ac_mod.shape[0]]

            # Same with the o'ble modes
            if (sum(blocks_o) == n and ko <= tol):
                Ao_mod, Bo_mod, Co_mod, ko_mod = Ao, Bo, Co, ko

                while ko_mod <= tol:  # Until cancel dist gets big
                    Ao_mod, Bo_mod, Co_mod = (Ao_mod[1:, 1:],
                                              Bo_mod[1:, :],
                                              Co_mod[:, 1:])

                    # If there is nothing left, break out everything
                    if Ao_mod.size == 0:
                        A, B, C = [(np.empty((1, 0)))]*3
                        run_out_of_states = True
                        break
                    else:
                        ko_mod = cancellation_distance(Ao_mod, Bo_mod)[0]

                ko = ko_mod
                blocks_o = [sum(blocks_o)-Ao_mod.shape[0]]

            # ===============End of Extra Check=====================

            if run_out_of_states:
                break

            if sum(blocks_c) > sum(blocks_o):
                remove_from = 'o'
            elif sum(blocks_c) < sum(blocks_o):
                remove_from = 'c'
            else:  # both have the same number of states to be removed
                if kc >= ko:
                    remove_from = 'o'
                else:
                    remove_from = 'c'

            if remove_from == 'c':
                l = int(sum(blocks_c))
                A, B, C = Ac[:l, :l], Bc[:l, :], Cc[:, :l]
            else:
                l = n - int(sum(blocks_o))
                A, B, C = Ao[l:, l:], Bo[l:, :], Co[:, l:]

    return A, B, C


def _minimal_realization_transfer(num, den, tol=1e-6):
    '''
    A helper function for obtaining a minimal representation of the
    Transfer() models.
    The method is pretty straightforward; going over the pole/zero pairs
    and removing them if they are either exactly the same or within their
    neigbourhood in 2-norm sense with threshold `tol`.
    '''
    # MIMO or not?
    if isinstance(num, list):
        # Don't touch the original data
        num = deepcopy(num)
        den = deepcopy(den)

        # Walk over entries for pole/zero cancellations
        m, p = len(num[0]), len(num)
        for row in range(p):
            for col in range(m):
                (num[row][col],
                 den[row][col]) = _minimal_realization_simplify(num[row][col],
                                                                den[row][col],
                                                                tol)
    # It's SISO search directly
    else:
        num, den = _minimal_realization_simplify(num, den, tol)

    return num, den


def _minimal_realization_simplify(num, den, tol):
    '''
    This is a simple distance checker between the each root of num and all
    roots of den to see whether there are any pairs that are sufficiently
    close to each other defined by `tol`.
    '''
    # Early exit if numerator is a scalar
    if num.size == 1:
        m = den[0, 0]
        return num/m, den/m

    # Get the gain from leading coefficients to work with monic polynomials
    k_gain = num[0, 0]/den[0, 0]
    plz = np.roots(den[0])
    zrz = np.roots(num[0])

    # Root finding algorithms are inherently ill-conditioned. Hence it might
    # happen that real multiplicities can turn out as complex pairs, e.g.,
    # np.roots(np.poly([1,2,3,2,3,4]))

    # This is a simple walk over zeros checking if there is something close
    # enough to it in the pole list with the slight extra check:
    # If we encounter 3.0 zero vs. 3.0+1e-9i pole, we look for another real
    # 3.0 in the zeros and for the conjugate of the pole and vice versa.

    # Zeros (both reals and one element of each complex pairs)
    zrz = np.r_[zrz[np.imag(zrz) == 0.], zrz[np.imag(zrz) > 0.]]

    safe_z = []

    for z in zrz:
        dist = np.abs(plz-z)
        bool_cz = np.imag(z) > 0
        # Do we have a match ?
        if np.min(dist) < tol + tol*np.abs(z):
            # Get the index and check the complex part
            match_index = np.argmin(dist)
            pz = plz[match_index]
            bool_cp = np.imag(pz) > 0

            if bool_cz and bool_cp:
                plz = np.delete(plz, match_index)
                # remove also the conjugate
                del_index, = np.where(plz == np.conj(pz))
                plz = np.delete(plz, del_index[0])

            elif bool_cz and not bool_cp:
                # We have a complex pair of zeros and a real pole
                # If there is another entry of this pole then we
                # cancel both of them otherwise we assume a real/real
                # cancellation and convert the other complex zero to real.

                # First get rid of the real pole
                plz = np.delete(plz, match_index)
                # Now search another real pole that is also close
                dist = np.abs(plz-z)
                if np.min(dist) < tol + tol*np.abs(z):
                    match_index = np.argmin(dist)
                    plz = np.delete(plz, match_index)
                else:
                    # It was a real/complex cancellation, make the zero real
                    safe_z += np.real(z)

            elif not bool_cz and bool_cp:
                # Same with above but this time we convert the other pole
                # to catch in the next iteration if another zero exists
                plz = np.delete(plz, match_index)
                conj_index, = np.where(plz == np.conj(pz))
                plz[conj_index[0]] = np.real(plz[conj_index[0]])

            else:
                plz = np.delete(plz, match_index)

        else:
            safe_z += [z]

            if bool_cz:
                safe_z += [np.conj(z)]

    return np.atleast_2d(k_gain*np.poly(safe_z)), np.atleast_2d(np.poly(plz))
