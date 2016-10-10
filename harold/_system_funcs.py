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
from numpy.linalg import cond, eig, norm
from scipy.linalg import svdvals, qr

from ._aux_linalg import haroldsvd, matrix_slice, e_i
#from ._classes import Transfer, State

"""
TODO Though the descriptor code also works up-to-production, I truncated
to explicit systems. I better ask around if anybody needs them (though
the answer to such question is always a yes).
"""

__all__ = ['system_norm', 'concatenate_state_matrices', 'staircase',
           'cancellation_distance', 'minimal_realization']


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

    if not isinstance(p,(int,float)):
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
            a , b , c = now_state.matrices[:3]
            x = lyapunov_eq_solver( a , b.dot(b.T) )
            return np.sqrt(np.trace(c.dot(x.dot(c.T))))
        else:
            a , b , c , d = now_state.matrices
            x = lyapunov_eq_solver( a , b.dot(b.T) , form = 'd' )
            return np.sqrt(np.trace(c.dot(x.dot(c.T))+d.dot(d.T)))


    elif np.isinf(p):
        if not now_state._isstable:
            return np.Inf,None

        a , b , c , d = now_state.matrices
#        n_s , (n_in , n_out) = now_state.NumberOfStates, now_state.shape

        # Initial gamma0 guess
        # Get the max of the largest svd of either
        #   - feedthrough matrix
        #   - G(iw) response at the pole with smallest damping
        #   - G(iw) at w = 0

        # We only need the svd vals hence call numpy svd
        lb1 = np.max(np.linalg.svd(d,compute_uv=False))

        # Formula (4.3) given in Bruinsma, Steinbuch Sys.Cont.Let. (1990)
        if any(np.abs(np.imag(now_state.poles)>1e-5)):
            low_damp_freq = np.abs(now_state.poles[np.argmax(
            [np.abs(np.imag(x)/np.real(x)/np.abs(x)) for x in now_state.poles]
            )])
        else:
            low_damp_freq = np.min(np.abs(now_state.poles))


        f , w = frequency_response(now_state,custom_grid=[0,low_damp_freq])
        if now_state._isSISO:
            lb2 = np.max(np.abs(f))
        else:
            lb2 = [np.linalg.svd(f[:,:,x],compute_uv=0) for x in range(2)]
            lb2 = np.max(lb2)

        # Finally
        gamma_lb = np.max([lb1,lb2])

        # Constant part of the Hamiltonian to shave off a few flops
        H_of_gam_const = np.c_[np.r_[a,np.zeros_like(a)],np.r_[c.T.dot(c),a.T]]
        H_of_gam_lfact = np.r_[b,c.T.dot(d)]
        H_of_gam_rfact = np.c_[-d.T.dot(c),b.T]

        # Start a for loop with a definite end !
        for x in range(max_iter_limit):

            # (Step b1)
            test_gamma = gamma_lb * ( 1 + 2*hinf_tolerance)

            # (Step b2)
            R_of_gam = d.T.dot(d) - test_gamma**2 * np.eye(d.shape[1])
            # TODO : It might be good to implement the result of Benner et al.
            # for the Hamiltonian later
            try:
                eigs_of_H = sp.linalg.eigvals(
                                H_of_gam_const - H_of_gam_lfact.dot(
                                    sp.linalg.solve(R_of_gam,H_of_gam_rfact)
                                )
                            )
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
                        mix_imag_eigs[np.abs(np.imag(eigs_of_H))>eig_tolerance]
                            ),decimals=7))

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


#TODO : type checking for both.

def concatenate_state_matrices(G):
    """
    Takes a State() model as input and returns the matrix

    .. math::

        \\left[\\begin{array}{c|c}A&B\\\\ \\hline C&D\\end{array}\\right]

    Parameters
    ----------

    G : State()

    Returns
    -------

    M : 2D Numpy array

    """
    if not isinstance(G, State):
        raise TypeError('concatenate_state_matrices() works on state '
                        'representations, but I found \"{0}\" object '
                        'instead.'.format(type(G).__name__))
    H = np.vstack((np.hstack((G.a, G.b)), np.hstack((G.c, G.d))))
    return H




def staircase(A,B,C,compute_T=False,form='c',invert=False,block_indices=False):
    """
    The staircase form is used very often to assess system properties.
    Given a state system matrix triplet A,B,C, this function computes
    the so-called controller/observer-Hessenberg form such that the resulting
    system matrices have the block-form (x denoting the nonzero blocks)

    .. math::

        \\begin{array}{c|c}
            \\begin{bmatrix}
                \\times & \\times & \\times & \\times & \\times \\\\
                \\times & \\times & \\times & \\times & \\times \\\\
                0       & \\times & \\times & \\times & \\times \\\\
                0       & 0       & \\times & \\times & \\times \\\\
                0       & 0       &  0      & \\times & \\times
            \\end{bmatrix} &
            \\begin{bmatrix}
                \\times \\\\
                0       \\\\
                0       \\\\
                0       \\\\
                0
            \\end{bmatrix} \\\\ \\hline
            \\begin{bmatrix}
                \\times & \\times & \\times & \\times & \\times \\\\
                \\times & \\times & \\times & \\times & \\times
            \\end{bmatrix}
        \\end{array}


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
    back the result. This is just silly to ask the user to do that. Hence
    the additional ``form`` option denoting whether it is the observer or
    the controller form that is requested.


    Parameters
    ----------

    A,B,C : {(n,n),(n,m),(p,n)} array_like
        System Matrices to be converted
    compute_T : bool, optional
        Whether the transformation matrix T should be computed or not
    form : { 'c' , 'o' }, optional
        Determines whether the controller- or observer-Hessenberg form
        will be computed.
    invert : bool, optional
        Whether to select which side the B or C matrix will be compressed.
        For example, the default case returns the B matrix with (if any)
        zero rows at the bottom. invert option flips this choice either in
        B or C matrices depending on the "form" switch.
    block_indices : bool, optional


    Returns
    -------

    Ah,Bh,Ch : {(n,n),(n,m),(p,n)} 2D numpy arrays
        Converted system matrices
    T : (n,n) 2D numpy array
        If the boolean ``compute_T`` is true, returns the transformation
        matrix such that

        .. math::

            \\left[\\begin{array}{c|c}
                T^{-1}AT &T^{-1}B \\\\ \\hline
                CT & D
            \\end{array}\\right]

        is in the desired staircase form.
    k: Numpy array
        If the boolean ``block_indices`` is true, returns the array
        of controllable/observable block sizes identified during block
        diagonalization

    """


    if not form in {'c','o'}:
        raise ValueError('The "form" key can only take values'
                         '\"c\" or \"o\" denoting\ncontroller- or '
                         'observer-Hessenberg form.')
    if form == 'o':
        A , B , C = A.T , C.T , B.T

    n = A.shape[0]
    ub , sb , vb , m0 = haroldsvd(B,also_rank=True)
    cble_block_indices = np.empty((1,0))

    # Trivially  Uncontrollable Case
    # Skip the first branch of the loop by making m0 greater than n
    # such that the matrices are returned as is without any computation
    if m0 == 0:
        m0 = n + 1
        cble_block_indices = np.array([0])

    # After these, start the regular case
    if n > m0:# If it is not a square system with full rank B

        A0 = ub.T.dot(A.dot(ub))
#        print(A0)

        # Row compress B and consistent zero blocks with the reported rank
        B0 = sb.dot(vb)
        B0[m0:,:] = 0.
        C0 = C.dot(ub)
        cble_block_indices = np.append(cble_block_indices,m0)

        if compute_T:
            P = sp.linalg.block_diag(np.eye(n-ub.T.shape[0]),ub.T)

        # Since we deal with submatrices, we need to increase the
        # default tolerance to reasonably high values that are
        # related to the original data to get exact zeros
        tol_from_A = n*sp.linalg.norm(A,1)*np.finfo(float).eps

        # Region of interest
        m = m0
        ROI_start = 0
        ROI_size = 0

        for dummy_row_counter in range(A.shape[0]):
            ROI_start += ROI_size
            ROI_size = m
#            print(ROI_start,ROI_size)
            h1,h2,h3,h4 = matrix_slice(
                                A0[ROI_start:,ROI_start:],
                                (ROI_size,ROI_size)
                                )
            uh3,sh3,vh3,m = haroldsvd(h3,also_rank=True,rank_tol = tol_from_A)

            # Make sure reported rank and sh3 are consistent about zeros
            sh3[ sh3 < tol_from_A ] = 0.

            # If the resulting subblock is not full row or zero rank
            if 0 < m < h3.shape[0]:
                cble_block_indices = np.append(cble_block_indices,m)
                if compute_T:
                    P = sp.linalg.block_diag(np.eye(n-uh3.shape[1]),uh3.T).dot(P)
                A0[ROI_start:,ROI_start:] = np.r_[
                                    np.c_[h1,h2],
                                    np.c_[sh3.dot(vh3),uh3.T.dot(h4)]
                                    ]
                A0 = A0.dot(sp.linalg.block_diag(np.eye(n-uh3.shape[1]),uh3))
                C0 = C0.dot(sp.linalg.block_diag(np.eye(n-uh3.shape[1]),uh3))
                # Clean up
                A0[abs(A0) < tol_from_A ] = 0.
                C0[abs(C0) < tol_from_A ] = 0.
            elif m == h3.shape[0]:
                cble_block_indices = np.append(cble_block_indices,m)
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
            A0 , B0 , C0 = A0.T , C0.T , B0.T

        if compute_T:
            if block_indices:
                return A0 , B0 , C0 , P.T , cble_block_indices
            else:
                return A0 , B0 , C0 , P.T
        else:
            if block_indices:
                return A0 , B0 , C0 , cble_block_indices
            else:
                return A0 , B0 , C0

    else: # Square system B full rank ==> trivially controllable
        cble_block_indices = np.array([n])
        if form == 'o':
            A , B , C = A.T , C.T , B.T

        if compute_T:
            if block_indices:
                return A,B,C,np.eye(n),cble_block_indices
            else:
                return A , B , C , np.eye(n)
        else:
            if block_indices:
                return A , B , C , cble_block_indices
            else:
                return A , B , C

def cancellation_distance(F,G):
    """
    Given matrices :math:`F,G`, computes the upper and lower bounds of
    the perturbation needed to render the pencil :math:`\\left[
    \\begin{array}{c|c}F-pI & G\\end{array}\\right]` rank deficient. It is
    used for assessing the controllability/observability degenerate distance
    and hence for minimality assessment.

    Implements the algorithm given in D.Boley SIMAX vol.11(4) 1990.

    Parameters
    ----------

    F,G : 2D arrays
        Pencil matrices to be checked for rank deficiency distance

    Returns
    -------

    upper2 : float
        Upper bound on the norm of the perturbation
        :math:`\\left[\\begin{array}{c|c}dF & dG\\end{array}\\right]` such
        that :math:`\\left[\\begin{array}{c|c}F+dF-pI & G+dG \\end{array}
        \\right]` is rank deficient.
    upper1 : float
        A theoretically softer upper bound than the upper2 for the
        same quantity.
    lower0 : float
        Lower bound on the same quantity given in upper2
    e_f    : complex
        Indicates the eigenvalue that renders [F + dF - pI | G + dG ]
        rank deficient i.e. equals to the p value at the closest rank
        deficiency.
    radius : float
        The perturbation with the norm bound "upper2" is located within
        a disk in the complex plane whose center is on "e_f" and whose
        radius is bounded by this output.

    """
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


def minimal_realization(A, B, C, mu_tol=1e-9):
    """
    Given state matrices :math:`A,B,C` computes minimal state matrices
    such that the system is controllable and observable within the
    given tolerance :math:`\\mu`.

    Implements a basic two pass algorithm :
     1- First distance to mode cancellation is computed then also
     the Hessenberg form is obtained with the identified o'ble/c'ble
     block numbers. If staircase form reports that there are no
     cancellations but the distance is less than the tolerance,
     distance wins and the respective mode is removed.

    Uses ``cancellation_distance()`` and ``staircase()`` for the tests.

    Parameters
    ----------

    A,B,C : {(n,n), (n,m), (pxn)} array_like
        System matrices to be checked for minimality
    mu_tol: float
        The sensitivity threshold for the cancellation to be compared
        with the first default output of cancellation_distance() function. The
        default value is (default value is :math:`10^{-9}`)

    Returns
    -------

    A,B,C : {(k,k), (k,m), (pxk)} array_like
        System matrices that are identified as minimal with k states
        instead of the original n where (k <= n)

    """

    keep_looking = True
    run_out_of_states = False

    while keep_looking:
        n = A.shape[0]
        # Make sure that we still have states left
        if n == 0:
            A , B , C = [(np.empty((1,0)))]*3
            break

        kc = cancellation_distance(A,B)[0]
        ko = cancellation_distance(A.T,C.T)[0]

        if min(kc,ko) > mu_tol: # no cancellation
            keep_looking= False
        else:

            Ac,Bc,Cc,blocks_c = staircase(A,B,C,block_indices=True)
            Ao,Bo,Co,blocks_o = staircase(A,B,C,form='o',invert=True,
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
            if ko>=kc:
                if (sum(blocks_c) == n and kc <= mu_tol):
                    Ac_mod , Bc_mod , Cc_mod , kc_mod = Ac,Bc,Cc,kc

                    while kc_mod <= mu_tol:# Until cancel dist gets big
                        Ac_mod,Bc_mod,Cc_mod = (
                                Ac_mod[:-1,:-1],Bc_mod[:-1,:],Cc_mod[:,:-1])

                        if Ac_mod.size == 0:
                            A , B , C = [(np.empty((1,0)))]*3
                            run_out_of_states = True
                            break
                        else:
                            kc_mod = cancellation_distance(Ac_mod,Bc_mod)[0]

                    kc = kc_mod
                    # Fake an iterable to fool the sum below
                    blocks_c = [sum(blocks_c)-Ac_mod.shape[0]]


            # Same with the o'ble modes
            if (sum(blocks_o) == n and ko <= mu_tol):
                Ao_mod , Bo_mod , Co_mod , ko_mod = Ao,Bo,Co,ko

                while ko_mod <= mu_tol:# Until cancel dist gets big
                    Ao_mod,Bo_mod,Co_mod = (
                            Ao_mod[1:,1:],Bo_mod[1:,:],Co_mod[:,1:])

                    # If there is nothing left, break out everything
                    if Ao_mod.size == 0:
                        A , B , C = [(np.empty((1,0)))]*3
                        run_out_of_states = True
                        break
                    else:
                        ko_mod = cancellation_distance(Ao_mod,Bo_mod)[0]


                ko = ko_mod
                blocks_o = [sum(blocks_o)-Ao_mod.shape[0]]

            # ===============End of Extra Check=====================

            if run_out_of_states: break

            if sum(blocks_c) > sum(blocks_o):
                remove_from = 'o'
            elif sum(blocks_c) < sum(blocks_o):
                remove_from = 'c'
            else: # both have the same number of states to be removed
                if kc >= ko:
                    remove_from = 'o'
                else:
                    remove_from = 'c'

            if remove_from == 'c':
                l = sum(blocks_c)
                A , B , C = Ac[:l,:l] , Bc[:l,:] , Cc[:,:l]
            else:
                l = n - sum(blocks_o)
                A , B , C = Ao[l:,l:] , Bo[l:,:] , Co[:,l:]

    return A , B, C