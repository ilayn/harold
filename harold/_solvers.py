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
from numpy.linalg._umath_linalg import solve

from scipy.linalg import qz, schur

__all__ = ['lyapunov_eq_solver']


def lyapunov_eq_solver(A, Y, E=None, form='c'):
    '''
    This function solves the Lyapunov and the generalized Lyapunov
    equations of the forms

    (1)                X A + A^T X + Y = 0

    (1')               A^T X A - X + Y = 0

    and

    (2)             E^T X A + A^T X E + Y = 0

    (2')            A^T X A - E^T X E + Y = 0

    for the unknown matrix `X` given square matrices A, E, and
    symmetric Y with compatible sizes. The numbered (primed) equations
    are the so-called continuous (discrete) time forms. The `form`
    keyword selects between the two.

    For (1), (1'), the `A` matrix is brought to real Schur form
    and for (2), (2') QZ decomposition is used. Then all have a similar
    forward substitution step. The method is a a modified implementation
    of T. Penzl (1998) which is essentially a modification of Bartels -
    Stewart method.

    If the argument `E` is not exactly a `None`-type then (2) is
    assumed.

    Parameters
    ----------
    A , Y , E: nxn array_like
        Data matrices for the equation. Y is a symmetric matrix.

    form : 'c' , 'continuous' , 'd' , 'discrete'
        The string selector to define which form of Lyapunov equation is
        going to be used.

    Returns
    -------

    X : nxn numpy array
        Solution to the selected Lyapunov equation.

    '''
    def check_matrices(a, y, e):
        arg_names = ['A', 'Y', 'E']
        a = np.atleast_2d(a)
        y = np.atleast_2d(y)
        if e is not None:
            e = np.atleast_2d(e)
            if e.shape[0] != e.shape[1]:
                raise ValueError('E matrix is not square. Its shape is'
                                 '{}'.format(e.shape))

        for ind, mat in enumerate((a, y)):
            if mat.shape[0] != mat.shape[1]:
                raise ValueError('The argument {} must be square. '
                                 'Its shape is {}'
                                 ''.format(arg_names[ind], mat.shape))

        # shapes are square now check compatibility
        if a.shape != y.shape or (e is not None and a.shape != e.shape):
            raise ValueError('The sizes of the arguments are not compatible. '
                             'For convenience I have received A , Y , E '
                             'matrices shaped as {}'
                             ''.format([a.shape, y.shape,
                                        e if e is None else e.shape]))
        return a, y, e

    if form not in ('c', 'continuous', 'd', 'discrete'):
        raise ValueError('The keyword "form" accepts only the following'
                         'choices:\n\'c\',\'continuous\',\'d\',\'discrete\'')

    A, Y, E = check_matrices(A, Y, E)

    if form in ('c', 'continuous'):
        if E is None:
            X_sol = _solve_continuous_lyapunov(A, Y)
        else:
            X_sol = _solve_continuous_generalized_lyapunov(A, E, Y)
    else:
        if E is None:
            X_sol = _solve_discrete_lyapunov(A, Y)
        else:
            X_sol = _solve_discrete_generalized_lyapunov(A, E, Y)

    return X_sol


def _solve_continuous_generalized_lyapunov(A, E, Y, tol=1e-12):
    '''
    Solves

                A.T X E + E.T X A + Y = 0

    for symmetric Y
    '''
    mat33 = np.zeros((3, 3), dtype=float)
    mat44 = np.zeros((4, 4), dtype=float)

    # =============================
    # Declare the inner functions
    # =============================

    def mini_sylvester(R, S, Yt, U=None, V=None):
        '''
        A helper function to solve the 1x1 or 2x2 Sylvester equations
        arising in the solution of the generalized continuous-time
        Lyapunov equations

        Note that, this doesn't have any protection against LinAlgError
        hence the caller needs to `try` to see whether it is properly
        executed.
        '''
        if U is None:
            if R.size == 1:
                return -Yt / (2 * R * S)
            else:
                a, b, c, d = R.ravel().tolist()
                e, f, g, h = S.ravel().tolist()

                mat33[0, :] = [2*a*e, 2*(a*g + e*c), 2*c*g]
                mat33[1, :] = [a*f + e*b, a*h + e*d + c*f + b*g, c*h + g*d]
                mat33[2, :] = [2*b*f, 2*(b*h + f*d), 2*d*h]

                a, b, c = solve(mat33, -Yt.reshape(-1, 1)[[0, 1, 3], :]
                                ).ravel().tolist()

                return np.array([[a, b], [b, c]], dtype=float)

        elif R.size == 4:
            if S.size == 4:
                a, b, c, d = R.reshape(1, 4).tolist()[0]
                e, f, g, h = S.reshape(1, 4).tolist()[0]
                k, l, m, n = U.reshape(1, 4).tolist()[0]
                p, q, r, s = V.reshape(1, 4).tolist()[0]

                mat44[0, :] = [a*e + k*p, a*g + k*r, c*e + m*p, c*g + m*r]
                mat44[1, :] = [a*f + k*q, a*h + k*s, c*f + m*q, c*h + m*s]
                mat44[2, :] = [b*e + l*p, b*g + l*r, d*e + n*p, d*g + n*r]
                mat44[3, :] = [b*f + l*q, b*h + l*s, d*f + n*q, d*h + n*s]

                return solve(mat44, -Yt.reshape(-1, 1)).reshape(2, 2)
            else:
                return solve(S[0, 0]*R.T + V[0, 0]*U.T, -Yt)
        elif S.size == 4:
            return solve(R[0, 0]*S.T+U[0, 0]*V.T, -Yt.T).T
        else:
            return -Yt / (R * S + U * V)

    # =============================
    # Prepare the data
    # =============================

    # if the problem is small then solve directly
    if A.shape[0] < 3:
        return mini_sylvester(A, E, Y)

    # If there are nontrivial entries on the subdiagonal, we have a 2x2 block.
    # Based on that we have the block sizes `bz` and starting positions `bs`.
    As, Es, Q, Z = qz(A, E)
    Ys = Z.T @ Y @ Z
    n = A.shape[0]
    subdiag_entries = np.abs(As[range(1, n), range(0, n-1)]) > tol
    subdiag_indices = [ind for ind, x in enumerate(subdiag_entries) if x]
    bz = np.ones(n)
    for x in subdiag_indices:
        bz[x] = 2
        bz[x+1] = np.nan

    bz = bz[~np.isnan(bz)].astype(int)
    bs = [0] + np.cumsum(bz[:-1]).tolist() + [None]
    total_blk = bz.size
    Xs = np.empty_like(Y)

    # =============================
    #  Main Loop
    # =============================

    # Now we know how the matrices should be partitioned. We then start
    # from the uppper left corner and alternate between updating the
    # Y term and solving the next entry of X. We walk over X row-wise

    for row in range(total_blk):
        thisr = bs[row]
        nextr = bs[row+1]
        # This block is executed at the second and further spins of the
        # for loop. Humans should start reading from (**)
        if row != 0:
            Ys[thisr:nextr, thisr:nextr] +=  \
                            As[thisr:nextr, thisr:nextr].T @ \
                            Xs[thisr:nextr, :thisr] @ \
                            Es[:thisr, thisr:nextr] + \
                                                                \
                            Es[thisr:nextr, thisr:nextr].T @ \
                            Xs[thisr:nextr, :thisr] @ \
                            As[:thisr, thisr:nextr]

        # (**) Solve for the diagonal via Akk , Ekk , Ykk and place it in Xkk
        tempx = mini_sylvester(As[thisr:nextr, thisr:nextr],
                               Es[thisr:nextr, thisr:nextr],
                               Ys[thisr:nextr, thisr:nextr])

        # Place it in the data
        Xs[thisr:nextr, thisr:nextr] = tempx

        # Form the common products of X * E and X * A
        tempx = Xs[thisr:nextr, :nextr]
        XE_of_row = tempx @ Es[:nextr, thisr:]
        XA_of_row = tempx @ As[:nextr, thisr:]

        # Update Y terms right of the diagonal
        Ys[thisr:nextr, thisr:] += \
            As[thisr:nextr, thisr:nextr].T @ XE_of_row + \
            Es[thisr:nextr, thisr:nextr].T @ XA_of_row

        # Walk over upper triangular terms
        for col in range(row + 1, total_blk):
            thisc = bs[col]
            nextc = bs[col+1]
            # The corresponding Y term has already been updated, solve for X
            tempx = mini_sylvester(As[thisr:nextr, thisr:nextr],
                                   Es[thisc:nextc, thisc:nextc],
                                   Ys[thisr:nextr, thisc:nextc],
                                   Es[thisr:nextr, thisr:nextr],
                                   As[thisc:nextc, thisc:nextc])

            # Place it in the data
            Xs[thisr:nextr, thisc:nextc] = tempx
            Xs[thisc:nextc, thisr:nextr] = tempx.T

            # Post column solution Y update
            # XA and XE terms
            tempe = tempx @ Es[thisc:nextc, thisc:]
            tempa = tempx @ As[thisc:nextc, thisc:]

            # Update Y towards left
            Ys[thisr:nextr, thisc:] += \
                As[thisr:nextr, thisr:nextr].T @ tempe + \
                Es[thisr:nextr, thisr:nextr].T @ tempa
            # Update Y downwards
            XE_of_row[:, (thisc - thisr):] += tempe
            XA_of_row[:, (thisc - thisr):] += tempa

            ugly_sl = slice(thisc - thisr,
                            nextc - thisr if nextc is not None else None)

            Ys[nextr:nextc, thisc:nextc] += \
                As[thisr:nextr, nextr:nextc].T @ XE_of_row[:, ugly_sl] + \
                Es[thisr:nextr, nextr:nextc].T @ XA_of_row[:, ugly_sl]

    return Q @ Xs @ Q.T


def _solve_discrete_generalized_lyapunov(A, E, Y, tol=1e-12):
    '''
    Solves

                A.T X A - E.T X E + Y = 0

    for symmetric Y
    '''
    mat33 = np.zeros((3, 3), dtype=float)
    mat44 = np.zeros((4, 4), dtype=float)

    def mini_sylvester(S, V, Yt, R=None, U=None):
        '''
        A helper function to solve the 1x1 or 2x2 Sylvester equations
        arising in the solution of the generalized continuous-time
        Lyapunov equations

        Note that, this doesn't have any protection against LinAlgError
        hence the caller needs to `try` to see whether it is properly
        executed.
        '''
        if R is None:
            if S.size == 1:
                return -Yt / (S ** 2 - V ** 2)
            else:
                a, b, c, d = S.ravel().tolist()
                e, f, g, h = V.ravel().tolist()

                mat33[0, :] = [a*a - e*e, 2 * (a*c - e*g), c*c - g*g]
                mat33[1, :] = [a*b - e*f, a*d - e*h + c*b - g*f, c*d - g*h]
                mat33[2, :] = [b*b - f*f, 2 * (b*d - f*h), d*d - h*h]

                a, b, c = solve(mat33, -Yt.reshape(-1, 1)[[0, 1, 3], :]
                                ).ravel().tolist()

                return np.array([[a, b], [b, c]], dtype=float)

        elif S.size == 4:
            if R.size == 4:
                a, b, c, d = R.ravel().tolist()
                e, f, g, h = S.ravel().tolist()
                k, l, m, n = U.ravel().tolist()
                p, q, r, s = V.ravel().tolist()

                mat44[0, :] = [a*e - k*p, a*g - k*r, c*e - m*p, c*g - m*r]
                mat44[1, :] = [a*f - k*q, a*h - k*s, c*f - m*q, c*h - m*s]
                mat44[2, :] = [b*e - l*p, b*g - l*r, d*e - n*p, d*g - n*r]
                mat44[3, :] = [b*f - l*q, b*h - l*s, d*f - n*q, d*h - n*s]

                return solve(mat44, -Yt.reshape(-1, 1)).reshape(2, 2)

            else:
                return solve(R[0, 0]*S.T - U[0, 0]*V.T, -Yt.T).T
        elif R.size == 4:
            return solve(S[0, 0]*R.T - V[0, 0]*U.T, -Yt)
        else:
            return -Yt / (R * S - U * V)

    # =============================
    # Prepare the data
    # =============================
    # if the problem is small then solve directly
    if A.shape[0] < 3:
        return mini_sylvester(A, E, Y)

    As, Es, Q, Z = qz(A, E, overwrite_a=True, overwrite_b=True)
    Ys = Z.T @ Y @ Z
    n = As.shape[0]
    # If there are nontrivial entries on the subdiagonal, we have a 2x2 block.
    # Based on that we have the block sizes `bz` and starting positions `bs`.

    subdiag_entries = np.abs(As[range(1, n), range(0, n-1)]) > tol
    subdiag_indices = [ind for ind, x in enumerate(subdiag_entries) if x]
    bz = np.ones(n)
    for x in subdiag_indices:
        bz[x] = 2
        bz[x+1] = np.nan

    bz = bz[~np.isnan(bz)].astype(int)
    bs = [0] + np.cumsum(bz[:-1]).tolist() + [None]
    total_blk = bz.size
    Xs = np.empty_like(Y)

    # =============================
    #  Main Loop
    # =============================

    # Now we know how the matrices should be partitioned. We then start
    # from the uppper left corner and alternate between updating the
    # Y term and solving the next entry of X. We walk over X row-wise

    for row in range(total_blk):

        thisr = bs[row]
        nextr = bs[row+1]

        # This block is executed at the second and further spins of the
        # for loop. Humans should start reading from (**)
        if row != 0:
            Ys[thisr:nextr, thisr:nextr] +=  \
                As[thisr:nextr, thisr:nextr].T @ \
                Xs[thisr:nextr, :thisr] @ \
                As[:thisr, thisr:nextr] - \
                Es[thisr:nextr, thisr:nextr].T @ \
                Xs[thisr:nextr, :thisr] @ \
                Es[:thisr, thisr:nextr]

        # (**) Solve for the diagonal via Akk , Ekk , Ykk and place it in Xkk
        tempx = mini_sylvester(As[thisr:nextr, thisr:nextr],
                               Es[thisr:nextr, thisr:nextr],
                               Ys[thisr:nextr, thisr:nextr])

        # Place it in the data
        Xs[thisr:nextr, thisr:nextr] = tempx

        # Form the common products of X * E and X * A
        tempx = Xs[thisr:nextr, :nextr]
        XE_of_row = tempx @ Es[:nextr, thisr:]
        XA_of_row = tempx @ As[:nextr, thisr:]

        # Update Y terms right of the diagonal
        Ys[thisr:nextr, thisr:] += \
            As[thisr:nextr, thisr:nextr].T @ XA_of_row - \
            Es[thisr:nextr, thisr:nextr].T @ XE_of_row

        # Walk over upper triangular terms
        for col in range(row + 1, total_blk):

            thisc = bs[col]
            nextc = bs[col+1]

            # The corresponding Y term has already been updated, solve for X
            tempx = mini_sylvester(As[thisc:nextc, thisc:nextc],
                                   Es[thisc:nextc, thisc:nextc],
                                   Ys[thisr:nextr, thisc:nextc],
                                   As[thisr:nextr, thisr:nextr],
                                   Es[thisr:nextr, thisr:nextr])

            # Place it in the data
            Xs[thisr:nextr, thisc:nextc] = tempx
            Xs[thisc:nextc, thisr:nextr] = tempx.T

            # Post column solution Y update

            # XA and XE terms
            tempe = tempx @ Es[thisc:nextc, thisc:]
            tempa = tempx @ As[thisc:nextc, thisc:]
            # Update Y towards left
            Ys[thisr:nextr, thisc:] += \
                As[thisr:nextr, thisr:nextr].T @ tempa - \
                Es[thisr:nextr, thisr:nextr].T @ tempe
            # Update Y downwards
            XE_of_row[:, (thisc - thisr):] += tempe
            XA_of_row[:, (thisc - thisr):] += tempa

            ugly_sl = slice(thisc - thisr,
                            nextc - thisr if nextc is not None else None)

            Ys[nextr:nextc, thisc:nextc] += \
                As[thisr:nextr, nextr:nextc].T @ XA_of_row[:, ugly_sl] - \
                Es[thisr:nextr, nextr:nextc].T @ XE_of_row[:, ugly_sl]

    return Q @ Xs @ Q.T


def _solve_continuous_lyapunov(A, Y):
    '''
            Solves A.T X + X A + Y = 0

    '''
    mat33 = np.zeros((3, 3), dtype=float)
    mat44 = np.zeros((4, 4), dtype=float)
    i2 = np.eye(2, dtype=float)

    def mini_sylvester(Ar, Yt, Al=None):
        '''
        A helper function to solve the 1x1 or 2x2 Sylvester equations
        arising in the solution of the continuous-time Lyapunov equations

        Note that, this doesn't have any protection against LinAlgError
        hence the caller needs to `try` to see whether it is properly
        executed.
        '''

        # The symmetric problem
        if Al is None:
            if Ar.size == 1:
                return - Yt / (Ar * 2)
            else:
                a, b, c, d = Ar.reshape(1, 4).tolist()[0]

                mat33[0, :] = [2*a, 2*c, 0]
                mat33[1, :] = [b, a + d, c]
                mat33[2, :] = [0, 2*b, 2*d]
                a, b, c = solve(mat33, -Yt.reshape(-1, 1)[[0, 1, 3], :]
                                ).ravel().tolist()

                return np.array([[a, b], [b, c]], dtype=float)

        # Nonsymmetric
        elif Ar.size == 4:
            if Al.size == 4:
                a00, a01, a10, a11 = Al.reshape(1, 4).tolist()[0]
                b00, b01, b10, b11 = Ar.reshape(1, 4).tolist()[0]

                mat44[0, :] = [a00+b00, b10, a10, 0]
                mat44[1, :] = [b01, a00 + b11, 0, a10]
                mat44[2, :] = [a01, 0, a11 + b00, b10]
                mat44[3, :] = [0, a01, b01, a11 + b11]

                return solve(mat44, -Yt.reshape(-1, 1)).reshape(2, 2)
            # Ar is 2x2 , Al is scalar
            else:
                return solve(Ar.T + Al[0, 0] * i2, -Yt.T).T

        elif Al.size == 4:
            return solve(Al.T + Ar[0, 0] * i2, -Yt)
        else:
            return -Yt / (Ar + Al)

    # =============================
    # Prepare the data
    # =============================
    # if the problem is small then solve directly
    if A.shape[0] < 3:
        return mini_sylvester(A, Y)

    As, S = schur(A, output='real')
    Ys = S.T @ Y @ S
    n = As.shape[0]

    # If there are nontrivial entries on the subdiagonal, we have a 2x2 block.
    # Based on that we have the block sizes `bz` and starting positions `bs`.

    subdiag_entries = np.abs(As[range(1, n), range(0, n-1)]) > 0
    subdiag_indices = [ind for ind, x in enumerate(subdiag_entries) if x]
    bz = np.ones(n)
    for x in subdiag_indices:
        bz[x] = 2
        bz[x+1] = np.nan

    bz = bz[~np.isnan(bz)].astype(int)
    bs = [0] + np.cumsum(bz[:-1]).tolist() + [None]
    total_blk = bz.size
    Xs = np.empty_like(Y)

    # =============================
    #  Main Loop
    # =============================

    # Now we know how the matrices should be partitioned. We then start
    # from the uppper left corner and alternate between updating the
    # Y term and solving the next entry of X. We walk over X row-wise
    for row in range(total_blk):
        thisr = bs[row]
        nextr = bs[row+1]

        # This block is executed at the second and further spins of the
        # for loop. Humans should start reading from (**)
        if row != 0:
            Ys[thisr:nextr, thisr:] +=  \
                      Xs[thisr:nextr, 0:thisr] @ As[0:thisr, thisr:]

        # (**) Solve for the diagonal via Akk , Ykk and place it in Xkk
        tempx = mini_sylvester(As[thisr:nextr, thisr:nextr],
                               Ys[thisr:nextr, thisr:nextr])

#        X_placer( tempx , row , row )
        Xs[thisr:nextr, thisr:nextr] = tempx
        # Update Y terms right of the diagonal
        Ys[thisr:nextr, nextr:] += tempx @ As[thisr:nextr, nextr:]

        # Walk over upper triangular terms
        for col in range(row + 1, total_blk):
            thisc = bs[col]
            nextc = bs[col+1]

            # The corresponding Y term has already been updated, solve for X
            tempx = mini_sylvester(As[thisc:nextc, thisc:nextc],
                                   Ys[thisr:nextr, thisc:nextc],
                                   As[thisr:nextr, thisr:nextr])

            # Place it in the data
            Xs[thisr:nextr, thisc:nextc] = tempx
            Xs[thisc:nextc, thisr:nextr] = tempx.T

            # Update Y towards left
            Ys[thisr:nextr, nextc:] += tempx @ As[thisc:nextc, nextc:]

            # Update Y downwards
            Ys[nextr:nextc, thisc:nextc] += \
                As[thisr:nextr, nextr:nextc].T @ tempx

    return S @ Xs @ S.T


def _solve_discrete_lyapunov(A, Y):
    '''
                 Solves     A.T X A - X + Y = 0
    '''
    mat33 = np.zeros((3, 3), dtype=float)
    mat44 = np.zeros((4, 4), dtype=float)
    i2 = np.eye(2)

    def mini_sylvester(Al, Yt, Ar=None):
        '''
        A helper function to solve the 1x1 or 2x2 Sylvester equations
        arising in the solution of the continuous-time Lyapunov equations

        Note that, this doesn't have any protection against LinAlgError
        hence the caller needs to `try` to see whether it is properly
        executed.
        '''
        # The symmetric problem
        if Ar is None:
            if Al.size == 1:
                return - Yt / (Al ** 2 - 1)
            else:
                a, b, c, d = Al.reshape(1, 4).tolist()[0]

                mat33[0, :] = [a**2 - 1, 2*a*c, c ** 2]
                mat33[1, :] = [a*b, a*d + b*c - 1, c*d]
                mat33[2, :] = [b ** 2, 2*b*d, d ** 2 - 1]
                a, b, c = solve(mat33, -Yt.reshape(-1, 1)[[0, 1, 3], :]
                                ).ravel().tolist()

                return np.array([[a, b], [b, c]], dtype=float)

        # Nonsymmetric
        elif Al.size == 4:
            if Ar.size == 4:
                a00, a01, a10, a11 = Al.ravel().tolist()
                b00, b01, b10, b11 = Ar.ravel().tolist()

                mat44[0, :] = [a00*b00 - 1, a00*b10, a10*b00, a10*b10]
                mat44[1, :] = [a00*b01, a00*b11 - 1, a10*b01, a10*b11]
                mat44[2, :] = [a01*b00, a01*b10, a11*b00 - 1, a11*b10]
                mat44[3, :] = [a01*b01, a01*b11, a11*b01, a11*b11 - 1]

                return solve(mat44, -Yt.reshape(-1, 1)).reshape(2, 2)
            else:
                return solve(Al.T * Ar[0, 0] - i2, -Yt)

        elif Ar.size == 4:
            return solve(Ar.T * Al[0, 0] - i2, -Yt.T).T
        else:
            return -Yt / (Ar * Al - 1)

    # =====================================

    if A.shape[0] < 3:
        return mini_sylvester(A, Y)

    As, S = schur(A, output='real')
    Ys = S.T @ Y @ S

    # If there are nontrivial entries on the subdiagonal, we have a 2x2 block.
    # Based on that we have the block sizes `bz` and starting positions `bs`.
    n = As.shape[0]
    subdiag_entries = np.abs(As[range(1, n), range(0, n-1)]) > 0
    subdiag_indices = [ind for ind, x in enumerate(subdiag_entries) if x]
    bz = np.ones(n)
    for x in subdiag_indices:
        bz[x] = 2
        bz[x+1] = np.nan

    bz = bz[~np.isnan(bz)].astype(int)
    bs = [0] + np.cumsum(bz[:-1]).tolist() + [None]
    total_blk = bz.size
    Xs = np.empty_like(Y)

    # =============================
    #  Main Loop
    # =============================

    # Now we know how the matrices should be partitioned. We then start
    # from the uppper left corner and alternate between updating the
    # Y term and solving the next entry of X. We walk over X row-wise

    for row in range(total_blk):
        thisr = bs[row]
        nextr = bs[row+1]

        if row != 0:
            Ys[thisr:nextr, thisr:nextr] +=  \
                As[thisr:nextr, thisr:nextr].T @ \
                Xs[thisr:nextr, :thisr] @ \
                As[:thisr, thisr:nextr]

        # (**) Solve for the diagonal via Akk , Ykk and place it in Xkk
        tempx = mini_sylvester(As[thisr:nextr, thisr:nextr],
                               Ys[thisr:nextr, thisr:nextr])

        Xs[thisr:nextr, thisr:nextr] = tempx
        XA_of_row = Xs[thisr:nextr, :nextr] @ As[:nextr, thisr:]

        # Update Y terms right of the diagonal
        Ys[thisr:nextr, thisr:] += As[thisr:nextr, thisr:nextr].T @ XA_of_row

        # Walk over upper triangular terms
        for col in range(row + 1, total_blk):
            thisc = bs[col]
            nextc = bs[col+1]

            # The corresponding Y term has already been updated, solve for X
            tempx = mini_sylvester(As[thisr:nextr, thisr:nextr],
                                   Ys[thisr:nextr, thisc:nextc],
                                   As[thisc:nextc, thisc:nextc])

            # Place it in the data
            Xs[thisr:nextr, thisc:nextc] = tempx
            Xs[thisc:nextc, thisr:nextr] = tempx.T

            # Post column solution Y update
            # XA terms
            tempa = tempx @ As[thisc:nextc, thisc:]
            # Update Y towards left
            Ys[thisr:nextr, thisc:] += As[thisr:nextr, thisr:nextr].T @ tempa
            # Update Y downwards
            XA_of_row[:, thisc - thisr:] += tempa

            ugly_sl = slice(thisc - thisr,
                            nextc - thisr if nextc is not None else None)

            Ys[nextr:nextc, thisc:nextc] += \
                As[thisr:nextr, nextr:nextc].T @ XA_of_row[:, ugly_sl]

    return S @ Xs @ S.T
