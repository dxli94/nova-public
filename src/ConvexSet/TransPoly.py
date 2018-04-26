import cvxopt as cvx
import numpy as np

from ConvexSet.Polyhedron import Polyhedron


class TransPoly(Polyhedron):
    """
    Class to represent convex sets which is a linear transformation of another convex set only:
    V = B. U
    """

    def __init__(self, trans_matrix_B, coeff_matrix_U, col_vec_U=None):
        self.trans_matrix_B = trans_matrix_B
        self.trans_matrix_B_tp = np.transpose(self.trans_matrix_B)
        self.coeff_matrix = coeff_matrix_U
        self.cvx_coeff_matrix = cvx.matrix(coeff_matrix_U, tc='d')
        self.cvx_col_vec = cvx.matrix(col_vec_U, tc='d')

        if col_vec_U is not None:
            assert self.coeff_matrix.shape[0] == col_vec_U.shape[0], \
                "Shapes of coefficient matrix %r and column vector %r do not match!" \
                % (self.coeff_matrix.shape, col_vec_U.shape)
        else:
            col_vec_U = np.zeros(shape=(coeff_matrix_U.shape[0], 1))

        self.col_vec = col_vec_U

        if self.coeff_matrix.size > 0:
            self.isEmpty = False
            self.isUniverse = False
        else:
            self.isEmpty = False
            self.isUniverse = True

    def __str__(self):
        str_repr = 'H-representative\n' + \
                   'Ax <= b \n'

        for row in zip(self.coeff_matrix, self.col_vec):
            str_repr += ' '.join(str(item) for item in row) + '\n'

        return str_repr

    def compute_support_function(self, direction, lp):
        if self.is_empty():
            raise RuntimeError("\n Compute Support Function called for an Empty Set.")
        elif self.is_universe():
            raise RuntimeError("\n Cannot Compute Support Function of a Universe Polytope.\n")
        else:
            direction = np.dot(self.trans_matrix_B_tp, direction)
            c = cvx.matrix(-direction, tc='d')
            return lp.lp(c, self.cvx_coeff_matrix, self.cvx_col_vec)

