from scipy.optimize import linprog
from ConvexSet.Polyhedron import Polyhedron

import numpy as np
import cdd


class TransPoly(Polyhedron):
    """
    Class to represent convex sets which is a linear transformation of another convex set only:
    V = B. U
    """

    def __init__(self, trans_matrix_B, coeff_matrix, col_vec_U=None):
        self.trans_matrix_B = np.matrix(trans_matrix_B)
        self.coeff_matrix = np.matrix(coeff_matrix)

        if col_vec_U is not None:
            col_vec_U = np.matrix(col_vec_U)
            assert coeff_matrix.shape[0] == col_vec_U.shape[0], \
                "Shapes of coefficient matrix %r and column vector %r do not match!" \
                % (coeff_matrix.shape, col_vec_U.shape)
        else:
            col_vec_U = np.zeros(shape=(coeff_matrix.shape[0], 1))

        self.col_vec = col_vec_U

        if coeff_matrix.size > 0:
            self.mat_poly = cdd.Matrix(np.hstack((self.col_vec, -self.coeff_matrix)).tolist())
            self.mat_poly.rep_type = cdd.RepType.INEQUALITY

            self.poly = self._gen_poly()
            self.vertices = self._update_vertices()

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

    def compute_support_function(self, direction):
        if self.is_empty():
            raise RuntimeError("\n Compute Support Function called for an Empty Set.")
        elif self.is_universe():
            raise RuntimeError("\n Cannot Compute Support Function of a Universe Polytope.\n")
        else:
            direction = np.squeeze(np.asarray(-np.matmul(np.transpose(self.trans_matrix_B), direction)))

            sf = linprog(c=direction,
                         A_ub=self.coeff_matrix, b_ub=self.col_vec,
                         bounds=(None, None))

            if sf.success:
                return -sf.fun
            else:
                raise RuntimeError(sf.message)
