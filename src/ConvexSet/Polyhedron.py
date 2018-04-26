import cdd
import cvxopt as cvx
import numpy as np


class Polyhedron:
    """ A class for polyhedron. The H-representation is adopted.
        Namely, a polyhedron is represented as the intersection of
        halfspaces defined by inequalities:
            Ax <= b,
        where
            A: m*d Coefficients Matrix
            x: d-dimension column vector
            b: m-dimension column vector
    """

    def __init__(self, coeff_matrix, col_vec=None):
        coeff_matrix = coeff_matrix

        if col_vec is not None:
            assert coeff_matrix.shape[0] == col_vec.shape[0], \
                "Shapes of coefficient matrix %r and column vector %r do not match!" \
                % (coeff_matrix.shape, col_vec.shape)
        else:
            col_vec = np.zeros(shape=(coeff_matrix.shape[0], 1))

        if coeff_matrix.size > 0:
            self.coeff_matrix = coeff_matrix
            self.col_vec = col_vec
            self.cvx_coeff_matrix = cvx.matrix(coeff_matrix, tc='d')
            self.cvx_col_vec = cvx.matrix(col_vec, tc='d')
            # cdd requires b-Ax <= 0 as inputs

            self.mat_poly = cdd.Matrix(np.hstack((self.col_vec, -self.coeff_matrix)))
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

        if self.col_vec is not None:
            for row in np.hstack((self.coeff_matrix, self.col_vec)):
                str_repr += ' '.join(str(item) for item in row) + '\n'
        else:
            for row in self.coeff_matrix:
                str_repr += ' '.join(str(item) for item in row) + '\n'

        return str_repr

    def _gen_poly(self):
        try:
            return cdd.Polyhedron(self.mat_poly)
        except RuntimeError:
            print('\nEntries too large/small, pycddlib cannot handle the precision. Terminating now!\n')
            exit(-1)

    def _update_vertices(self):
        # return self.vertices.extend(gen[1:] for gen in self.poly.get_generators() if gen[0] == 1)
        return [gen[1:] for gen in self.poly.get_generators() if gen[0] == 1]

    def add_constraint(self, coeff_matrix, col_vec):
        self.mat_poly.extend([np.hstack((col_vec, -coeff_matrix))])
        self.poly = self._gen_poly()
        self.vertices = self._update_vertices()

        self.isEmpty = False
        self.isUniverse = False

    def get_inequalities(self):
        return np.hstack((self.coeff_matrix, self.col_vec))

    def is_empty(self):
        return self.isEmpty

    def is_universe(self):
        return self.isUniverse

    def compute_support_function(self, direction, lp):
        if self.is_empty():
            raise RuntimeError("\n Compute Support Function called for an Empty Set.")
        elif self.is_universe():
            raise RuntimeError("\n Cannot Compute Support Function of a Universe Polytope.\n")
        else:
            c = cvx.matrix(-direction, tc='d')
            return lp.lp(c, self.cvx_coeff_matrix, self.cvx_col_vec)

            # sol = cvx.solvers.lp(c, A, b, solver='glpk')
            # return direction.dot(np.array(sol['x']))[0]

    @staticmethod
    def compute_support_functions(coeff_mat, direction, b, lp):
        c = cvx.matrix(-direction, tc='d')
        coeff_mat = cvx.matrix(coeff_mat, tc='d')
        return lp.lp(c, coeff_mat, b)

    def compute_max_norm(self, lp):
        coeff_matrix = self.coeff_matrix
        dim_for_max_norm = coeff_matrix.shape[1]

        if self.isEmpty:
            return 0
        elif self.isUniverse:
            raise RuntimeError("Universe Unbounded Polytope!!!")
        else:
            generator_directions = []
            direction = np.zeros(dim_for_max_norm)
            for i in range(0, dim_for_max_norm):
                direction[i] = 1
                generator_directions.append(direction.copy())

                direction[i] = -1
                generator_directions.append(direction.copy())

            return max([self.compute_support_function(d, lp) for d in generator_directions])

    # def is_intersect_with(self, pl_2):
    #     """
    #     Process: Add all constraints of P1(the calling polyhedron) and P2 to form new constraints;
    #     then run lp_solver to tests if the constraints have no feasible solution.
    #     """
    #     raise NotImplementedError


if __name__ == '__main__':
    directions = np.array([[1, 0], [2, 0],
                           [-1, 0], [-2, 0],
                           [1, 1]])