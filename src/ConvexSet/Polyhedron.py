from scipy.optimize import linprog

import numpy as np
import cdd


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

    def __init__(self, coeff_matrix, col_vec):
        coeff_matrix = np.matrix(coeff_matrix)
        col_vec = np.matrix(col_vec)

        assert coeff_matrix.shape[0] == col_vec.shape[0], \
            "Shapes of coefficient matrix %r and column vector %r do not match!" \
            % (coeff_matrix.shape, col_vec.shape)

        if coeff_matrix.size > 0:
            self.coeff_matrix = coeff_matrix
            self.vec_col = col_vec
            # cdd requires b-Ax <= 0 as inputs
            self.mat_poly = cdd.Matrix(np.hstack((self.vec_col, -self.coeff_matrix)).tolist())
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
        for row in np.hstack((self.coeff_matrix, self.vec_col)):
            str_repr += ' '.join(str(item) for item in row) + '\n'

        return str_repr

    def _gen_poly(self):
        return cdd.Polyhedron(self.mat_poly)

    def _update_vertices(self):
        # return self.vertices.extend(gen[1:] for gen in self.poly.get_generators() if gen[0] == 1)
        return [gen[1:] for gen in self.poly.get_generators() if gen[0] == 1]

    def add_constraint(self, coeff_matrix, col_vec):
        self.mat_poly.extend(np.matrix(np.hstack((np.matrix(col_vec), -np.matrix(coeff_matrix)))).tolist())
        self.poly = self._gen_poly()
        self.vertices = self._update_vertices()

        self.isEmpty = False
        self.isUniverse = False

    def get_inequalities(self):
        return np.hstack((self.coeff_matrix, self.vec_col))

    def is_empty(self):
        return self.isEmpty

    def is_universe(self):
        return self.isUniverse

    def compute_support_function(self, direction):
        if self.is_empty():
            raise RuntimeError("\n Compute Support Function called for an Empty Set.")
        elif self.is_universe():
            raise RuntimeError("\n Cannot Compute Support Function of a Universe Polytope.\n")
        else:
            # Note that these bounds have to be set to (None, None) to allow (-inf, +inf),
            # otherwise (0, +inf) by default.
            # Besides, Scipy only deals with min(). Here we need max(). max(f(x)) = -min(-f(x))
            sf = linprog(c=-direction,
                         A_ub=self.coeff_matrix, b_ub=self.vec_col,
                         bounds=(None, None))
            if sf.success:
                return -sf.fun
            else:
                raise RuntimeError(sf.message)

    def compute_max_norm(self):
        coeff_matrix = np.matrix(self.coeff_matrix)
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

            return max([self.compute_support_function(d) for d in generator_directions])

    def is_intersect_with(self, pl_2):
        """
        Process: Add all constraints of P1(the calling polyhedron) and P2 to form new constraints;
        then run lp_solver to test if the constraints have no feasible solution.
        """
        raise NotImplementedError

if __name__ == '__main__':
    # test creation
    coeff_matrix = [[1, 1], [-1, 0], [0, -1]]
    col_vec = [[2], [0], [0]]
    poly = Polyhedron(coeff_matrix, col_vec)
    test_1 = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]
    assert all(item in poly.vertices for item in test_1) and all(item in test_1 for item in poly.vertices)

    # test add_constraint()
    poly.add_constraint([1, -1], [0])
    test_2 = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]
    assert all(item in poly.vertices for item in test_2) and all(item in test_2 for item in poly.vertices)

    # test get_inequalities()
    assert poly.get_inequalities().all() == np.matrix([[1, 1, 2], [-1, 0, 0], [0, -1, 0]]).all()

    # test compute_max_norm()
    assert poly.compute_max_norm() == 2
