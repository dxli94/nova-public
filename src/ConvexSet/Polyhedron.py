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
        return cdd.Polyhedron(self.mat_poly)

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

    def compute_support_function(self, direction):
        if self.is_empty():
            raise RuntimeError("\n Compute Support Function called for an Empty Set.")
        elif self.is_universe():
            raise RuntimeError("\n Cannot Compute Support Function of a Universe Polytope.\n")
        else:
            # Note that these bounds have to be set to (None, None) to allow (-inf, +inf),
            # otherwise (0, +inf) by default.
            # Besides, Scipy only deals with min(). Here we need max(). max(f(x)) = -min(-f(x))
            # A_ub * x <= b_ub
            sf = linprog(c=-direction,
                         A_ub=self.coeff_matrix, b_ub=self.col_vec,
                         bounds=(None, None))
            if sf.success:
                return -sf.fun
            else:
                raise RuntimeError(sf.message)

    def compute_max_norm(self):
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

            return max([self.compute_support_function(d) for d in generator_directions])

    # def is_intersect_with(self, pl_2):
    #     """
    #     Process: Add all constraints of P1(the calling polyhedron) and P2 to form new constraints;
    #     then run lp_solver to test if the constraints have no feasible solution.
    #     """
    #     raise NotImplementedError
