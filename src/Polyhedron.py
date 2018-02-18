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

    def __init__(self, mat):
        """
            mat - type: list, numpy array, numpy Matrix
                  m*(d+1) Matrix, first m-column is A, last column is b.
        """
        self.mat_input = mat

        mat_restack = self._restack_mat(mat)
        self.mat_poly = cdd.Matrix(mat_restack)
        self.mat_poly.rep_type = cdd.RepType.INEQUALITY

        self._gen_poly()

        # TODO - need to actually test and then set!!
        self.isEmpty = False
        self.isUniverse = False

    def __str__(self):
        str_repr = 'H-representative\n' + \
                   'Ax <= b \n'
        for row in self.mat_input:
            str_repr += ' '.join(str(item) for item in row) + '\n'

        return str_repr

    def _restack_mat(self, mat):
        np_mat = np.matrix(mat)
        mat_A, vec_b = np.hsplit(np_mat, np.array([np_mat.shape[1] - 1, ]))
        self.A = mat_A
        self.b = vec_b

        # cdd requires b-Ax <= 0 as inputs
        return np.hstack((vec_b, -mat_A)).tolist()

    def _gen_poly(self):
        self.poly = cdd.Polyhedron(self.mat_poly)
        self.vertices = []
        self.vertices.extend(gen[1:] for gen in self.poly.get_generators() if gen[0] == 1)

    def add_constraint(self, constr):
        self.mat_poly.extend(self._restack_mat(constr))
        self._gen_poly()

    def get_inequalities(self):
        return self.mat_input

    def is_empty(self):
        return self.isEmpty

    def is_universe(self):
        return self.isUniverse

    def is_intersect_with(self, pl):
        """
        Process: Add all constraints of P1(the calling polyhedron) and P2 to form new constraints;
        then run lp_solver to test if the constraints have no feasible solution.
        """
        raise NotImplementedError


if __name__ == '__main__':
    # test creation
    # mat = np.array([[1, 1, 2], [-1, 0, 0], [0, -1, 0]])
    mat = [[1, 1, 2], [-1, 0, 0], [0, -1, 0]]
    poly = Polyhedron(mat)
    test_1 = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]
    assert all(item in poly.vertices for item in test_1) and all(item in test_1 for item in poly.vertices)

    # test add_constraint
    poly.add_constraint([[1, -1, 0]])
    test_2 = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]
    assert all(item in poly.vertices for item in test_2) and all(item in test_2 for item in poly.vertices)

    # print(poly)
    # print(poly.get_inequalities())
