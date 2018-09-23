import cvxopt as cvx

import utils.ppl_helper as pplHelper
from misc.basic_vector_operations import *


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
        coeff_matrix = np.array(coeff_matrix)
        col_vec = np.array(col_vec)

        assert coeff_matrix.size > 0, "Do not support empty polytope!"
        assert coeff_matrix.shape[0] == col_vec.shape[0], \
            "Shapes of coefficient matrix %r and column vector %r do not match!" % (coeff_matrix.shape, col_vec.shape)

        if coeff_matrix.size > 0:
            self.coeff_matrix = coeff_matrix
            self.col_vec = col_vec
            self.vertices = None
            self.cvx_coeff_matrix = cvx.matrix(coeff_matrix, tc='d')
            self.cvx_col_vec = cvx.matrix(col_vec, tc='d')
            # cdd requires b-Ax <= 0 as inputs

            # self.mat_poly = cdd.Matrix(np.hstack((self.col_vec, -self.coeff_matrix)))
            # self.mat_poly.rep_type = cdd.RepType.INEQUALITY

    def get_vertices(self):
        if self.vertices is None:
            self.vertices = pplHelper.get_vertices(self.coeff_matrix, self.col_vec, self.coeff_matrix.shape[1])
        return self.vertices

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

    # def _gen_poly(self):
    #     try:
    #         return cdd.Polyhedron(self.mat_poly)
    #     except RuntimeError:
    #         print('\nEntries too large/small, pycddlib cannot handle the precision. Terminating now!\n')
    #         exit(-1)

    # def get_2dVertices(self, dim1, dim2):
    #     # return self.vertices.extend(gen[1:] for gen in self.poly.get_generators() if gen[0] == 1)
    #     # return [gen[1:] for gen in self.poly.get_generators() if gen[0] == 1]
    #     set_vertices = self.enumerate_2dVertices(dim1, dim2)
    #
    #     return set_vertices
    #
    # def enumerate_2dVertices(self, i, j):
    #     all_vertices = []
    #     dim_num = self.coeff_matrix.shape[1]
    #
    #     u = np.zeros(dim_num)
    #     u[i] = 1
    #     v = np.zeros(dim_num)
    #     v[j] = 1
    #
    #     self.enum_2dVert_restrict(u, v, i, j, all_vertices)
    #
    #     u[i] = -1
    #     self.enum_2dVert_restrict(u, v, i, j, all_vertices)
    #
    #     v[j] = -1
    #     self.enum_2dVert_restrict(u, v, i, j, all_vertices)
    #
    #     u[i] = 1
    #     self.enum_2dVert_restrict(u, v, i, j, all_vertices)
    #
    #     return all_vertices
    #
    # def enum_2dVert_restrict(self, u, v, i, j, pts):
    #     # sv_u = np.zeros(self.coeff_matrix.shape[0])
    #     # sv_v = np.zeros(self.coeff_matrix.shape[0])
    #
    #     lp = GlpkWrapper(sys_dim=self.coeff_matrix.shape[1])
    #     sv_u = Polyhedron.compute_support_vectors(self.coeff_matrix, u, self.col_vec, lp)
    #     sv_v = Polyhedron.compute_support_vectors(self.coeff_matrix, v, self.col_vec, lp)
    #
    #     p1 = (sv_u[i], sv_u[j])
    #     p2 = (sv_v[i], sv_v[j])
    #
    #     pts.append(p1)
    #     pts.append(p2)
    #
    #     bisector = (normalize_vector(u) + normalize_vector(v)) / 2
    #     sv_bisect = Polyhedron.compute_support_vectors(self.coeff_matrix, bisector, self.col_vec, lp)
    #     p3 = (sv_bisect[i], sv_bisect[j])
    #
    #     if is_collinear(p1, p2, p3):
    #         return
    #     else:
    #         pts.append(p3)
    #         self.enum_2dVert_restrict(u, bisector, i, j, pts)
    #         self.enum_2dVert_restrict(bisector, v, i, j, pts)

    def get_inequalities(self):
        return np.hstack((self.coeff_matrix, self.col_vec))

    def compute_support_function(self, direction, lp):
        c = cvx.matrix(-direction, tc='d')
        return lp.lp(c, self.cvx_coeff_matrix, self.cvx_col_vec)

    @staticmethod
    def compute_support_functions(coeff_mat, direction, b, lp):
        c = cvx.matrix(-direction, tc='d')
        coeff_mat = cvx.matrix(coeff_mat, tc='d')
        b = cvx.matrix(b, tc='d')
        # print(c, coeff_mat, b)
        # print('\n')

        return lp.lp(c, coeff_mat, b)

    @staticmethod
    def compute_support_vectors(coeff_mat, direction, b, lp):
        c = cvx.matrix(-direction, tc='d')
        coeff_mat = cvx.matrix(coeff_mat, tc='d')
        b = cvx.matrix(b, tc='d')

        return lp.find_opt_point(c, coeff_mat, b)

    # @staticmethod
    # def is_collinear(p1, p2, p3):
    #     x1, y1 = p1
    #     x2, y2 = p2
    #     x3, y3 = p3
    #
    #     if np.isclose(x1, x2):
    #         if np.isclose(x2, x3):
    #             return True
    #         else:
    #             m = (y2 - y3) / (x2 - x3)
    #             c = y3 - m * x3
    #             return np.isclose(m*x1+c, y1)
    #     else:
    #         m = (y2 - y1) / (x2 - x1)
    #         c = y1 - m * x1
    #         return np.isclose(m*x3+c, y3)

    # @staticmethod
    # def normalize_vector(u):
    #     u = np.array(u)
    #     sqr_sum = sum(ele**2 for ele in u)
    #
    #     return u / sqr_sum

    def compute_max_norm(self, lp):
        coeff_matrix = self.coeff_matrix
        dim_for_max_norm = coeff_matrix.shape[1]

        generator_directions = []
        for i in range(0, dim_for_max_norm):
            direction = np.zeros(dim_for_max_norm)

            direction[i] = 1
            generator_directions.append(direction.copy())

            direction[i] = -1
            generator_directions.append(direction.copy())

        vals = [self.compute_support_function(cvx.matrix(d, tc='d'), lp) for d in generator_directions]
        return max(vals)


if __name__ == '__main__':
    directions = np.array([[1, 0], [2, 0],
                           [-1, 0], [-2, 0],
                           [1, 1]])

    coeff_mat = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    b = np.array([[1], [1], [1], [1]])
    # from utils.GlpkWrapper import GlpkWrapper
    # lp = GlpkWrapper(sys_dim=2)
    # sf_vals = [Polyhedron.compute_support_functions(coeff_mat, l, b, lp) for l in directions]
    #
    poly = Polyhedron(coeff_mat, b)
    print(poly.get_vertices())
    # sf_vals = [poly.compute_support_functions(coeff_mat, l, b, lp) for l in directions]
    #
    # print(set(poly.get_2dVertices(0, 1)))


