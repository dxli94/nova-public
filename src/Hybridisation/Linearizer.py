from ConvexSet.Polyhedron import Polyhedron
import numpy as np
from scipy.optimize import minimize

generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


class Linearizer:
    def __init__(self, dim, nonlin_dyn, is_linear):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.is_linear = is_linear

    def gen_abs_dynamics(self, abs_domain):
        vertices = Polyhedron(*abs_domain.to_constraints()).vertices
        abs_domain_corners = np.array(vertices)
        abs_domain_centre = np.average(abs_domain_corners, axis=0)

        abs_domain_lower_bounds = abs_domain_corners.min(axis=0)
        abs_domain_upper_bounds = abs_domain_corners.max(axis=0)

        # matrix_A, b = self.jacobian_linearize(abs_domain_centre, self.nonlin_dyn)
        matrix_A, b = self.jacobian_linearize_without_b(abs_domain_centre, self.nonlin_dyn)

        u_max_array = []
        for i in range(self.dim):
            # affine_dynamic = str(matrix_A[i][0]) + '*x[0] + ' + str(matrix_A[i][1]) + '*x[1]'
            x = abs_domain_centre
            coeff_vec = matrix_A[i]
            bias = b[i]

            # maximize (ax+b-g(x)) is equiv. to -minimize(g(x)-(ax+b))
            def err_func(x):
                lin_func = np.dot(coeff_vec, x) + bias
                non_lin_func = self.nonlin_dyn.eval(x)
                err = non_lin_func[i] - lin_func
                # print(err)
                return err

            # maxmize (g(x)-(ax+b)) is equiv. to -minimize(ax+b-g(x))
            def minus_err_func(x):
                lin_func = np.dot(coeff_vec, x) + bias
                non_lin_func = self.nonlin_dyn.eval(x)
                err = lin_func - non_lin_func[i]
                return err

            # get_err_func([1, 1], x)
            bound = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

            u_min = -minimize(err_func, x, bounds=bound).fun
            u_max = -minimize(minus_err_func, x, bounds=bound).fun
            u_max_array.extend([b[i]+u_max, -b[i]+u_min])
            # u_max_array.extend([b[i], -b[i]])

        col_vec = np.array(u_max_array)

        # poly_U
        poly_U = (generator_2d_matrix, col_vec.reshape(len(col_vec), 1))

        return matrix_A, poly_U

    @staticmethod
    def jacobian_linearize(abs_center, non_linear_dyn):
        # f(x, y) = J(x - x0, y - y0) * [x - x0, y - y0].T + g(x0, y0)
        mat_a = np.array(non_linear_dyn.eval_jacobian(abs_center)).astype(np.float64)
        b = non_linear_dyn.eval(abs_center) - mat_a.dot(abs_center)

        return mat_a, b

    @staticmethod
    def jacobian_linearize_without_b(abs_center, non_linear_dyn):
        mat_a = np.array(non_linear_dyn.eval_jacobian(abs_center)).astype(np.float64)
        b = [0] * len(abs_center)
        return mat_a, b