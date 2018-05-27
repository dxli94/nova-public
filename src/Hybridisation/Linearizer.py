from ConvexSet.Polyhedron import Polyhedron
import numpy as np
from scipy.optimize import minimize

generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


class Linearizer:
    def __init__(self, dim, nonlin_dyn, is_linear):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.is_linear = is_linear

    # maximize (ax+b-g(x)) is equiv. to -minimize(g(x)-(ax+b)) todo double-check, seems to always return x0
    def err_func(self, x, *args):
        coeff_vec = args[0]
        bias = args[1]
        i = args[2]

        lin_func = np.dot(coeff_vec, x) + bias
        non_lin_func = self.nonlin_dyn.eval(x)
        err = non_lin_func[i] - lin_func
        return err

    # maxmize (g(x)-(ax+b)) is equiv. to -minimize(ax+b-g(x)) todo double-check, seems to always return x0
    def minus_err_func(self, x, *args):
        coeff_vec = args[0]
        bias = args[1]
        i = args[2]

        lin_func = np.dot(coeff_vec, x) + bias
        non_lin_func = self.nonlin_dyn.eval(x)
        err = lin_func - non_lin_func[i]
        return err

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
            x0 = abs_domain_centre
            coeff_vec = matrix_A[i]
            bias = b[i]
            bound = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

            args = (coeff_vec, bias, i)
            resmin = minimize(self.err_func, x0, bounds=bound, tol=1e-25, args=args)
            resmax = minimize(self.minus_err_func, x0, bounds=bound, tol=1e-25, args=args)

            u_min = -resmin.fun
            u_max = -resmax.fun

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