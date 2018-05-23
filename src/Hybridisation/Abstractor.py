from ConvexSet.Polyhedron import Polyhedron
import numpy as np
from scipy.optimize import minimize

generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


class Abstractor:
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

        matrix_A, b = self.jacobian_linearise(abs_domain_centre, self.nonlin_dyn)

        u_max_array = []
        for i in range(self.dim):
            if self.is_linear[i]:
                u_max_array.extend([b[i], -b[i]])
                # u_max_array.extend([0] * 2)
            else:
                # affine_dynamic = str(matrix_A[i][0]) + '*x[0] + ' + str(matrix_A[i][1]) + '*x[1]'
                x = abs_domain_centre
                coeff_vec = matrix_A[i]

                def err_func(x):
                    lin_func = np.dot(coeff_vec, x)
                    non_lin_func = self.nonlin_dyn.eval(x)
                    err = non_lin_func[1] - lin_func
                    return err

                def minus_err_func(x):
                    lin_func = np.dot(coeff_vec, x)
                    non_lin_func = self.nonlin_dyn.eval(x)
                    err = lin_func - non_lin_func[1]
                    return err

                # get_err_func([1, 1], x)
                bound = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

                u_min = minimize(err_func, x, bounds=bound).fun
                u_max = -minimize(minus_err_func, x, bounds=bound).fun

                # print(u_min, u_max)

                # u_max_array.extend([u_max, -u_min])
                u_max_array.extend([b[i], -b[i]])

                # u_max_array.extend([0] * 2)

        col_vec = np.array(u_max_array)

        # poly_U
        poly_U = (generator_2d_matrix, col_vec.reshape(len(col_vec), 1))

        return matrix_A, poly_U

    def jacobian_linearise(self, abs_centre, non_linear_dyn):
        # f(x, y) = J(x - x0, y - y0) * [x - x0, y - y0].T + g(x0, y0)
        mat_a = np.array(non_linear_dyn.eval_jacobian(abs_centre)).astype(np.float64)
        b = non_linear_dyn.eval(abs_centre) - mat_a.dot(abs_centre)

        return mat_a, b