from itertools import product

import numpy as np
from pyibex import Function, IntervalVector
from scipy.optimize import basinhopping

from utils.pykodiak.pykodiak_interface import Kodiak
from utils.timerutil import Timers


def get_generator_matrix(dim):
    rv = np.zeros(shape=(2*dim, dim))

    for i, row in enumerate(rv):
        row[i // 2] = (-1)**i

    return rv


class Linearizer:
    def __init__(self, dim, nonlin_dyn, is_linear):
        self.dim = dim
        self.target_dyn = nonlin_dyn
        self.is_linear = is_linear
        self.is_scaled = False

        self.generator_matrix = get_generator_matrix(dim)

    # maximize (ax+b-g(x)) is equiv. to -minimize(g(x)-(ax+b))
    def err_func(self, x, *args):
        coeff_vec = args[0]
        bias = args[1]
        i = args[2]

        lin_func = np.dot(coeff_vec, x) + bias
        non_lin_func = self.target_dyn.eval(x)
        err = non_lin_func[i] - lin_func
        # return 1
        return err

    # maxmize (g(x)-(ax+b)) is equiv. to -minimize(ax+b-g(x))
    def minus_err_func(self, x, *args):
        coeff_vec = args[0]
        bias = args[1]
        i = args[2]

        lin_func = np.dot(coeff_vec, x) + bias
        non_lin_func = self.target_dyn.eval(x)
        err = lin_func - non_lin_func[i]
        return err

    def err_func_jac(self, x, args):
        coeff = args[0]
        i = args[2]
        nonlin_jac = self.target_dyn.eval_jacobian(x)[i]

        return nonlin_jac - coeff

    def minus_err_func_jac(self, x, args):
        coeff = args[0]
        i = args[2]
        nonlin_jac = self.target_dyn.eval_jacobian(x)[i]

        return coeff - nonlin_jac

    def gen_abs_dynamics(self, abs_domain_bounds):
        # Timers.tic('total')

        # Somehow this is faster than taking mean/average directly
        abs_domain_centre = np.sum(abs_domain_bounds, axis=0) / 2

        abs_domain_lower_bounds = abs_domain_bounds[0]
        abs_domain_upper_bounds = abs_domain_bounds[1]

        matrix_A, b = self.jacobian_linearize(abs_domain_centre, self.target_dyn)

        u_bounds = []
        # Timers.tic('loop')
        for i in range(self.dim):
            if not self.is_scaled and self.is_linear[i]:
                u_min = u_max = 0
            else:
                coeff = matrix_A[i]
                bias = b[i]

                bounds = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

                Timers.tic('minmax')
                kodiak_res = Kodiak.minmax(self.target_dyn.kodiak_ders[i], coeff, bias, bounds)
                u_min, u_max = -kodiak_res[0], kodiak_res[1]
                Timers.toc('minmax')

                # print(u_min, u_max)
                # print('\n')
                # args = (coeff, bias, i)
                # x0 = abs_domain_centre
                #
                # minimizer_kwargs_1 = dict(method='L-BFGS-B', bounds=bounds, args=args, jac=lambda *args: self.err_func_jac(args[0], args[1:]))
                # minimizer_kwargs_2 = dict(method='L-BFGS-B', bounds=bounds, args=args, jac=lambda *args: self.minus_err_func_jac(args[0], args[1:]))

                # Timers.tic('basinhopping')
                # u_min = -basinhopping(self.err_func, x0, minimizer_kwargs=minimizer_kwargs_1, niter_success=3).fun
                # u_max = -basinhopping(self.minus_err_func, x0, minimizer_kwargs=minimizer_kwargs_2, niter_success=3).fun
                # Timers.toc('basinhopping')
                # print(u_min, u_max)
                # exit()
                # print('\n')

                # exit()

            u_bounds.extend([u_max, u_min])

        col_vec = np.array(u_bounds)

        # generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        poly_U = (self.generator_matrix, col_vec.reshape(len(col_vec), 1))

        return matrix_A, poly_U, b

    @staticmethod
    def maximize_diff(args):
        func, x0, kwargs, niter_success = args
        return -basinhopping(func, x0, minimizer_kwargs=kwargs, niter=niter_success).func

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

    def set_target_dyn(self, dynamics):
        self.target_dyn = dynamics
