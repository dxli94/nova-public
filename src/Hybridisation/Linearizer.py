from ConvexSet.Polyhedron import Polyhedron
import numpy as np
from scipy.optimize import minimize, basinhopping
from itertools import product
from pyibex import Function, IntervalVector

generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


class Linearizer:
    def __init__(self, dim, nonlin_dyn, is_linear):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.is_linear = is_linear
        self.is_scaled = False

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

    def gen_abs_dynamics(self, abs_domain_bounds):
        # Timers.tic('total')

        # Somehow this is faster than taking mean/average directly
        abs_domain_centre = np.sum(abs_domain_bounds, axis=0) / 2

        abs_domain_lower_bounds = abs_domain_bounds[0]
        abs_domain_upper_bounds = abs_domain_bounds[1]

        matrix_A, b = self.jacobian_linearize(abs_domain_centre, self.nonlin_dyn)

        u_bounds = []
        # Timers.tic('loop')
        for i in range(self.dim):
            if self.is_linear[i] and not self.is_scaled:
                u_min = u_max = 0
            else:
                coeff = matrix_A[i]
                bias = b[i]
                affine_dyn_rep = coeff2ibex(coeff, bias)
                nonlin_dyn_rep = sympy2ibex(self.nonlin_dyn.str_rep[i])

                diff_rep = '{}-({})'.format(nonlin_dyn_rep, affine_dyn_rep)
                bounds = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

                args = (coeff, bias, i)
                x0 = abs_domain_centre
                # res = self.interval_diff(diff_rep, bounds)
                minimizer_kwargs = dict(method='L-BFGS-B', bounds=bounds, args=args)
                # u_min = -basinhopping(self.err_func, x0, minimizer_kwargs=minimizer_kwargs, niter=20).fun
                # u_max = -basinhopping(self.minus_err_func, x0, minimizer_kwargs=minimizer_kwargs, niter=20).fun
                # u_min = -basinhopping(self.err_func, x0, minimizer_kwargs=minimizer_kwargs, niter=200).fun
                # print(u_min)
                # u_min = -basinhopping(self.err_func, x0, minimizer_kwargs=minimizer_kwargs, niter=5).fun
                # u_max = -basinhopping(self.minus_err_func, x0, minimizer_kwargs=minimizer_kwargs, niter=5).fun

                u_min = -basinhopping(self.err_func, x0, minimizer_kwargs=minimizer_kwargs, niter_success=3).fun
                u_max = -basinhopping(self.minus_err_func, x0, minimizer_kwargs=minimizer_kwargs, niter_success=3).fun

            u_bounds.extend([u_max, u_min])

            # if self.is_scaled:
            #     print(u_bounds)

        col_vec = np.array(u_bounds)

        poly_U = (generator_2d_matrix, col_vec.reshape(len(col_vec), 1))

        return matrix_A, poly_U, b

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

    def set_nonlin_dyn(self, dynamics):
        self.nonlin_dyn = dynamics

    def interval_diff(self, f, bounds):
        ibex_func = Function("x[{}]".format(self.dim), f)
        # [[[], []], [[], []], [[], []]]
        split_bounds = []
        n = 2

        for bound in bounds:
            lb, ub = bound
            stepsize = (ub-lb)/n
            temp = []

            start = lb
            for i in range(n):
                temp.append([start, start + stepsize])
                start += stepsize

            split_bounds.append(temp)

        catersian_comb = product(*split_bounds)
        maxval = -1e6
        minval = 1e6
        minvec = []
        maxvec = []

        for elem in catersian_comb:
            val = ibex_func.eval(IntervalVector(elem))
            if val[1] > maxval:
                maxvec = elem
                maxval = val[1]

            if val[0] < minval:
                minvec = elem
                minval = val[0]

        # print(minval, maxval)
        # print(minvec, maxvec)
        # print(f)
        # exit()

        return minvec, maxvec
        # return minval, maxval

def sympy2ibex(sympy_str):
    ibex_str = ''
    flag = False
    for c in sympy_str:
        if c == 'x':
            ibex_str += 'x['
            flag = True
        elif flag:
            ibex_str += c + ']'
            flag = False
        else:
            ibex_str += c

    return ibex_str


def coeff2ibex(coeff_vec, bias):
    ibex_str = ''
    for idx, elem in enumerate(coeff_vec):
        ibex_str += str(elem) + '*x[{}]+'.format(idx)

    ibex_str += str(bias)
    return ibex_str


# if __name__ == '__main__':
#     s = '1+x0^2*x1-1.5*x0-x0'
#     sympy_s = sympy2ibex(s)
#     # print(sympy2ibex(s))
#     f = Function('x[2]', sympy_s)
#     bounds = IntervalVector([[0, 1], [1, 2]])
#
#     import time
#     start_time = time.time()
#     for i in range(100000):
#         f.eval(bounds)
#     print(time.time() - start_time)
#
#     a = [1, 2, 3]
#     bias = 0.1
    # print(coeff2ibex(a, bias))
