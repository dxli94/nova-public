import PPLHelper
from ConvexSet.HyperBox import HyperBox
from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from PostOperator import PostOperator
import SuppFuncUtils

import itertools
import numpy as np
import sympy
import pyibex

from scipy.optimize import minimize, least_squares

from SysDynamics import SysDynamics

generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def evaluate_exp(nonli_dyn, x, y):
    # for now just hard-code it
    # return np.array([y, -1*x-y])
    return np.array([y, (1 - x * x) * y - x])


class reachParams:
    def __init__(self, alpha=None, beta=None, delta_tp=None):
        self.alpha = alpha
        self.beta = beta
        self.delta_tp = delta_tp


class Hybridiser:
    def __init__(self, dim, nonlin_dyn, tau, directions, init_mat_X0, init_col_X0, is_linear):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.tau = tau
        self.coeff_matrix_B = np.identity(dim)
        self.post_opt = PostOperator()
        self.is_linear = is_linear

        # assume 2 dims for now, not hard to generalise
        self.variables = sympy.symbols('x,y', real=True)
        x, y = self.variables
        non_linear_dynamics = [y, (1 - x ** 2) * y - x]
        self.jacobian_func = sympy.Matrix(non_linear_dynamics).jacobian(self.variables)

        # the following attributes would be updated along the flowpipe construction
        self.directions = directions
        self.abs_dynamics = None
        self.abs_domain = None
        self.init_mat_X0 = init_mat_X0
        self.init_col_X0 = init_col_X0
        self.reach_params = reachParams()
        self.P = np.zeros(len(self.directions))
        # self.P_temp = np.zeros(len(self.directions))
        self.P_temp = np.array([np.inf, -np.inf]*2)
        self.X = np.zeros(len(self.directions))
        self.init_X = np.zeros(len(self.directions))

    def hybridise(self, bbox, starting_epsilon, lp):
        bbox.bloat(starting_epsilon)
        matrix_A, poly_U = self.gen_abs_dynamics(abs_domain=bbox)

        self.set_abs_dynamics(matrix_A, poly_U)
        self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau, lp)
        self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau, lp)
        self.reach_params.delta_tp = np.transpose(SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau))
        self.set_abs_domain(bbox)

    def gen_abs_dynamics(self, abs_domain):
        vertices = Polyhedron(*abs_domain.to_constraints()).vertices
        abs_domain_corners = np.array(vertices)
        abs_domain_centre = np.average(abs_domain_corners, axis=0)

        abs_domain_lower_bounds = abs_domain_corners.min(axis=0)
        abs_domain_upper_bounds = abs_domain_corners.max(axis=0)
        # for i in range(0, 100):
        # points_around_centre = self.do_sampling(abs_domain_lower_bounds, abs_domain_upper_bounds, abs_domain_centre)
        # # exit()
        # points_around_centre.add(tuple(abs_domain_centre))
        center_and_corners = np.append([abs_domain_centre], abs_domain_corners, axis=0)
        # sampling_points = [(cc, evaluate_exp(self.nonlin_dyn, cc[0], cc[1])) for cc in points_around_centre]
        sampling_points = [(cc, evaluate_exp(self.nonlin_dyn, cc[0], cc[1])) for cc in center_and_corners]
        # print(sampling_points)
        coeff_map = self.approx_non_linear_dyn(sampling_points)
        # matrix_A
        matrix_A = np.array(list(coeff_map[i] for i in range(self.dim)))

        u_max_array = []
        for i in range(matrix_A.shape[0]):
            if self.is_linear[i]:
                u_max_array.extend([0] * 2)
            else:
                # affine_dynamic = str(matrix_A[i][0]) + '*x[0] + ' + str(matrix_A[i][1]) + '*x[1]'
                x = abs_domain_centre
                coeff_vec = matrix_A[i]

                def err_func(x):
                    lin_func = np.dot(coeff_vec, x)
                    non_lin_func = evaluate_exp('', *x)
                    err = non_lin_func[1] - lin_func
                    return err

                def minus_err_func(x):
                    lin_func = np.dot(coeff_vec, x)
                    non_lin_func = evaluate_exp('', *x)
                    err = lin_func - non_lin_func[1]
                    return err

                # get_err_func([1, 1], x)
                bound = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

                u_min = minimize(err_func, x, bounds=bound).fun
                u_max = -minimize(minus_err_func, x, bounds=bound).fun

                u_max_array.extend([u_max, -u_min])
                # u_max_array.extend([0] * 2)

                # print('\n')
                # exit()
                # assuming 2 dimensions, can be easily generalised to n-dimension case
                # affine_dynamic = str(matrix_A[i][0]) + '*x[0] + ' + str(matrix_A[i][1]) + '*x[1]'
                # error_func_str = str(self.nonlin_dyn[i]) + '-(' + affine_dynamic + ')'
                # try:
                #     error_func = pyibex.Function("x[%d]" % self.dim, error_func_str)
                # except RuntimeError:
                #     print('severe error.')
                #
                # xy = pyibex.IntervalVector(
                #     [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)])
                # u_max_temp = error_func.eval(xy)
                # u_max = max(abs(u_max_temp[0]), abs(u_max_temp[1]))
                # u_min = min(abs(u_max_temp[0]), abs(u_max_temp[1]))
                # Todo Remember to change this back!!
                # print([u_max, u_min])
                # u_max_array.extend([u_max, u_min])
                #
        col_vec = np.array(u_max_array)
        # print(col_vec)

        # poly_U
        poly_U = (generator_2d_matrix, col_vec.reshape(len(col_vec), 1))
        return matrix_A, poly_U

    def do_sampling(self, abs_domain_lower_bounds, abs_domain_upper_bounds, abs_domain_centre):
        num = self.dim - 1
        width = abs_domain_upper_bounds - abs_domain_lower_bounds
        # print(c_x, c_y)
        new_width = []
        for wd in width:
            if wd >= 1e-6:
                new_width.append(wd)
            else:
                new_width.append(1e-6)
        new_width = np.array(new_width)
        points = set()
        # print(abs_domain_upper_bounds)
        # print(abs_domain_lower_bounds)
        i = 0
        while i < num:
            point = tuple(new_width * np.random.random_sample() + abs_domain_centre)
            # y = new_width[i] * np.random.random_sample() + c_y
            if point not in points:
                points.add(point)
                i += 1
        return points

    def approx_non_linear_dyn(self, sampling_points):
        coeff_map = {}
        combination = itertools.combinations(sampling_points, self.dim)
        for comb in combination:
            x = np.array(list(comb[i][0] for i in range(self.dim)))
            for k in range(self.dim):
                if k not in coeff_map:
                    b_x = np.array(list(comb[i][1][k] for i in range(self.dim)))
                    try:
                        a_x = np.linalg.solve(x, b_x)
                    except np.linalg.LinAlgError:
                        continue

                    # if any(np.isnan(a_x)) or any(np.isinf(a_x)):
                    #     continue
                    coeff_map[k] = a_x

                    if len(coeff_map) == self.dim:
                        # print(coeff_map)
                        return coeff_map

    def compute_alpha_step(self, lp):
        poly_init = Polyhedron(self.directions, self.X)
        trans_poly_U = TransPoly(self.abs_dynamics.matrix_B,
                                 self.abs_dynamics.coeff_matrix_U,
                                 self.abs_dynamics.col_vec_U)

        sf_arr = [self.post_opt.compute_initial_sf(self.reach_params.delta_tp, poly_init, trans_poly_U, l,
                                                   self.reach_params.alpha, self.tau, lp) for l in self.directions]
        self.P_temp = np.array(sf_arr)

    def compute_beta_step(self, s_vec, r_vec, lp):
        sf_vec = []
        current_s_array = []
        current_r_array = []
        poly_init = Polyhedron(self.directions, self.init_X)
        trans_poly_U = TransPoly(self.abs_dynamics.matrix_B, self.abs_dynamics.coeff_matrix_U,
                                 self.abs_dynamics.col_vec_U)

        for idx in range(len(r_vec)):
            prev_r = r_vec[idx]
            prev_s = s_vec[idx]
            r = np.dot(self.reach_params.delta_tp, prev_r)
            s = prev_s + self.post_opt.compute_sf_w(prev_r, trans_poly_U, self.reach_params.beta, self.tau, lp)
            sf = s + self.post_opt.compute_initial_sf(self.reach_params.delta_tp,
                                                      poly_init, trans_poly_U,
                                                      r,
                                                      self.reach_params.alpha,
                                                      self.tau,
                                                      lp)
            sf_vec.append(sf)
            current_s_array.append(s)
            current_r_array.append(r)

        self.P_temp = np.array(sf_vec)
        return current_s_array, current_r_array

    def compute_gamma_step(self, lp):
        sf_vec = []
        poly_init = Polyhedron(self.directions, self.X)
        # print(poly_init.vertices)
        trans_poly_U = TransPoly(self.abs_dynamics.matrix_B, self.abs_dynamics.coeff_matrix_U,
                                 self.abs_dynamics.col_vec_U)

        for l, sf_val in zip(self.directions, self.X):
            r = np.dot(self.reach_params.delta_tp, l)
            s = self.post_opt.compute_sf_w(r, trans_poly_U, 0, self.tau, lp)
            # s = 0
            sf_X0 = poly_init.compute_support_function(r, lp)
            sf = sf_X0 + s
            sf_vec.append([sf])

        self.X = np.array(sf_vec)
        # self.X = np.array(sf_vec).reshape(len(self.directions), 1)

    def set_abs_dynamics(self, matrix_A, poly_U):
        abs_dynamics = SysDynamics(dim=self.dim,
                                   init_coeff_matrix_X0=self.init_mat_X0,
                                   init_col_vec_X0=self.init_col_X0,
                                   dynamics_matrix_A=matrix_A,
                                   dynamics_matrix_B=self.coeff_matrix_B,
                                   dynamics_coeff_matrix_U=poly_U[0],
                                   dynamics_col_vec_U=poly_U[1])
        self.abs_dynamics = abs_dynamics

    def set_abs_domain(self, abs_domain):
        self.abs_domain = abs_domain

    def update_init_image(self, init_mat, init_col):
        self.init_mat_X0 = init_mat
        self.init_col_X0 = init_col

    def refine_domain(self):
        # take the convex hall of P_{i} and P_{i+1}
        bounds = np.maximum(self.P_temp.reshape(len(self.directions), 1),
                            self.P.reshape(len(self.directions), 1))

        vertices = Polyhedron(self.directions, bounds).vertices
        bbox = HyperBox(vertices)

        return bbox

    def reset_P_temp(self):
        self.P_temp = np.array([np.inf, -np.inf]*2)


def estimate_error(func, bound, minOrMax, x):
    from scipy.optimize import minimize_scalar
    res = minimize(func, x, bounds=[bound, bound])

    print(res)


if __name__ == '__main__':
    from scipy.optimize import leastsq

    def func(x, p):
        return np.dot(x, p.reshape(2, 1).flatten())

    def residuals(p, y, x):
        return (y - func(x, p)).tolist()

    abs_bound = [[0, 1], [0, 1]]

    points = []
    x = np.random.random_sample((30, 2))
    y0 = [evaluate_exp('', x0, x1)[1] for x0, x1 in x]
    p0 = [10, 1]
    plsq = leastsq(residuals, p0, args=(y0, x))
    print(plsq[0])