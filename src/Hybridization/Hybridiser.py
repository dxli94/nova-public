import PPLHelper
from ConvexSet.Polyhedron import Polyhedron
from PostOperator import PostOperator
import SuppFuncUtils

import itertools
import numpy as np
import sympy
import pyibex

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
        # print(self.jacobian_func)
        # exit()

        # the following attributes would be updated along the flowpipe construction
        self.directions = directions
        self.abs_dynamics = None
        self.abs_domain = None
        self.init_mat_X0 = init_mat_X0
        self.init_col_X0 = init_col_X0
        self.sf = np.zeros(len(self.directions))
        self.reach_params = reachParams()
        self.prev_sf = np.zeros(len(self.directions))
        # self.r_array = []

    def hybridise(self, bbox, starting_epsilon):
        bbox.bloat(starting_epsilon)
        matrix_A, poly_U = self.gen_abs_dynamics(abs_domain=bbox)

        self.set_abs_dynamics(matrix_A, poly_U)
        self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau)
        self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau)
        self.reach_params.delta_tp = np.transpose(SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau))
        self.set_abs_domain(bbox)

    def gen_abs_dynamics(self, abs_domain):
        vertices = Polyhedron(*abs_domain.to_constraints()).vertices
        abs_domain_corners = np.array(vertices)
        abs_domain_centre = np.average(abs_domain_corners, axis=0)

        # matrix_A = np.array(self.jacobian_func.subs(list(zip(self.variables, abs_domain_centre)))).astype(np.float64)
        # matrix_A = np.array(self.jacobian_func.subs(list(zip(self.variables, abs_domain_corners[-1])))).astype(np.float64)

        abs_domain_lower_bounds = abs_domain_corners.min(axis=0)
        abs_domain_upper_bounds = abs_domain_corners.max(axis=0)

        center_and_corners = np.append([abs_domain_centre], abs_domain_corners, axis=0)
        sampling_points = [(cc, evaluate_exp(self.nonlin_dyn, cc[0], cc[1])) for cc in center_and_corners]
        coeff_map = self.approx_non_linear_dyn(sampling_points)

        # matrix_A
        matrix_A = np.array(list(coeff_map[i] for i in range(self.dim)))

        # print('matrix_A: ' + str(matrix_A))
        # print('abs centre: ' + str(abs_domain_centre))
        # print('approx. val: ' + str(np.dot(matrix_A, abs_domain_centre)))
        # print('real val: ' + str(evaluate_exp(self.nonlin_dyn, abs_domain_centre[0], abs_domain_centre[1])))
        # print('\n')

        u_max_array = []
        for i in range(matrix_A.shape[0]):
            if self.is_linear[i]:
                u_max_array.extend([0] * 2)
            else:
                # assuming 2 dimensions, can be easily generalised to n-dimension case
                affine_dynamic = str(matrix_A[i][0]) + '*x[0] + ' + str(matrix_A[i][1]) + '*x[1]'
                error_func_str = str(self.nonlin_dyn[i]) + '-(' + affine_dynamic + ')'
                error_func = pyibex.Function("x[%d]" % self.dim, error_func_str)

                xy = pyibex.IntervalVector(
                    [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)])
                u_max_temp = error_func.eval(xy)
                u_max = max(abs(u_max_temp[0]), abs(u_max_temp[1]))
                """
                Remember to change this back!!
                TODO
                """
                # u_max_array.extend([u_max] * 2)

                u_max_array.extend([0] * 2)
        col_vec = np.array(u_max_array)
        # print(col_vec)

        # poly_U
        poly_U = (generator_2d_matrix, col_vec.reshape(len(col_vec), 1))

        return matrix_A, poly_U

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
                        coeff_map[k] = a_x
                    except np.linalg.LinAlgError:
                        continue

                    if len(coeff_map) == self.dim:
                        return coeff_map

    def compute_initial_image(self):
        sf_array = self.post_opt.compute_initial(abs_dynamics=self.abs_dynamics,
                                                 delta_tp=self.reach_params.delta_tp,
                                                 tau=self.tau,
                                                 alpha=self.reach_params.alpha,
                                                 directions=self.directions)
        self.sf = np.array(sf_array)

    # abs_dynamics, delta_tp, tau, alpha, beta, prev_directions, sf_current
    def compute_next_image(self, s_arr, r_arr):
        next_image, s_arr, r_arr = self.post_opt.compute_next(abs_dynamics=self.abs_dynamics,
                                                              delta_tp=self.reach_params.delta_tp,
                                                              tau=self.tau,
                                                              alpha=self.reach_params.alpha,
                                                              beta=self.reach_params.beta,
                                                              prev_directions=r_arr,
                                                              prev_s_array=s_arr)
        # print(next_image)
        self.sf = np.array(next_image)
        return s_arr, r_arr

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
        # self.init_coeff_matrix = init_coeff_matrix_X0
        # self.init_col_vec = init_col_vec_X0
        self.init_mat_X0 = init_mat
        self.init_col_X0 = init_col
