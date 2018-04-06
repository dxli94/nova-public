import PPLHelper
from ConvexSet.Polyhedron import Polyhedron
from PostOperator import PostOperator
import SuppFuncUtils

import itertools
import numpy as np
import pyibex

from SysDynamics import SysDynamics

generator_2d_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def evaluate_exp(nonli_dyn, x, y):
    # for now just hard-code it
    return np.array([y, (1 - x * x) * y - x])


class reachParams:
    def __init__(self, alpha=None, beta=None, delta_tp=None):
        self.alpha = alpha
        self.beta = beta
        self.delta_tp = delta_tp


class Hybridizer:
    def __init__(self, dim, nonlin_dyn, starting_epsilon, tau, directions):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.epsilon = starting_epsilon
        self.tau = tau
        self.coeff_matrix_B = np.identity(dim)
        self.post_opt = PostOperator()

        # the following attributes would be updated along the flowpipe construction
        self.directions = directions
        self.current_image_coff_mat = None
        self.current_image_col_vec = None
        self.abs_dynamics = None
        self.abs_domain = None
        self.reach_params = reachParams()

    def reset_abs_domain(self, bbox, starting_epsilon):
        self.abs_domain = bbox
        self.epsilon = starting_epsilon

    def hybridise(self, bbox):
        """
        Take the hybridisation domain, do the following:
            1) approximate non-linear dynamics on the hybridisation domain using affine dynamics;
            2) compute image reachable in the current time step;
            3) if image is contained in the hybridisation domain, then return the affine dynnamics;
               else: re-bloat the hybridisation domain with larger epsilon and goto 2).

        :param bbox: hybridisation domain to start with.
        :return: affine dynamics, hybridisation domain constructed.
        """
        bbox.bloat(self.epsilon)
        ppl_bbox = bbox.to_ppl()

        while True:
            # now, refine the bounding box by checking whether the bbox contains the image.
            # 1) approximate non-linear dynamic g(x) with affine dynamic Ax + u
            matrix_A, poly_U = self.gen_abs_dynamics(abs_domain=bbox)
            self.set_abs_dynamics(matrix_A, poly_U)

            self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau)
            self.reach_params.delta_tp = np.transpose(SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau))
            sf = self.post_opt.compute_initial(self.abs_dynamics, self.reach_params.delta_tp, self.tau,
                                               self.reach_params.alpha, self.directions)
            ppl_image = PPLHelper.create_ppl_polyhedra_from_support_functions(sf, self.directions, self.dim)

            if PPLHelper.contains(ppl_bbox, ppl_image):
                self.set_abs_domain(bbox)
                self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau)
                self.current_image_coff_mat = self.directions
                self.current_image_col_vec = np.array(sf)
                break
            else:
                bbox.bloat(self.epsilon)
                ppl_bbox = bbox.to_ppl()
                self.epsilon *= 2

    def gen_abs_dynamics(self, abs_domain):
        vertices = Polyhedron(*abs_domain.to_constraints()).vertices
        abs_domain_corners = np.array(vertices)
        abs_domain_centre = np.average(abs_domain_corners, axis=0)
        center_and_corners = np.append([abs_domain_centre], abs_domain_corners, axis=0)

        abs_domain_lower_bounds = abs_domain_corners.min(axis=0)
        abs_domain_upper_bounds = abs_domain_corners.max(axis=0)

        sampling_points = [(cc, evaluate_exp(self.nonlin_dyn, cc[0], cc[1])) for cc in center_and_corners]
        coeff_map = self.approx_non_linear_dyn(sampling_points)

        # matrix_A
        matrix_A = np.array(list(coeff_map[i] for i in range(self.dim)))

        u_max_array = []
        for i in range(matrix_A.shape[0]):
            # assuming 2 dimensions, can be easily generalised to n-dimension case
            affine_dynamic = str(matrix_A[i][0]) + '*x[0] + ' + str(matrix_A[i][1]) + '*x[1]'
            error_func_str = str(self.nonlin_dyn[i]) + '-(' + affine_dynamic + ')'
            error_func = pyibex.Function("x[%d]" % self.dim, error_func_str)

            xy = pyibex.IntervalVector(
                [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)])
            u_max_temp = error_func.eval(xy)
            u_max = max(u_max_temp[0], u_max_temp[1])
            u_max_array.extend([u_max] * 2)

        col_vec = np.array(u_max_array)

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
        return []

    # abs_dynamics, delta_tp, tau, alpha, beta, prev_directions, sf_current
    def compute_next_image(self):
        next_image, next_directions = self.post_opt.compute_next(abs_dynamics=self.abs_dynamics,
                                                                 delta_tp=self.reach_params.delta_tp,
                                                                 tau=self.tau,
                                                                 alpha=self.reach_params.alpha,
                                                                 beta=self.reach_params.beta,
                                                                 prev_directions=self.directions,
                                                                 prev_s_list=self.current_image_col_vec)
        self.set_current_image(next_directions, next_image)

    def set_abs_dynamics(self, matrix_A, poly_U):
        abs_dynamics = SysDynamics(dim=self.dim,
                                   init_coeff_matrix_X0=self.current_image_coff_mat,
                                   init_col_vec_X0=self.current_image_col_vec,
                                   dynamics_matrix_A=matrix_A,
                                   dynamics_matrix_B=self.coeff_matrix_B,
                                   dynamics_coeff_matrix_U=poly_U[0],
                                   dynamics_col_vec_U=poly_U[1])
        self.abs_dynamics = abs_dynamics

    def set_abs_domain(self, abs_domain):
        self.abs_domain = abs_domain

    def set_current_image(self, init_matrix_X0, init_col_vec_X0):
        self.current_image_coff_mat = init_matrix_X0
        self.current_image_col_vec = init_col_vec_X0
