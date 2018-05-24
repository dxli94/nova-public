import numpy as np

import SuppFuncUtils
from AffinePostOpt import PostOperator
from ConvexSet.HyperBox import HyperBox, hyperbox_contain
from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from Hybridisation.Linearizer import Linearizer
from SysDynamics import AffineDynamics
from utils.GlpkWrapper import GlpkWrapper


def compute_support_functions_for_polyhedra(poly, directions, lp):
    vec = np.array([poly.compute_support_function(l, lp) for l in directions])
    return vec.reshape(len(vec), 1)


class reachParams:
    def __init__(self, alpha=None, beta=None, delta_tp=None, tau=None):
        self.alpha = alpha
        self.beta = beta
        self.delta_tp = delta_tp
        self.tau = tau


class NonlinPostOpt:
    def __init__(self, dim, nonlin_dyn, time_horizon, tau, directions, init_mat_X0, init_col_X0, is_linear, start_epsilon):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.time_horizon = time_horizon
        self.tau = tau
        self.coeff_matrix_B = np.identity(dim)
        self.is_linear = is_linear
        self.start_epsilon = start_epsilon

        # the following attributes would be updated along the flowpipe construction
        self.directions = directions
        self.abs_dynamics = None
        self.abs_domain = None
        self.init_poly = None
        self.trans_poly_U = None
        self.init_mat_X0 = init_mat_X0
        self.init_col_X0 = init_col_X0
        self.reach_params = reachParams()
        self.P = np.zeros(len(self.directions))
        self.P_temp = np.array([np.inf, -np.inf] * 2)
        self.X = np.zeros(len(self.directions))
        self.init_X = np.zeros(len(self.directions))
        self.init_X_in_each_domain = np.zeros(len(self.directions))

        self.lin_post_opt = PostOperator()
        self.lp_solver = GlpkWrapper(dim)
        self.dyn_linearizer = Linearizer(dim, nonlin_dyn, is_linear)

    def compute_post(self):
        time_frames = int(np.ceil(self.time_horizon / self.tau))
        init_poly = Polyhedron(self.init_mat_X0, self.init_col_X0)

        self.X = compute_support_functions_for_polyhedra(init_poly, self.directions, self.lp_solver)
        self.init_X = self.X
        self.init_X_in_each_domain = self.X
        self.init_poly = Polyhedron(self.directions, self.init_X)

        # B := \bb(X0)
        bbox = HyperBox(self.init_poly.vertices)
        # (A, V) := L(f, B), s.t. f(x) = (A, V) over-approx. g(x)
        self.hybridise(bbox, 1e-6)
        # P_{0} := \alpha(X_{0})
        self.P_temp = self.X
        self.P = self.X
        i = 0

        # initialise support function matrix, [r], [s]
        sf_mat = []
        bbox_mat = []
        x_mat = [self.X]

        s_on_each_direction = [0] * len(self.directions)
        r_on_each_direction = self.directions

        trans_poly_U_list = []
        beta_list = []

        flag = True  # whether we have a new abstraction domain
        isalpha = False
        epsilon = self.start_epsilon
        delta_product = 1
        delta_product_list_without_first_one = [1]

        while i < time_frames:
            if flag:
                # P_{i+1} := \alpha(X_{i})
                self.compute_alpha_step()
                s_temp = [0] * len(self.directions)
                r_temp = self.directions
                isalpha = True
            else:
                # P_{i+1} := \beta(P_{i})
                s_temp, r_temp = self.compute_beta_step(s_on_each_direction, r_on_each_direction)

            # if P_{i+1} \subset B
            # Todo P_temp is not a hyperbox rather an rotated rectangon. Checking the bounding box is sufficient but not necessary. Needs to be refine
            if hyperbox_contain(self.abs_domain.to_constraints()[1], self.P_temp):
                self.P = self.P_temp
                if i != 0:
                    temp = []
                    for elem in delta_product_list_without_first_one:
                        temp.append(np.dot(elem, self.reach_params.delta_tp))
                    temp.append(1)
                    delta_product_list_without_first_one = temp
                delta_product = np.dot(delta_product, self.reach_params.delta_tp)

                sf_mat.append(self.P)
                bbox_mat.append(bbox.to_constraints()[1])
                x_mat.append(self.X)

                if isalpha:
                    self.init_X_in_each_domain = self.X
                    isalpha = False

                self.compute_gamma_step(i, trans_poly_U_list, beta_list,
                                        delta_product, delta_product_list_without_first_one)

                s_on_each_direction, r_on_each_direction = s_temp, r_temp
                i += 1
                if i % 100 == 0:
                    print(i)

                flag = False
                epsilon = self.start_epsilon
            else:
                bbox = self.refine_domain()
                self.hybridise(bbox, epsilon)
                epsilon *= 2
                flag = True

        return sf_mat

    def hybridise(self, bbox, starting_epsilon):
        bbox.bloat(starting_epsilon)
        matrix_A, poly_U = self.dyn_linearizer.gen_abs_dynamics(abs_domain=bbox)

        self.set_abs_dynamics(matrix_A, poly_U)
        self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau, self.lp_solver)
        self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau, self.lp_solver)
        self.reach_params.delta_tp = np.transpose(SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau))
        self.set_abs_domain(bbox)

        self.trans_poly_U = TransPoly(self.abs_dynamics.matrix_B, self.abs_dynamics.coeff_matrix_U,
                                      self.abs_dynamics.col_vec_U)

    def compute_alpha_step(self):
        # wrap X_{i} on template directions
        poly = Polyhedron(self.directions, self.X)
        sf_arr = [self.lin_post_opt.compute_initial_sf(self.reach_params.delta_tp, poly, self.trans_poly_U, l,
                                                       self.reach_params.alpha, self.tau, self.lp_solver) for l in self.directions]
        self.P_temp = np.array(sf_arr)

    def compute_beta_step(self, s_vec, r_vec):
        sf_vec = []
        current_s_array = []
        current_r_array = []
        poly = Polyhedron(self.directions, self.init_X_in_each_domain)

        for idx in range(len(r_vec)):
            prev_r = r_vec[idx]
            prev_s = s_vec[idx]
            r = np.dot(self.reach_params.delta_tp, prev_r)
            s = prev_s + self.lin_post_opt.compute_sf_w(prev_r, self.trans_poly_U, self.reach_params.beta, self.tau, self.lp_solver)
            sf = s + self.lin_post_opt.compute_initial_sf(self.reach_params.delta_tp,
                                                          poly, self.trans_poly_U,
                                                          r,
                                                          self.reach_params.alpha,
                                                          self.tau,
                                                          self.lp_solver)
            sf_vec.append(sf)
            current_s_array.append(s)
            current_r_array.append(r)

        self.P_temp = np.array(sf_vec)
        return current_s_array, current_r_array

    def compute_gamma_step(self, n, trans_poly_U_list, beta_list, delta_product, delta_list_without_first_one):
        sf_vec = []
        next_s_arr = []

        trans_poly_U_list.append(self.trans_poly_U)
        beta_list.append(self.reach_params.beta)

        for l_idx, l in enumerate(self.directions):
            r = np.dot(delta_product, l)
            sf_X0 = self.init_poly.compute_support_function(r, self.lp_solver)

            # direction_matrix[l_idx].append(r)
            s = self.compute_sum_sf_w(n, l, delta_list_without_first_one, trans_poly_U_list, beta_list)

            sf = sf_X0 + s
            sf_vec.append([sf])
            next_s_arr.append(s)

        self.X = np.array(sf_vec)

        return next_s_arr

    def compute_sum_sf_w(self, n, l, delta_list_without_first_one, trans_poly_U_list, beta_list):
        sum = 0

        for i in range(0, n + 1):
            direction = np.dot(delta_list_without_first_one[i], l)
            trans_poly = trans_poly_U_list[i]
            beta = beta_list[i]
            sum += self.lin_post_opt.compute_sf_w(direction, trans_poly, beta, self.tau, self.lp_solver)

            # sum += self.post_opt.compute_sf_w(direction, trans_poly, 0, self.tau, lp)
        return sum

    def set_abs_dynamics(self, matrix_A, poly_U):
        abs_dynamics = AffineDynamics(dim=self.dim,
                                      init_coeff_matrix_X0=self.init_mat_X0,
                                      init_col_vec_X0=self.init_col_X0,
                                      dynamics_matrix_A=matrix_A,
                                      dynamics_matrix_B=self.coeff_matrix_B,
                                      dynamics_coeff_matrix_U=poly_U[0],
                                      dynamics_col_vec_U=poly_U[1])
        self.abs_dynamics = abs_dynamics

    def set_abs_domain(self, abs_domain):
        self.abs_domain = abs_domain

    def refine_domain(self):
        # take the convex hall of P_{i} and P_{i+1}
        bounds = np.maximum(self.P_temp.reshape(len(self.directions), 1),
                            self.P.reshape(len(self.directions), 1))

        vertices = Polyhedron(self.directions, bounds).vertices
        bbox = HyperBox(vertices)

        return bbox
