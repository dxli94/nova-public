import numpy as np

import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox, hyperbox_contain_by_bounds
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.Linearizer import Linearizer
from SysDynamics import AffineDynamics
from utils.GlpkWrapper import GlpkWrapper


class reachParams:
    def __init__(self, alpha=None, beta=None, delta_tp=None, tau=None):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta_tp
        self.tau = tau


class NonlinPostOpt:
    def __init__(self, dim, nonlin_dyn, time_horizon, tau, directions, init_mat_X0, init_col_X0, is_linear,
                 start_epsilon):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.time_horizon = time_horizon
        self.tau = tau
        self.coeff_matrix_B = np.identity(dim)
        self.is_linear = is_linear
        self.start_epsilon = start_epsilon
        self.init_mat_X0 = init_mat_X0
        self.init_col_X0 = init_col_X0
        self.directions = directions

        # the following attributes would be updated along the flowpipe construction
        self.abs_dynamics = None
        self.abs_domain = None
        self.poly_U = None
        self.reach_params = reachParams()

        # self.lin_post_opt = PostOperator()
        self.lp_solver = GlpkWrapper(dim)
        self.dyn_linearizer = Linearizer(dim, nonlin_dyn, is_linear)

    def compute_post(self):
        time_frames = int(np.ceil(self.time_horizon / self.tau))
        init_poly = Polyhedron(self.init_mat_X0, self.init_col_X0)

        vertices = init_poly.vertices
        init_set_ub = np.amax(vertices, axis=0)
        init_set_lb = np.amin(vertices, axis=0)

        # reachable states in dense time
        tube_ub = init_set_ub
        tube_lb = init_set_lb
        # initial reachable set in discrete time in the current abstract domain
        # changes when the abstract domain is large enough to contain next image in alfa step
        current_init_set_ub = init_set_ub
        current_init_set_lb = init_set_lb
        # initial reachable set in discrete time in the next abstract domain
        # changes when dynamics is changed
        next_init_set_ub = init_set_ub
        next_init_set_lb = init_set_lb

        input_lb_seq = init_set_lb
        input_ub_seq = init_set_ub

        # B := \bb(X0)
        bbox = HyperBox(init_poly.vertices)
        # (A, V) := L(f, B), such that f(x) = (A, V) over-approx. g(x)
        current_input_lb, current_input_ub = self.hybridize(bbox, 1e-6)
        # input_lb_seq, input_ub_seq = self.update_input_bounds_seq(input_ub_seq, input_lb_seq,
        #                                                           current_input_ub, current_input_lb)

        delta_list = []

        i = 0

        sf_mat = np.zeros((time_frames, 2*self.dim))

        flag = True  # whether we have a new abstraction domain
        isalpha = False
        epsilon = self.start_epsilon
        # delta_product = 1
        # delta_product_list_without_first_one = [1]

        while i < time_frames:
            if flag:
                # P_{i+1} := \alpha(X_{i})
                temp_tube_lb, temp_tube_ub = self.compute_alpha_step(current_init_set_lb,
                                                                     current_init_set_ub,
                                                                     current_input_lb,
                                                                     current_input_ub)
                isalpha = True
            else:
                # P_{i+1} := \beta(P_{i})
                temp_tube_lb, temp_tube_ub = self.compute_beta_step(tube_lb, tube_ub,
                                                                    current_input_lb,
                                                                    current_input_ub)

            # if P_{i+1} \subset B
            if hyperbox_contain_by_bounds(self.abs_domain.bounds, [temp_tube_lb, temp_tube_ub]):
                tube_lb, tube_ub = temp_tube_lb, temp_tube_ub
                sf_mat[i] = np.append(tube_lb, tube_ub)

                if isalpha:
                    current_init_set_lb, current_init_set_ub = next_init_set_lb, next_init_set_ub
                    isalpha = False

                delta_list = self.update_delta_list(delta_list)
                input_lb_seq, input_ub_seq = self.update_input_bounds(input_ub_seq, input_lb_seq,
                                                                      current_input_ub, current_input_lb)
                next_init_set_lb, next_init_set_ub = self.compute_gamma_step(input_ub_seq, input_lb_seq,
                                                                             delta_list)

                i += 1
                if i % 100 == 0:
                    print(i)

                flag = False
                epsilon = self.start_epsilon
            else:
                bbox = self.refine_domain(tube_lb, tube_ub, temp_tube_lb, temp_tube_lb)

                current_input_lb, current_input_ub = self.hybridize(bbox, epsilon)
                epsilon *= 2
                flag = True

        return sf_mat

    def hybridize(self, bbox, starting_epsilon):
        bbox.bloat(starting_epsilon)
        matrix_A, poly_U = self.dyn_linearizer.gen_abs_dynamics(abs_domain=bbox)

        self.set_abs_dynamics(matrix_A, poly_U)
        self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau, self.lp_solver)
        self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau, self.lp_solver)
        self.reach_params.delta = SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau)
        self.set_abs_domain(bbox)

        self.poly_U = Polyhedron(self.abs_dynamics.coeff_matrix_U, self.abs_dynamics.col_vec_U)

        vertices = self.poly_U.vertices
        err_lb = np.amin(vertices, axis=0)
        err_ub = np.amax(vertices, axis=0)

        return err_lb, err_ub

    def compute_alpha_step(self, init_lb, init_ub, input_lb, input_ub):
        reach_tube_lb = np.empty(self.dim)
        reach_tube_ub = np.empty(self.dim)

        # input and bloated term W_α = τV ⊕ α_τ·B,
        # using inf norm, B is a square of width 2 at origin
        input_lb = input_lb * self.tau - self.reach_params.alpha
        input_ub = input_ub * self.tau + self.reach_params.alpha

        factors = self.reach_params.delta
        for j in range(factors.shape[0]):
            row = factors[j, :]

            pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)

            # e^At · X ⊕ τV ⊕ α_τ·B
            maxval = pos_clip.dot(init_ub) + neg_clip.dot(init_lb) + input_ub[j]
            minval = neg_clip.dot(init_ub) + pos_clip.dot(init_lb) + input_lb[j]

            reach_tube_lb[j] = minval
            reach_tube_ub[j] = maxval

        # Ω0 = CH(X0, e^At · X ⊕ τV ⊕ α_τ·B)
        reach_tube_lb = np.amin([init_lb, reach_tube_lb], axis=0)
        reach_tube_ub = np.amax([init_ub, reach_tube_ub], axis=0)

        return reach_tube_lb, reach_tube_ub

    def compute_beta_step(self, reach_tube_lb, reach_tube_ub, input_lb, input_ub):
        # Ω_{i+1} = e^At · Ω_{i} ⊕ τV ⊕ β_τ·B
        res_lb = np.empty(self.dim)
        res_ub = np.empty(self.dim)

        # input and bloated term W_β = τV ⊕ β_τ·B
        input_lb = input_lb * self.tau - self.reach_params.beta
        input_ub = input_ub * self.tau + self.reach_params.beta

        factors = self.reach_params.delta
        for j in range(factors.shape[0]):
            row = factors[j, :]

            pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)

            # e^At · Ω_{i} ⊕ τV ⊕ α_τ·B
            maxval = pos_clip.dot(reach_tube_ub) + neg_clip.dot(reach_tube_lb) + input_ub[j]
            minval = neg_clip.dot(reach_tube_ub) + pos_clip.dot(reach_tube_lb) + input_lb[j]

            res_lb[j] = minval
            res_ub[j] = maxval

        return res_lb, res_ub

    def compute_gamma_step(self, input_ub_seq, input_lb_seq, delta_list):
        res_lb = np.empty(self.dim)
        res_ub = np.empty(self.dim)

        factors = delta_list.transpose(1, 0, 2).reshape(2, -1)
        for j in range(factors.shape[0]):
            row = factors[j, :]

            pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(input_ub_seq) + neg_clip.dot(input_lb_seq)
            minval = neg_clip.dot(input_ub_seq) + pos_clip.dot(input_lb_seq)

            res_lb[j] = minval
            res_ub[j] = maxval

        return res_lb, res_ub

    def update_delta_list(self, delta_list):
        dyn_coeff_mat = self.abs_dynamics.get_dyn_coeff_matrix_A()
        delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, self.tau)

        if len(delta_list) == 0:
            delta_list = np.array([delta])
        else:
            delta_list = np.tensordot(delta, delta_list, axes=((1), (1))).swapaxes(0, 1)
        delta_list = np.vstack((delta_list, [np.eye(self.dim)]))

        return delta_list

    def update_input_bounds(self, ub, lb, next_ub, next_lb):
        return np.append(lb, next_lb), np.append(ub, next_ub)

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

    def refine_domain(self, tube_lb, tube_ub, temp_tube_lb, temp_tube_ub):
        bbox_lb = np.amin([tube_lb, temp_tube_lb], axis=0)
        bbox_ub = np.amax([tube_ub, temp_tube_ub], axis=0)
        bbox = HyperBox([bbox_lb, bbox_ub])

        return bbox

    def update_input_bounds_seq(self, ub, lb, next_ub, next_lb):
        return np.append(lb, next_lb), np.append(ub, next_ub)

    def get_projections(self, directions, opdims, sf_mat):
        " sloppy implementation, change later on"

        directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        ret = []

        for sf_row in sf_mat:
            sf_row = np.multiply(sf_row, [-1, -1, 1, 1]).reshape(sf_row.shape[0], 1)

            print(sf_row)
            ret.append(Polyhedron(directions, sf_row))
        return ret
