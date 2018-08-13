import numpy as np

import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox, hyperbox_contain_by_bounds
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.Linearizer import Linearizer
from SysDynamics import AffineDynamics
from timerutil import Timers
from utils.GlpkWrapper import GlpkWrapper


class reachParams:
    def __init__(self, alpha=None, beta=None, delta=None, tau=None):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.tau = tau


def get_canno_dir_indices(directions):
    ub_indices = []
    lb_indices = []

    for idx, d in enumerate(directions):
        if np.isclose(np.sum(d), 1):  # due to the way of generating directions, exact equality is not proper.
            ub_indices.append(idx)
        elif np.isclose(np.sum(d), -1):
            lb_indices.append(idx)
        else:
            continue

    return lb_indices, ub_indices


def extract_bounds_from_sf(sf_vec, canno_dir_indices):
    lb_indices = canno_dir_indices[0]
    ub_indices = canno_dir_indices[1]

    lb = -sf_vec[lb_indices]
    ub = sf_vec[ub_indices]

    return lb, ub


class NonlinPostOpt:
    def __init__(self, dim, nonlin_dyn, time_horizon, tau, directions,
                 init_mat_X0, init_col_X0, is_linear, start_epsilon):
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
        self.canno_dir_indices = get_canno_dir_indices(directions)

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

        input_lb_seq = init_set_lb
        input_ub_seq = init_set_ub

        # B := \bb(X0)
        bbox = HyperBox(init_poly.vertices)
        # (A, V) := L(f, B), such that f(x) = (A, V) over-approx. g(x)
        bbox.bloat(1e-6)
        current_input_lb, current_input_ub = self.hybridize(bbox)

        phi_list = []

        i = 0
        last_alpha_iter = 0

        # sf_mat = np.zeros((time_frames, self.directions.shape[0]))
        bound_mat = np.zeros((time_frames, 2*self.dim))
        sf_mat = np.zeros((time_frames, self.directions.shape[0]))

        flag = True  # whether we have a new abstraction domain
        epsilon = self.start_epsilon

        # Timers.tic('total')
        while i < time_frames:
            if flag:
                # P_{i+1} := \alpha(X_{i})
                temp_tube_lb, temp_tube_ub = self.compute_alpha_step(current_init_set_lb,
                                                                     current_init_set_ub,
                                                                     current_input_lb,
                                                                     current_input_ub)
                last_alpha_iter = i
            else:
                temp_tube_lb, temp_tube_ub = self.compute_beta_step(tube_lb, tube_ub,
                                                                    input_lb_seq, input_ub_seq,
                                                                    phi_list, i, last_alpha_iter)
                # temp_tube_lb, temp_tube_ub = extract_bounds_from_sf(temp_tube_sf, self.canno_dir_indices)

                # temp_tube_lb, temp_tube_ub = self.compute_beta_step(tube_lb, tube_ub,
                #                                                     input_lb_seq, input_ub_seq,
                #                                                     phi_list, i, last_alpha_iter)

            # if P_{i+1} \subset B
            if hyperbox_contain_by_bounds(self.abs_domain.bounds, [temp_tube_lb, temp_tube_ub]):
                tube_lb, tube_ub = temp_tube_lb, temp_tube_ub
                # print('input_lb: {}, input_ub: {}'.format(current_input_lb, current_input_ub))

                phi_list = self.update_phi_list(phi_list)
                input_lb_seq, input_ub_seq = self.update_wb_seq(input_lb_seq, input_ub_seq,
                                                                current_input_lb, current_input_ub)

                next_init_sf = self.compute_gamma_step(input_lb_seq, input_ub_seq, phi_list)
                next_init_set_lb, next_init_set_ub = extract_bounds_from_sf(next_init_sf, self.canno_dir_indices)
                # initial reachable set in discrete time
                current_init_set_lb, current_init_set_ub = next_init_set_lb, next_init_set_ub

#                 sf_mat[i] = np.append(next_init_set_lb, next_init_set_ub)
                bound_mat[i] = np.append(tube_lb, tube_ub)
                sf_mat[i] = next_init_sf


                i += 1
                if i % 100 == 0:
                    print(i)

                flag = False
                # if self.start_epsilon > epsilon/2:
                #     epsilon = self.start_epsilon
                #     print('start eps')
                # else:
                #     epsilon = epsilon / 2
                #     print('32')

                epsilon = self.start_epsilon
            else:
                bbox = self.refine_domain(tube_lb, tube_ub, temp_tube_lb, temp_tube_ub)
                bbox.bloat(epsilon)

                # Timers.tic('hybridize')
                current_input_lb, current_input_ub = self.hybridize(bbox)
                # Timers.toc('hybridize')
                epsilon *= 2
                flag = True

        return sf_mat, bound_mat

    def hybridize(self, bbox):
        matrix_A, poly_U = self.dyn_linearizer.gen_abs_dynamics(abs_domain=bbox)

        self.set_abs_dynamics(matrix_A, poly_U)
        # 15.9%
        self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau, self.lp_solver)
        # 6.8 %
        self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau, self.lp_solver)
        # 13.2 %
        self.reach_params.delta = SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau)

        self.set_abs_domain(bbox)
        self.poly_U = Polyhedron(self.abs_dynamics.coeff_matrix_U, self.abs_dynamics.col_vec_U)

        vertices = self.poly_U.vertices

        err_lb = np.amin(vertices, axis=0)
        err_ub = np.amax(vertices, axis=0)

        return err_lb, err_ub

        # def compute_alpha_step(self, current_init_sf, Vi_lb, Vi_ub):
        #     # Ω0 = CH(Xi, e^Aτ · Xi ⊕ τV ⊕ α_τ·B)
        #     Xi_bounds = np.array([Xi_lb, Xi_ub]).T
        #     Vi_bounds = np.array([Vi_lb, Vi_ub]).T
        #     ball_bounds = np.array([[-1, 1]] * self.dim)

        # def compute_alpha_step_on_all_directions(self, Xi_lb, Xi_ub, Vi_lb, Vi_ub):
        #     delta_T = self.reach_params.delta.T
        #     delta_T_l = delta_T.dot(self.directions.T)
        #
        #     signs_delta_T_l = np.where(delta_T_l > 0, 1, 0)  # put 1 for positive elem, 0 for non-positive ones.
        #     signs_l = np.where(self.directions.T > 0, 1, 0)
        #
        #     Xi_bounds = np.array([Xi_lb, Xi_ub]).T
        #     Vi_bounds = np.array([Vi_lb, Vi_ub]).T
        #     ball_bounds = np.array([[-1, 1]] * self.dim)
        #
        #     # could be optimized?
        #     for i in range(self.dim):
        #         optm_Xi_index = signs_delta_T_l[i]
        #         optm_Vi_index = signs_l[i]
        #         if i == 0:
        #             optm_Xi = Xi_bounds[i][optm_Xi_index]
        #             optm_Vi = Vi_bounds[i][optm_Vi_index]
        #             optm_ball = ball_bounds[i][optm_Vi_index]
        #         else:
        #             optm_Xi = np.vstack((optm_Xi, Xi_bounds[i][optm_Xi_index]))
        #             optm_Vi = np.vstack((optm_Vi, Vi_bounds[i][optm_Vi_index]))
        #             optm_ball = np.vstack((optm_ball, ball_bounds[i][optm_Vi_index]))

        # Seems to be even slower (because small dimensionality?)
        # optm_Xi = Xi_bounds[np.arange(signs_delta_T_l.shape[0])[:, np.newaxis], signs_delta_T_l]
        # optm_Vi = Vi_bounds[np.arange(signs_l.shape[0])[:, np.newaxis], signs_l]
        # optm_ball = ball_bounds[np.arange(signs_l.shape[0])[:, np.newaxis], signs_l]

        # e^Aτ · Xi ⊕ τV ⊕ α_τ·B
        # temp_sf = np.einsum('ij,ij->j', delta_T_l, optm_Xi) \
        #           + self.tau * np.einsum('ij,ij->j', self.directions.T, optm_Vi) \
        #           + self.reach_params.alpha * np.einsum('ij,ij->j', self.directions.T, optm_ball)
        # Xi_sf = np.einsum('ij,ij->j', self.directions.T, optm_Xi)
        # sf = np.amax([temp_sf, Xi_sf], axis=0)
        #
        # return sf

    def compute_alpha_step(self, Xi_lb, Xi_ub, Vi_lb, Vi_ub):
        """
        When we have a new abstraction domain, the dynamic changes. To
        preserve the conservativeness, we cannot simply repeat beta step
        on the dense time reachable tube. Instead, we need to take the
        discrete time reachable SET at the time point when the domain
        is constructed (e.g. t_new), and then do alpha step to bloat it
        to the dense time tube indicating reachable states between
        time interval [t_new, t_new + τ].

        Ω0 = CH(Xi, e^Aτ · Xi ⊕ τV ⊕ α_τ·B)

        :param Xi_lb: lower bounds of discrete reachable set at t_new in the current (i-th) domain
        :param Xi_ub: upper bounds of discrete reachable set at t_new in the current (i-th) domain
        :param Vi_lb: lower bounds of input set in the current (i-th) domain (linearization error)
        :param Vi_ub: upper bounds of input set in the current (i-th) domain (linearization error)
        :return:
        """
        reach_tube_lb = np.empty(self.dim)
        reach_tube_ub = np.empty(self.dim)

        # input and bloated term W_α = τV ⊕ α_τ·B,
        # using inf norm, B is a square of width 2 at origin
        W_alpha_lb = Vi_lb * self.tau - self.reach_params.alpha
        W_alpha_ub = Vi_ub * self.tau + self.reach_params.alpha

        factors = self.reach_params.delta
        for j in range(factors.shape[0]):
            row = factors[j, :]

            pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)

            # e^Aτ · Xi ⊕ τV ⊕ α_τ·B
            maxval = pos_clip.dot(Xi_ub) + neg_clip.dot(Xi_lb) + W_alpha_ub[j]
            minval = neg_clip.dot(Xi_ub) + pos_clip.dot(Xi_lb) + W_alpha_lb[j]

            reach_tube_lb[j] = minval
            reach_tube_ub[j] = maxval

        # Ω0 = CH(Xi, e^Aτ · X ⊕ τV ⊕ α_τ·B)
        reach_tube_lb = np.amin([Xi_lb, reach_tube_lb], axis=0)
        reach_tube_ub = np.amax([Xi_ub, reach_tube_ub], axis=0)

        return reach_tube_lb, reach_tube_ub

    # tube_lb, tube_ub,
    # input_lb_seq, input_ub_seq,
    # phi_list, i
    def compute_beta_step(self, omega_lb, omega_ub, input_lb_seq, input_ub_seq, phi_list, i, last_alpha_iter):
        """
        As long as the continuous post would stay within the current abstraction domain,
        we could propagate the tube using beta step.

        The reccurrency relation:
           Ω_{i} = e^Aτ · Ω_{i-1} ⊕ τV_{i-1} ⊕ β_{i-1}·B
                  is unfolded as
           Ω_{i} = Φ_{n} ... Φ_{1} Ω0
                   ⊕ Φ_{n} ... Φ_{2} W_{1}
                   ⊕ Φ_{n} ... Φ_{3} W_{2}
                   ⊕ ...
                   ⊕ Φ_{n} W_{n-1}
                   ⊕ W_{n},
        where W_{i} = τV_{i} ⊕ β_{i}·τ·B

        Notice that Ω0 is the initial reach tube in the current domain.
        Correspondingly, Φ_{1} should be the first mat exp in the current domain.

        :param omega_lb:
        :param omega_ub:
        :return:
        """

        res_lb = np.empty(self.dim)
        res_ub = np.empty(self.dim)

        # as we care only about the current domain
        offset = last_alpha_iter - i - 1
        sub_phi_list = phi_list[offset:]
        input_lb_seq = np.hstack((omega_lb, input_lb_seq[self.dim * (offset + 1):]))
        input_ub_seq = np.hstack((omega_ub, input_ub_seq[self.dim * (offset + 1):]))

        # print(len(sub_phi_list), len(input_lb_seq), len(input_ub_seq))

        factors = sub_phi_list.transpose(1, 0, 2).reshape(2, -1)
        for j in range(factors.shape[0]):
            row = factors[j, :]

            pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(input_ub_seq) + neg_clip.dot(input_lb_seq)
            minval = neg_clip.dot(input_ub_seq) + pos_clip.dot(input_lb_seq)

            res_lb[j] = minval
            res_ub[j] = maxval

        return res_lb, res_ub

    def compute_gamma_step(self, input_lb_seq, input_ub_seq, phi_list):
        """
        Compute X_i from X0. The sequence of X is the discrete time
        reachable set at particular time points. Namely, X_i is the
        set of reachable states at i * τ time.

        The reccurrency relation:
           X_{i} = e^Aτ · X_{i-1} ⊕ τV_{i-1} ⊕ β_{i-1}·B
                  is unfolded as
           X_{n} = Φ_{n} ... Φ_{1} X0
                   ⊕ Φ_{n} ... Φ_{2} W_{1}
                   ⊕ Φ_{n} ... Φ_{3} W_{2}
                   ⊕ ...
                   ⊕ Φ_{n} W_{n-1}
                   ⊕ W_{n},
        where W_{i} = τV_{i} ⊕ β_{i}·B

        :param input_ub_seq: upper bounds of the sequence {X0, W_{1}, ..., W_{n-1}, W_{n}}
        :param input_lb_seq: lower bounds of the sequence {X0, W_{1}, ..., W_{n-1}, W_{n}}
        :param phi_list:   [Φ_{n} Φ_{n-1} … Φ_{1},
                            Φ_{n} Φ_{n-1} … Φ_{2},
                            Φ_{n} Φ_{n-1} … Φ_{3},
                            ...,
                            Φ_{n} Φ_{n-1},
                            Φ_{n}]
        :return: res_lb: lower bounds of X_i
                 res_ub: upper bounds of X_i
        """

        sf_vec = np.empty(self.directions.shape[0])

        input_bounds = np.array([input_lb_seq, input_ub_seq]).T

        # print('phi_list: {}'.format(phi_list))
        for idx, l in enumerate(self.directions):
            delta_T_l = phi_list.dot(l).reshape(-1, 1)
            signs_delta_T_l = np.where(delta_T_l > 0, 1, 0)

            optm_input = input_bounds[np.arange(signs_delta_T_l.shape[0])[:, np.newaxis], signs_delta_T_l]
            sf_val = np.einsum('ij,ij->j', delta_T_l, optm_input)

            # optm_input = np.empty(signs_delta_T_l.shape)
            # for idx_, elem in enumerate(signs_delta_T_l):
            #     if elem == 0:
            #         optm_input[idx_] = input_bounds[idx_][0]
            #     else:
            #         optm_input[idx_] = input_bounds[idx_][1]

            # sf_val = np.dot(delta_T_l, optm_input)
            #
            # print('direction: {}'.format(l))
            # print('input bounds: {}'.format(input_bounds))
            # print('delta_T_l: {}'.format(delta_T_l))
            # print('optm_input: {}'.format(optm_input))
            # print('sf_val: {}'.format(sf_val))
            # print('\n')
            sf_vec[idx] = sf_val

        # ================ DEPRECATED ===================== #
        # res_lb = np.empty(self.dim)
        # res_ub = np.empty(self.dim)
        #
        # factors = phi_list.transpose(1, 0, 2).reshape(2, -1)
        #
        # print('factors: {}'.format(factors))
        # for j in range(factors.shape[0]):
        #     row = factors[j, :]
        #
        #     pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
        #     neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)
        #
        #     maxval = pos_clip.dot(input_ub_seq) + neg_clip.dot(input_lb_seq)
        #     minval = neg_clip.dot(input_ub_seq) + pos_clip.dot(input_lb_seq)
        #
        #     res_lb[j] = minval
        #     res_ub[j] = maxval
        # print('\n')
        # print(res_lb, res_ub)
        # print(sf_vec)

        return sf_vec

    # res_lb = np.empty(self.dim)
    # res_ub = np.empty(self.dim)
    #
    # factors = phi_list.transpose(1, 0, 2).reshape(2, -1)
    # for j in range(factors.shape[0]):
    #     row = factors[j, :]
    #
    #     pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
    #     neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)
    #
    #     maxval = pos_clip.dot(input_ub_seq) + neg_clip.dot(input_lb_seq)
    #     minval = neg_clip.dot(input_ub_seq) + pos_clip.dot(input_lb_seq)
    #
    #     res_lb[j] = minval
    #     res_ub[j] = maxval
    #
    # return res_lb, res_ub



    def update_phi_list(self, phi_list):
        """
        phi_list contains the product of delta_transpose.
        After n-times update, phi_list looks like this:
        [ Φ_{1}^T … Φ_{n-1}^T Φ_{n}^T, Φ_{2}^T … Φ_{n}^T, ..., Φ_{n-1}^T Φ_{n}^T, Φ_{n}^T]
        """
        dyn_coeff_mat = self.abs_dynamics.get_dyn_coeff_matrix_A()
        delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, self.tau)
        delta_T = delta.T

        if len(phi_list) == 0:
            phi_list = np.array([delta_T])
        else:
            phi_list = np.tensordot(phi_list, delta_T, axes=(2, 0))
        phi_list = np.vstack((phi_list, [np.eye(self.dim)]))

        return phi_list

    def update_wb_seq(self, lb, ub, next_lb, next_ub):
        """
         W_{i} = τV_{i} ⊕ β_{i}·B
        """
        next_lb = next_lb * self.tau - self.reach_params.beta
        next_ub = next_ub * self.tau + self.reach_params.beta

        return np.append(lb, next_lb), np.append(ub, next_ub)
        # return np.vstack((lb, next_lb)), np.vstack((ub, next_ub))

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
            ret.append(Polyhedron(directions, sf_row))
            # exit()
        return ret
