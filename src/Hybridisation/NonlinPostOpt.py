import numpy as np

import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox, hyperbox_contain_by_bounds
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.TrackedVar import TrackedVar as tvar
from Hybridisation.Linearizer import Linearizer
from SysDynamics import AffineDynamics, GeneralDynamics
from timerutil import Timers
from utils.GlpkWrapper import GlpkWrapper
import time


class reachParams:
    def __init__(self, alpha=None, beta=None, delta=None, tau=None):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.tau = tau

# class stateHolder:


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
    def __init__(self, dim, nonlin_dyn, time_horizon, tau, init_coeff, init_col, is_linear, directions, start_epsilon,
                 pseudo_var, id_to_vars):
        self.dim = dim
        self.nonlin_dyn = nonlin_dyn
        self.time_horizon = time_horizon
        self.tau = tau
        self.coeff_matrix_B = np.identity(dim)
        self.is_linear = is_linear
        self.start_epsilon = start_epsilon
        self.init_coeff = init_coeff
        self.init_col = init_col
        self.template_directions = directions
        self.canno_dir_indices = get_canno_dir_indices(directions)
        self.pseudo_var = pseudo_var

        # if the bound is larger than this, give up to avoid any further numeric issue in libs.
        self.max_tolerance = 1e3

        self.id_to_vars = id_to_vars

        # the following attributes would be updated along the flowpipe construction
        self.abs_dynamics = None
        self.abs_domain = None
        self.poly_U = None
        self.reach_params = reachParams()

        self.scaled_nonlin_dyn = None

        self.lp_solver = GlpkWrapper(dim)
        self.dyn_linearizer = Linearizer(dim, nonlin_dyn, is_linear)
        self.pseudo_var = pseudo_var

        # add initialization for psedo-variable
        if self.pseudo_var:
            self.pseudo_dim = self.dim + 1

            self.init_coeff = np.hstack((self.init_coeff, np.zeros(shape=(self.init_coeff.shape[0], 1))))

            temp_coeff_pos = np.zeros(self.pseudo_dim)
            temp_coeff_pos[self.pseudo_dim - 1] = 1
            temp_col_pos = np.ones(1)

            temp_coeff_neg = np.zeros(self.pseudo_dim)
            temp_coeff_neg[self.pseudo_dim - 1] = -1
            temp_col_neg = -np.ones(1)

            self.init_coeff = np.vstack((self.init_coeff, temp_coeff_neg, temp_coeff_pos))
            self.init_col = np.vstack((self.init_col, temp_col_neg, temp_col_pos))

            self.lp_solver_on_pseudo_dim = GlpkWrapper(self.pseudo_dim)

            # add a zero column for all directions
            self.template_directions = np.hstack(
                (self.template_directions, np.zeros((self.template_directions.shape[0], 1))))
        else:
            self.pseudo_dim = self.dim
            self.lp_solver_on_pseudo_dim = GlpkWrapper(self.dim)

    def compute_post(self):

        Timers.tic('total')
        time_frames = int(np.ceil(self.time_horizon / self.tau))
        tvars = []

        init_poly = Polyhedron(self.init_coeff, self.init_col)
        vertices = init_poly.get_vertices()

        init_set_lb = tvar(np.amin(vertices, axis=0))
        init_set_ub = tvar(np.amax(vertices, axis=0))
        tvars.append(init_set_lb)
        tvars.append(init_set_ub)

        # init_set_lb = np.amin(vertices, axis=0)
        # init_set_ub = np.amax(vertices, axis=0)

        # reachable states in dense time
        tube_lb = tvar(init_set_lb.get_val())
        tube_ub = tvar(init_set_ub.get_val())
        tvars.append(tube_lb)
        tvars.append(tube_ub)
        # tube_lb, tube_ub = init_set_lb, init_set_ub

        # temporary variables for reachable states in dense time
        temp_tube_lb = tvar(init_set_lb.get_val())
        temp_tube_ub = tvar(init_set_ub.get_val())
        tvars.append(temp_tube_lb)
        tvars.append(temp_tube_ub)

        # temp_tube_lb, temp_tube_ub = init_set_lb, init_set_ub
        # initial reachable set in discrete time in the current abstract domain
        # changes when the abstract domain is large enough to contain next image in alfa step
        current_init_set_lb = tvar(init_set_lb.get_val())
        current_init_set_ub = tvar(init_set_ub.get_val())
        tvars.append(current_init_set_lb)
        tvars.append(current_init_set_ub)

        # current_init_set_lb, current_init_set_ub = init_set_lb, init_set_ub

        input_lb_seq = tvar(init_set_lb.get_val())
        input_ub_seq = tvar(init_set_ub.get_val())
        tvars.append(input_lb_seq)
        tvars.append(input_ub_seq)
        # input_lb_seq, input_ub_seq = init_set_lb, init_set_ub

        phi_list = tvar([])
        tvars.append(phi_list)
        # phi_list = []

        # B := \bb(X0)
        bbox = HyperBox(init_poly.vertices)
        # (A, V) := L(f, B), such that f(x) = (A, V) over-approx. g(x)
        bbox.bloat(1e-6)
        epsilon = self.start_epsilon
        i = 0
        # current_input_lb, current_input_ub = self.hybridize(bbox)
        ct = 0
        current_vol = 1e10

        use_time_scaling = True
        if use_time_scaling:
            scaled = False
            # # vanderpol. time step = 0.01, d=0.2
            dwell_from = [200]#, 650]
            # dwell_steps = [100, 100]

            # vanderpol, time step = 0.03, small init [1.25, 1.55], [2.28, 2.32]
            # dwell_from = [65, 210]
            # dwell_steps = [60, 40]
            # d = [0.2, 0.2]

            # # vanderpol. time step = 0.005, larger init: [1, 1.5], [2, 2.45]
            # dwell_from = [300]  ##, 1300]
            # dwell_steps = [500]  #, 100]
            # d = [0.5]  #, 0.4]

            # vanderpol. time step = 0.01, d=0.2
            # dwell_from = [200, 650]
            # dwell_steps = [100, 100]
            # d = [0.1, 0.1]

            # coupled vanderpol, time step = 0.005
            # dwell_from = [400, 1150]
            # dwell_steps = [200, 200]
            # d = [0.2, 0.2]

            # coupled vanderpol, time step = 0.01
            # dwell_from = [220, 650]
            # dwell_steps = [10, 50]
            # d = [0.1, 0.1]

            # brusselator, time step = 0.01 (scale times 20)
            # dwell_from = [200]
            # dwell_steps = [50]
            # d = [0.3]

            # buckling_column. time step = 0.01
            # dwell_from = [50, 200, 700, 1100]
            # dwell_steps = [100, 100, 100, 100]
            # d = [0.2, 0.2, 0.2, 0.2]

            # dwell_from = [50, 200, 700]
            # dwell_steps = [100, 100, 100]

            # predator-prey (not working)
            # dwell_from = [550]
            # dwell_steps = [50]
            # d = [0.5]

            # lac operon
            # dwell_from = [50]
            # dwell_steps = [50]
            # d = [5]

            # pbt (wrong)
            # dwell_from = [300, 1400, 1700]
            # dwell_steps = [400, 200, 1000]
            # d = [0.3, 0.3, 0.3]

            # lorentz
            # dwell_from = [300]
            # dwell_steps = [20]
            # d = [0.3]

        else:
            dwell_from = []
            dwell_steps = [0]

        total_walltime = 0
        start_walltime = time.time()

        # sf_mat = np.zeros((time_frames + sum(dwell_steps), self.template_directions.shape[0]))
        sf_mat = []

        while i < time_frames:
            bbox = self.refine_domain(tube_lb.get_val(),
                                      tube_ub.get_val(),
                                      temp_tube_lb.get_val(),
                                      temp_tube_ub.get_val()
                                      )
            bbox.bloat(epsilon)

            Timers.tic('hybridize')
            current_input_lb, current_input_ub = self.hybridize(bbox)
            Timers.toc('hybridize')
            epsilon *= 2

            # P_{i+1} := \alpha(X_{i})
            res_alpha_step = self.compute_alpha_step(current_init_set_lb.get_val(),
                                                     current_init_set_ub.get_val(),
                                                     current_input_lb,
                                                     current_input_ub)
            temp_tube_lb.set_val(res_alpha_step[0])
            temp_tube_ub.set_val(res_alpha_step[1])

            if any(np.abs(temp_tube_lb.get_val()) >= self.max_tolerance) or \
                    any(np.abs(temp_tube_ub.get_val()) >= self.max_tolerance):
                print('Computation not completed after {} iterations. Abort now.'.format(i))
                break

            # if P_{i+1} \subset B
            if hyperbox_contain_by_bounds(self.abs_domain.bounds, [temp_tube_lb.get_val(), temp_tube_ub.get_val()]):
                tube_lb.set_val(temp_tube_lb.get_val())
                tube_ub.set_val(temp_tube_ub.get_val())
                # tube_lb, tube_ub = temp_tube_lb, temp_tube_ub

                prev_vol = current_vol
                current_vol = self.compute_vol(tube_lb.get_val(), tube_ub.get_val())

                phi_list.set_val(self.update_phi_list(phi_list.get_val()))
                res_update_wb = self.update_wb_seq(input_lb_seq.get_val(),
                                                                input_ub_seq.get_val(),
                                                                current_input_lb,
                                                                current_input_ub)
                input_lb_seq.set_val(res_update_wb[0])
                input_ub_seq.set_val(res_update_wb[1])

                next_init_sf = self.compute_gamma_step(input_lb_seq.get_val(),
                                                       input_ub_seq.get_val(),
                                                       phi_list.get_val())
                next_init_set_lb, next_init_set_ub = extract_bounds_from_sf(next_init_sf, self.canno_dir_indices)
                # if self.pseudo_var:
                #     next_init_set_lb = np.hstack((next_init_set_lb, 1))
                #     next_init_set_ub = np.hstack((next_init_set_ub, 1))

                # initial reachable set in discrete time
                current_init_set_lb.set_val(next_init_set_lb)
                current_init_set_ub.set_val(next_init_set_ub)

                #                 sf_mat[i] = np.append(next_init_set_lb, next_init_set_ub)
                # sf_mat[i] = next_init_sf
                sf_mat.append(next_init_sf)
                # epsilon = self.start_epsilon
                epsilon /= 4

                i += 1
                if i % 100 == 0:
                    now = time.time()
                    walltime_elapsed = now - start_walltime
                    total_walltime += walltime_elapsed
                    print('{} / {} steps ({:.2f}%) completed in {:.2f} secs. '
                          'Total time elapsed: {:.2f} secs'.format(i, time_frames, 100 * i / time_frames, walltime_elapsed,
                                                                   total_walltime))
                    start_walltime = now

                if use_time_scaling:
                    flag_scaling = i in dwell_from
                    if flag_scaling:
                        print('start time scaling at step {}'.format(i))
                        scaling_config = self.get_scaling_configs(tube_lb.get_val(), tube_ub.get_val())
                        self.scaled_nonlin_dyn = self.scale_dynamics(*scaling_config)
                        self.dyn_linearizer.set_nonlin_dyn(self.scaled_nonlin_dyn)
                        self.dyn_linearizer.is_scaled = True
                        scaled = True

                    if scaled:
                        time_frames += 1
                        stop_scaling = current_vol > prev_vol and not flag_scaling
                        if stop_scaling:
                            self.dyn_linearizer.set_nonlin_dyn(self.nonlin_dyn)
                            self.dyn_linearizer.is_scaled = False
                            scaled = False

                            print('stopped at {} scaling steps'.format(ct))
                            ct = 0
                        else:
                            ct += 1
                            # print(ct)

        print('Completed flowpipe computation in {:.2f} secs.\n'.format(total_walltime))
        Timers.toc('total')
        Timers.print_stats()

        return np.array(sf_mat)

    def hybridize(self, bbox):
        if self.pseudo_var:
            domain_bounds = bbox.bounds[:, :-1]
        else:
            domain_bounds = bbox.bounds

        Timers.tic('gen_abs_dynamics')
        matrix_A, poly_w, c = self.dyn_linearizer.gen_abs_dynamics(abs_domain_bounds=domain_bounds)
        Timers.toc('gen_abs_dynamics')

        # todo computing bloating factors can avoid calling LP
        self.set_abs_dynamics(matrix_A, poly_w, c)
        # 15.9%
        self.reach_params.alpha = SuppFuncUtils.compute_alpha(self.abs_dynamics, self.tau, self.lp_solver_on_pseudo_dim)
        # 6.8 %
        # self.reach_params.beta = SuppFuncUtils.compute_beta(self.abs_dynamics, self.tau, self.lp_solver_on_pseudo_dim)
        self.reach_params.beta = SuppFuncUtils.compute_beta_no_offset(self.abs_dynamics, self.tau)
        # 13.2 %
        self.reach_params.delta = SuppFuncUtils.mat_exp(self.abs_dynamics.matrix_A, self.tau)

        self.set_abs_domain(bbox)
        self.poly_U = Polyhedron(self.abs_dynamics.coeff_matrix_U, self.abs_dynamics.col_vec_U)

        # todo This can be optimized using a hyperbox rather a polyhedron.
        Timers.tic('get_vertices')

        vertices = self.poly_U.get_vertices()
        Timers.toc('get_vertices')

        err_lb = np.amin(vertices, axis=0)
        err_ub = np.amax(vertices, axis=0)

        return err_lb, err_ub

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
        reach_tube_lb = np.empty(self.pseudo_dim)
        reach_tube_ub = np.empty(self.pseudo_dim)

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

        if self.pseudo_var:
            reach_tube_lb[self.pseudo_dim - 1] = 1
            reach_tube_ub[self.pseudo_dim - 1] = 1

        return reach_tube_lb, reach_tube_ub

    # def compute_beta_step(self, omega_lb, omega_ub, input_lb_seq, input_ub_seq, phi_list, i, last_alpha_iter):
    #     """
    #     As long as the continuous post would stay within the current abstraction domain,
    #     we could propagate the tube using beta step.
    #
    #     The reccurrency relation:
    #        Ω_{i} = e^Aτ · Ω_{i-1} ⊕ τV_{i-1} ⊕ β_{i-1}·B
    #               is unfolded as
    #        Ω_{i} = Φ_{n} ... Φ_{1} Ω0
    #                ⊕ Φ_{n} ... Φ_{2} W_{1}
    #                ⊕ Φ_{n} ... Φ_{3} W_{2}
    #                ⊕ ...
    #                ⊕ Φ_{n} W_{n-1}
    #                ⊕ W_{n},
    #     where W_{i} = τV_{i} ⊕ β_{i}·τ·B
    #
    #     Notice that Ω0 is the initial reach tube in the current domain.
    #     Correspondingly, Φ_{1} should be the first mat exp in the current domain.
    #
    #     :param omega_lb:
    #     :param omega_ub:
    #     :return:
    #     """
    #
    #     res_lb = np.empty(self.pseudo_dim)
    #     res_ub = np.empty(self.pseudo_dim)
    #
    #     # as we care only about the current domain
    #     offset = last_alpha_iter - i - 1
    #     sub_phi_list = phi_list[offset:]
    #     input_lb_seq = np.hstack((omega_lb, input_lb_seq[self.dim * (offset + 1):]))
    #     input_ub_seq = np.hstack((omega_ub, input_ub_seq[self.dim * (offset + 1):]))
    #
    #     # print(len(sub_phi_list), len(input_lb_seq), len(input_ub_seq))
    #
    #     factors = sub_phi_list.transpose(1, 0, 2).reshape(2, -1)
    #     for j in range(factors.shape[0]):
    #         row = factors[j, :]
    #
    #         pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
    #         neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)
    #
    #         maxval = pos_clip.dot(input_ub_seq) + neg_clip.dot(input_lb_seq)
    #         minval = neg_clip.dot(input_ub_seq) + pos_clip.dot(input_lb_seq)
    #
    #         res_lb[j] = minval
    #         res_ub[j] = maxval
    #
    #     res_lb[self.pseudo_dim - 1] = 1
    #     res_ub[self.pseudo_dim - 1] = 1
    #
    #     return res_lb, res_ub

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

        sf_vec = np.empty(self.template_directions.shape[0])

        input_bounds = np.array([input_lb_seq, input_ub_seq]).T

        for idx, l in enumerate(self.template_directions):
            '''
            Multiply each phi product in phi_list with the direction, then get a list of new directions.
            Reshape it to a column vector. The number of rows corresponds to length of input sequence.

            E.g. Given
                     phi_list = [ [1, 2,  [-1, 0,
                                   3, 4],  0, -1] ]
                     direction l = [1, 0]
                     input_lb_seq = [[1, 2], [3, 4]]
                     input_ub_seq = [[5, 6], [7, 8]]

                 1) phi_list.dot(l) gives:
                 [ [1,   [-1,
                    3],   0] ]

                 2) Reshaping it gives:
                 delta_T_l =
                 [ 1
                   3,
                   -1,
                   0 ]

                 3) If the element is positive, to maximize x\cdot l, we take the upper bound of input. Otherwise, lower bound.
                 signs_delta_T_l =
                 [ 1
                   1,
                   0,
                   0 ]

                 4) input_bounds zips input_lb_seq and input_ub_seq
                 [  [1, 5],
                    [2, 6],
                    [3, 7],
                    [4, 8] ]

                 5) index input_bounds by signs_delta_T_l, we get
                 [ 5,      (index 1 from [1, 5])
                   6,      (index 1 from [2, 6])
                   3,      (index 0 from [3, 7])
                   4 ]     (index 0 from [4, 8])

                 6) Now (5, 6, 3, 4) is the vector maximizing x \cdot l
                 einsum computes the inner product of (5, 6, 3, 4) and delta_T_l.
                 This is same as first reshape them into row vectors and then take the dot.
            '''

            delta_T_l = phi_list.dot(l).reshape(-1, 1)
            signs_delta_T_l = np.where(delta_T_l > 0, 1, 0)

            optm_input = input_bounds[np.arange(signs_delta_T_l.shape[0])[:, np.newaxis], signs_delta_T_l]

            sf_val = np.einsum('ij,ij->j', delta_T_l, optm_input)
            sf_vec[idx] = sf_val

        return sf_vec

    def update_phi_list(self, phi_list):
        """
        phi_list contains the product of delta_transpose.
        After n-times update, phi_list looks like this:
        [ Φ_{n}^T Φ_{n-1}^T … Φ_{1}^T, Φ_{n-1}^T … Φ_{1}^T, ..., Φ_{1}^T]
        """
        dyn_coeff_mat = self.abs_dynamics.get_dyn_coeff_matrix_A()
        delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, self.tau)
        delta_T = delta.T

        # print('delta_T: {}'.format(delta_T))
        # print('original phi_list: {}'.format(phi_list))

        if len(phi_list) == 0:
            phi_list = np.array([delta_T])
        else:
            phi_list = np.tensordot(phi_list, delta_T, axes=(2, 0))
        phi_list = np.vstack((phi_list, [np.eye(self.pseudo_dim)]))

        # print('after tensordot phi_list: {}'.format(phi_list))
        # print('\n')
        return phi_list

    # def update_wb_seq(self, lb, ub, next_lb, next_ub):
    #     """
    #      W_{i} = τV_{i} ⊕ β_{i}·B
    #     """
    #     next_lb = next_lb * self.tau - self.reach_params.beta
    #     next_ub = next_ub * self.tau + self.reach_params.beta
    #
    #     return np.append(lb, next_lb), np.append(ub, next_ub)
        # return np.vstack((lb, next_lb)), np.vstack((ub, next_ub))

    def update_wb_seq(self, lb_seq, ub_seq, next_lb, next_ub):
        """
         W_{i} = τV_{i} ⊕ τd ⊕ β_{i}·B, where d = (1/τ) * \int_{0}^{τ}[(e^(τ-s)A-I)c]ds.
         c is the center of the boxed uncertainty region.
        """
        c = (next_lb + next_ub) / 2
        A = self.abs_dynamics.get_dyn_coeff_matrix_A()
        M = SuppFuncUtils.mat_exp_int(A, t_min=0, t_max=self.tau)
        tau_d = np.dot(M, c)

        next_lb = next_lb * self.tau - self.reach_params.beta + tau_d
        next_ub = next_ub * self.tau + self.reach_params.beta + tau_d

        return np.append(lb_seq, next_lb), np.append(ub_seq, next_ub)

    def set_abs_dynamics(self, matrix_A, poly_w, c):
        if self.pseudo_var:
            matrix_A = np.hstack((matrix_A, c.reshape(len(c), 1)))
            matrix_A = np.vstack((matrix_A, np.zeros(shape=(1, self.pseudo_dim))))

            coeff_matrix_B = np.identity(n=self.pseudo_dim)

            pos_iden = np.identity(self.pseudo_dim)
            neg_iden = -np.identity(self.pseudo_dim)
            w_mat = np.hstack((pos_iden, neg_iden)).reshape(self.pseudo_dim * 2, -1)
            w_vec = np.vstack((poly_w[1], 0, 0))
            poly_w = (w_mat, w_vec)
        else:
            u = poly_w[1]
            b = np.dstack((c, -c)).reshape((self.dim * 2, -1))
            poly_w = (poly_w[0], u + b)
            coeff_matrix_B = self.coeff_matrix_B

        abs_dynamics = AffineDynamics(dim=self.dim,
                                      init_coeff_matrix_X0=self.init_coeff,
                                      init_col_vec_X0=self.init_col,
                                      dynamics_matrix_A=matrix_A,
                                      dynamics_matrix_B=coeff_matrix_B,
                                      dynamics_coeff_matrix_U=poly_w[0],
                                      dynamics_col_vec_U=poly_w[1])
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
        # " sloppy implementation, change later on"
        #
        # directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        # ret = []
        #
        # for sf_row in sf_mat:
        #     sf_row = np.multiply(sf_row, [-1, -1, 1, 1]).reshape(sf_row.shape[0], 1)
        #     ret.append(Polyhedron(directions, sf_row))
        #     # exit()
        # return ret

        " sloppy implementation, change later on"
        directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        ret = []

        for sf_row in sf_mat:
            sf_row = sf_row.reshape(2, -1)[:, 0:-1]
            sf_row = sf_row.reshape(1, -1).flatten()
            sf_row = np.multiply(sf_row, [-1, -1, 1, 1]).reshape(sf_row.shape[0], 1)
            ret.append(Polyhedron(directions, sf_row))
        return ret

    def get_scaling_configs(self, tube_lb, tube_ub):
        # domain center
        c = np.sum(self.abs_domain.bounds, axis=0) / 2

        # compute derivative
        print(self.nonlin_dyn.str_rep)
        deriv = self.nonlin_dyn.eval(c)

        # get norm vector
        norm = np.dot(deriv, deriv) ** 0.5
        norm_vec = deriv / norm

        # get the minimal distance between the domain center and the reachable set boundary
        # along the normal vector direction.
        p = self.get_pedal_point(norm_vec, tube_lb, tube_ub)

        with open('../out/sca_cent.out', 'a') as opfile:
            opfile.write(' '.join(str(elem) for elem in c.tolist()) + '\n')

        with open('../out/pivots.out', 'a') as opfile:
            opfile.write(' '.join(str(elem) for elem in p.tolist()) + '\n')

        return norm_vec, p

    @staticmethod
    def get_pedal_point(l, tube_lb, tube_ub):
        pos_clip = np.where(l >= 0, 1, 0)
        neg_clip = np.where(l < 0, 1, 0)

        pp = pos_clip * tube_ub + neg_clip * tube_lb
        return pp

    @staticmethod
    def compute_vol(tube_lb, tube_ub):
        widths = tube_ub - tube_lb
        return np.prod(widths)

    def scale_dynamics(self, norm_vec, p):
        """
        1. Compute center of the abstraction domain;
        2. Evaluate the derivative of the center as the normal vector (v) direction;
        3. Decide on a hyperline, such that
           1) perpendicular to v;
           2) some distance ahead; such distance should be far enough (otherwise a part
              of the image would have already crossed the surface while another part
              is left behind??)
        """
        # # scaling function -(a/||a||) \cdot x + b
        # p = domain_center + np.dot(norm_vec, d)
        b = np.dot(norm_vec, p)

        # a = norm_vec / norm
        a_prime = [-elem for elem in norm_vec]

        scaling_func_str = ''
        for idx, elem in enumerate(a_prime):
            scaling_func_str += '{}*x{}+'.format(elem, idx)
        scaling_func_str = '{}+{}'.format(scaling_func_str, b)

        scaled_dynamics = []
        for dyn in self.nonlin_dyn.dynamics:
            scaled_dynamics.append('({})*({})'.format(scaling_func_str, dyn))

        return GeneralDynamics(self.id_to_vars, *scaled_dynamics)

if __name__ == '__main__':
    nlpost = NonlinPostOpt

    # test get_pedal_point()
    tube_lb = np.array([0, 0])
    tube_ub = np.array([1, 1])
    l = np.array([1, 0])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, tube_lb, tube_ub), [1, 1])
    l = np.array([1, 1])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, tube_lb, tube_ub), [1, 1])
    l = np.array([0.5, 1])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, tube_lb, tube_ub), [1, 1])
    l = np.array([-0.5, 1])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, tube_lb, tube_ub), [0, 1])
    l = np.array([-0.5, -0.5])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, tube_lb, tube_ub), [0, 0])
    l = np.array([0.5, -0.5])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, tube_lb, tube_ub), [1, 0])