import numpy as np

import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox, hyperbox_contain_by_bounds
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.PostOptStateholder import PostOptStateholder
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
                 pseudo_var, scaling_per, scaling_cutoff, id_to_vars):
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
        self.scaling_per = scaling_per
        self.scaling_cutoff = scaling_cutoff

        # if the bound is larger than this, give up to avoid any further numeric issue in libs.
        self.max_tolerance = 1e5

        self.id_to_vars = id_to_vars

        # the following attributes would be updated along the flowpipe construction
        self.abs_dynamics = None
        self.abs_domain = None
        self.poly_U = None
        self.reach_params = reachParams()

        self.posh = None

        self.scaled_nonlin_dyn = None
        self.tau_d = None

        self.lp_solver = GlpkWrapper(dim)
        self.dyn_linearizer = Linearizer(dim, nonlin_dyn, is_linear)
        self.pseudo_var = pseudo_var

        self.pseudo_dim = self.dim
        self.lp_solver_on_pseudo_dim = GlpkWrapper(self.dim)

    def compute_post(self):

        Timers.tic('total')
        time_frames = int(np.ceil(self.time_horizon / self.tau))

        init_poly = Polyhedron(self.init_coeff, self.init_col)
        vertices = init_poly.get_vertices()

        init_set_lb = np.amin(vertices, axis=0)
        init_set_ub = np.amax(vertices, axis=0)

        self.posh = PostOptStateholder(init_set_lb, init_set_ub)
        # ======
        next_init_sf = self.compute_gamma_step(init_set_lb, init_set_ub, np.identity(self.dim))
        # ======

        # B := \bb(X0)
        bbox = HyperBox(init_poly.vertices)
        # (A, V) := L(f, B), such that f(x) = (A, V) over-approx. g(x)
        bbox.bloat(1e-6)
        epsilon = self.start_epsilon
        i = 0
        # current_input_lb, current_input_ub = self.hybridize(bbox)
        ct = 0
        current_vol = 1e10

        # use_time_scaling = True
        use_time_scaling = True
        scaled = False

        total_walltime = 0
        start_walltime = time.time()

        # sf_mat = np.zeros((time_frames + sum(dwell_steps), self.template_directions.shape[0]))
        sf_mat = []

        while i < time_frames:
            bbox = self.refine_domain(self.posh.tube_lb.get_val(),
                                      self.posh.tube_ub.get_val(),
                                      self.posh.temp_tube_lb.get_val(),
                                      self.posh.temp_tube_ub.get_val()
                                      )
            bbox.bloat(epsilon)

            Timers.tic('hybridize')
            current_input_lb, current_input_ub = self.hybridize(bbox)
            Timers.toc('hybridize')
            epsilon *= 2

            Timers.tic('compute_alpha')
            sf_tube = self.compute_alpha_step(self.posh.input_lb_seq.get_val(),
                                              self.posh.input_ub_seq.get_val(),
                                              current_input_lb,
                                              current_input_ub,
                                              self.posh.phi_list.get_val(),
                                              next_init_sf)
            Timers.toc('compute_alpha')
            alpha_bounds = extract_bounds_from_sf(sf_tube, self.canno_dir_indices)
            self.posh.temp_tube_lb.set_val(alpha_bounds[0])
            self.posh.temp_tube_ub.set_val(alpha_bounds[1])

            if any(np.abs(self.posh.temp_tube_lb.get_val()) >= self.max_tolerance) or \
                    any(np.abs(self.posh.temp_tube_ub.get_val()) >= self.max_tolerance):
                print('Computation not completed after {} iterations. Abort now.'.format(i))
                break

            # if P_{i+1} \subset B
            if hyperbox_contain_by_bounds(self.abs_domain.bounds,
                                          [self.posh.temp_tube_lb.get_val(), self.posh.temp_tube_ub.get_val()]):
                self.posh.tube_lb.set_val(self.posh.temp_tube_lb.get_val())
                self.posh.tube_ub.set_val(self.posh.temp_tube_ub.get_val())

                prev_vol = current_vol
                current_vol = self.compute_vol(self.posh.tube_lb.get_val(), self.posh.tube_ub.get_val())

                self.posh.phi_list.set_val(self.update_phi_list(self.posh.phi_list.get_val()))
                res_update_wb = self.update_wb_seq(self.posh.input_lb_seq.get_val(),
                                                   self.posh.input_ub_seq.get_val(),
                                                   current_input_lb,
                                                   current_input_ub
                                                   )
                self.posh.input_lb_seq.set_val(res_update_wb[0])
                self.posh.input_ub_seq.set_val(res_update_wb[1])

                Timers.tic('compute gamma')
                next_init_sf = self.compute_gamma_step(self.posh.input_lb_seq.get_val(),
                                                       self.posh.input_ub_seq.get_val(),
                                                       self.posh.phi_list.get_val())
                Timers.toc('compute gamma')
                next_init_set_lb, next_init_set_ub = extract_bounds_from_sf(next_init_sf, self.canno_dir_indices)

                # initial reachable set in discrete time
                self.posh.current_init_set_lb.set_val(next_init_set_lb)
                self.posh.current_init_set_ub.set_val(next_init_set_ub)

                # print(self.abs_dynamics.matrix_A)
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
                    if scaled:
                        imprv_rate = (prev_vol - current_vol)/prev_vol
                        stop_scaling = imprv_rate < self.scaling_cutoff
                        # print('{}%'.format(imprv_rate*100))
                        # stop_scaling = current_vol > prev_vol
                        if stop_scaling:
                            self.dyn_linearizer.set_nonlin_dyn(self.nonlin_dyn)
                            self.dyn_linearizer.is_scaled = False
                            scaled = False

                            # rollbacks to the previous state
                            self.posh.rollback()

                            # print('stopped at {} scaling steps'.format(ct))
                            ct = 0
                            i -= 1
                        else:
                            time_frames += 1
                            ct += 1
                            # sf_mat.append(next_init_sf)
                            sf_mat.append(sf_tube)
                    else:
                        # check whether to do dynamic scaling at the current step
                        # sf_mat.append(next_init_sf)
                        sf_mat.append(sf_tube)
                        scaling_stepsize = max(int(time_frames * self.scaling_per), 1)
                        start_scaling = i % scaling_stepsize == 0
                        if start_scaling:
                            # print('start time scaling at step {}'.format(i))
                            scaling_config = self.get_scaling_configs(self.posh.tube_lb.get_val(), self.posh.tube_ub.get_val())
                            self.scaled_nonlin_dyn = self.scale_dynamics(*scaling_config)
                            self.dyn_linearizer.set_nonlin_dyn(self.scaled_nonlin_dyn)
                            self.dyn_linearizer.is_scaled = True
                            scaled = True
                else:
                    # sf_mat.append(next_init_sf)
                    sf_mat.append(sf_tube)
                epsilon /= 4
        print('Completed flowpipe computation in {:.2f} secs.\n'.format(total_walltime))
        Timers.toc('total')
        Timers.print_stats()

        return np.array(sf_mat)

    def hybridize(self, bbox):
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

    # def compute_alpha_step(self, Xi_lb, Xi_ub, Vi_lb, Vi_ub):
    #     """
    #     When we have a new abstraction domain, the dynamic changes. To
    #     preserve the conservativeness, we cannot simply repeat beta step
    #     on the dense time reachable tube. Instead, we need to take the
    #     discrete time reachable SET at the time point when the domain
    #     is constructed (e.g. t_new), and then do alpha step to bloat it
    #     to the dense time tube indicating reachable states between
    #     time interval [t_new, t_new + τ].
    #
    #     Ω0 = CH(Xi, e^Aτ · Xi ⊕ τV ⊕ α_τ·B)
    #
    #     :param Xi_lb: lower bounds of discrete reachable set at t_new in the current (i-th) domain
    #     :param Xi_ub: upper bounds of discrete reachable set at t_new in the current (i-th) domain
    #     :param Vi_lb: lower bounds of input set in the current (i-th) domain (linearization error)
    #     :param Vi_ub: upper bounds of input set in the current (i-th) domain (linearization error)
    #     :return:
    #     """
    #     reach_tube_lb = np.empty(self.pseudo_dim)
    #     reach_tube_ub = np.empty(self.pseudo_dim)
    #
    #     # input and bloated term W_α = τV ⊕ α_τ·B,
    #     # using inf norm, B is a square of width 2 at origin
    #     W_alpha_lb = Vi_lb * self.tau - self.reach_params.alpha
    #     W_alpha_ub = Vi_ub * self.tau + self.reach_params.alpha
    #
    #     factors = self.reach_params.delta
    #     for j in range(factors.shape[0]):
    #         row = factors[j, :]
    #
    #         pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
    #         neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)
    #
    #         # e^Aτ · Xi ⊕ τV ⊕ α_τ·B
    #         maxval = pos_clip.dot(Xi_ub) + neg_clip.dot(Xi_lb) + W_alpha_ub[j]
    #         minval = neg_clip.dot(Xi_ub) + pos_clip.dot(Xi_lb) + W_alpha_lb[j]
    #
    #         reach_tube_lb[j] = minval
    #         reach_tube_ub[j] = maxval
    #
    #     # Ω0 = CH(Xi, e^Aτ · X ⊕ τV ⊕ α_τ·B)
    #     reach_tube_lb = np.amin([Xi_lb, reach_tube_lb], axis=0)
    #     reach_tube_ub = np.amax([Xi_ub, reach_tube_ub], axis=0)
    #
    #     return reach_tube_lb, reach_tube_ub

    def compute_alpha_step(self, input_lb_seq, input_ub_seq, Vi_lb, Vi_ub, phi_list, sf_X0):
        W_alpha_lb = Vi_lb * self.tau - self.reach_params.alpha
        W_alpha_ub = Vi_ub * self.tau + self.reach_params.alpha

        next_lb_seq = np.hstack((input_lb_seq, W_alpha_lb))
        next_ub_seq = np.hstack((input_ub_seq, W_alpha_ub))

        delta_T = self.reach_params.delta.T

        if len(phi_list) == 0:
            phi_list = np.array([delta_T])
        else:
            phi_list = np.tensordot(phi_list, delta_T, axes=(2, 0))
        phi_list = np.vstack((phi_list, [np.eye(self.pseudo_dim)]))

        sf_vec = np.empty(self.template_directions.shape[0])
        for idx, l in enumerate(self.template_directions):
            delta_T_l = phi_list.dot(l).reshape(1, -1)
            pos_clip = np.clip(a=delta_T_l, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=delta_T_l, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(next_ub_seq) + neg_clip.dot(next_lb_seq)
            sf_vec[idx] = maxval

        return np.maximum(sf_vec, sf_X0)

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

        for idx, l in enumerate(self.template_directions):
            # '''
            # Multiply each phi product in phi_list with the direction, then get a list of new directions.
            # Reshape it to a column vector. The number of rows corresponds to length of input sequence.
            #
            # E.g. Given
            #          phi_list = [ [1, 2,  [-1, 0,
            #                        3, 4],  0, -1] ]
            #          direction l = [1, 0]
            #          input_lb_seq = [[1, 2], [3, 4]]
            #          input_ub_seq = [[5, 6], [7, 8]]
            #
            #      1) phi_list.dot(l) gives:
            #      [ [1,   [-1,
            #         3],   0] ]
            #
            #      2) Reshaping it gives:
            #      delta_T_l =
            #      [ 1
            #        3,
            #        -1,
            #        0 ]
            #
            #      3) If the element is positive, to maximize x\cdot l, we take the upper bound of input. Otherwise, lower bound.
            #      signs_delta_T_l =
            #      [ 1
            #        1,
            #        0,
            #        0 ]
            #
            #      4) input_bounds zips input_lb_seq and input_ub_seq
            #      [  [1, 5],
            #         [2, 6],
            #         [3, 7],
            #         [4, 8] ]
            #
            #      5) index input_bounds by signs_delta_T_l, we get
            #      [ 5,      (index 1 from [1, 5])
            #        6,      (index 1 from [2, 6])
            #        3,      (index 0 from [3, 7])
            #        4 ]     (index 0 from [4, 8])
            #
            #      6) Now (5, 6, 3, 4) is the vector maximizing x \cdot l
            #      einsum computes the inner product of (5, 6, 3, 4) and delta_T_l.
            #      This is same as first reshape them into row vectors and then take the dot.
            # '''

            # delta_T_l = phi_list.dot(l).reshape(-1, 1)
            # signs_delta_T_l = np.where(delta_T_l > 0, 1, 0)
            #
            # optm_input = input_bounds[np.arange(signs_delta_T_l.shape[0])[:, np.newaxis], signs_delta_T_l]
            #
            # sf_val = np.einsum('ij,ij->j', delta_T_l, optm_input)
            # sf_vec[idx] = sf_val

            delta_T_l = phi_list.dot(l).reshape(1, -1)
            pos_clip = np.clip(a=delta_T_l, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=delta_T_l, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(input_ub_seq) + neg_clip.dot(input_lb_seq)
            sf_vec[idx] = maxval

        return sf_vec

    def update_phi_list(self, phi_list):
        """
        phi_list contains the product of delta_transpose.
        After n-times update, phi_list looks like this:
        [ Φ_{n}^T Φ_{n-1}^T … Φ_{1}^T, Φ_{n-1}^T … Φ_{1}^T, ..., Φ_{1}^T]
        """
        delta_T = self.reach_params.delta.T

        if len(phi_list) == 0:
            phi_list = np.array([delta_T])
        else:
            phi_list = np.tensordot(phi_list, delta_T, axes=(2, 0))
        phi_list = np.vstack((phi_list, [np.eye(self.pseudo_dim)]))

        return phi_list

    def update_wb_seq(self, lb_seq, ub_seq, next_lb, next_ub):
        """
         W_{i} = τV_{i} ⊕ τd ⊕ β_{i}·B, where d = (1/τ) * \int_{0}^{τ}[(e^(τ-s)A-I)c]ds.
         c is the center of the boxed uncertainty region.
        """
        c = (next_lb + next_ub) / 2
        A = self.abs_dynamics.get_dyn_coeff_matrix_A()
        M = SuppFuncUtils.mat_exp_int(A, t_min=0, t_max=self.tau)
        self.tau_d = np.dot(M, c)

        next_lb = next_lb * self.tau - self.reach_params.beta + self.tau_d
        next_ub = next_ub * self.tau + self.reach_params.beta + self.tau_d

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

        init_lb = self.posh.current_init_set_lb.get_val()
        init_ub = self.posh.current_init_set_ub.get_val()

        init_col = np.vstack((init_ub, -init_lb)).T.reshape(1, -1).flatten()

        abs_dynamics = AffineDynamics(dim=self.dim,
                                      init_coeff_matrix_X0=self.init_coeff,
                                      init_col_vec_X0=init_col,
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

    def get_scaling_configs(self, tube_lb, tube_ub):
        # domain center
        c = np.sum(self.abs_domain.bounds, axis=0) / 2

        # compute derivative
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

        return norm_vec, p, c

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
        # return sum(widths)

    def scale_dynamics(self, norm_vec, p, c):
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

        temp_scaled_dynamics = []
        for dyn in self.nonlin_dyn.dynamics:
            temp_scaled_dynamics.append('({})*({})'.format(scaling_func_str, dyn))

        # =====
        Timers.tic('compute m')
        temp_gd = GeneralDynamics(self.id_to_vars, *temp_scaled_dynamics)
        gd_norm = np.linalg.norm(temp_gd.eval_jacobian(c), np.inf)
        norm = np.linalg.norm(self.abs_dynamics.matrix_A, np.inf)
        m = norm / gd_norm
        scaled_dynamics = []
        for dyn in temp_scaled_dynamics:
            scaled_dynamics.append('{}*({})'.format(m, dyn))
        Timers.toc('compute m')
        # =====

        # return temp_gd

        return GeneralDynamics(self.id_to_vars, *scaled_dynamics)

if __name__ == '__main__':
    nlpost = NonlinPostOpt

    # test get_pedal_point()
    main_tube_lb = np.array([0, 0])
    main_tube_ub = np.array([1, 1])
    l = np.array([1, 0])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, main_tube_lb, main_tube_ub), [1, 1])
    l = np.array([1, 1])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, main_tube_lb, main_tube_ub), [1, 1])
    l = np.array([0.5, 1])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, main_tube_lb, main_tube_ub), [1, 1])
    l = np.array([-0.5, 1])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, main_tube_lb, main_tube_ub), [0, 1])
    l = np.array([-0.5, -0.5])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, main_tube_lb, main_tube_ub), [0, 0])
    l = np.array([0.5, -0.5])
    np.testing.assert_almost_equal(nlpost.get_pedal_point(l, main_tube_lb, main_tube_ub), [1, 0])