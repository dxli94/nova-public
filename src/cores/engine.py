import sys
import time

import numpy as np

from convex_set import hyperbox
from convex_set.hyperbox import HyperBox
from cores.hybrid_automata import NonlinHybridAutomaton
from cores.linearizer import Linearizer
from cores.post_opt.fwd_continuous_opt import FwdContinuousPostOpeartor
from cores.sys_dynamics import AffineDynamics, GeneralDynamics
from utils import simulator
from utils import suppfunc_utils
from utils.containers import AppSetting
from utils.plotter import Plotter


class NovaEngine:
    """
    main computation object, initialize and call run().
    """

    def __init__(self, ha, app_settings):

        assert isinstance(app_settings, AppSetting)
        assert isinstance(ha, NonlinHybridAutomaton)

        self._hybrid_automaton = ha
        self._settings = app_settings
        self._dim = len(ha.variables)

        self._linearizer = None
        self._abs_dynamics = None
        self._abs_domain = None

        self._post_operator_factory = 0
        self._post_operator = None

        self._cur_step = 0
        self._cur_mode = None

        self._reach_error = False

        if app_settings.reach.error_model == 1:
            self._post_operator_factory = FwdContinuousPostOpeartor
        else:
            raise ValueError("Error model type {} not understood.".format(app_settings.reach.error_model))

    def run(self, init_list):
        """
        Run the engine: reachability analysis; simulation and plotting.

        init is a list of (LinearAutomatonMode, HyperRectangle)

        """
        mode, continuous_set = init_list[0]  # [0] mode, [1] continuous set
        # ha = next_state[0].parent
        self._cur_mode = mode

        # 1. run reachability analysis
        # for now assuming we have a single initial set
        self._post_operator = self._post_operator_factory(continuous_set, self._settings.reach.directions)
        self._linearizer = Linearizer(self._dim, self._cur_mode.dynamics, self._cur_mode.is_linear)
        supp_matrix = self._run_reachability()

        # 2. run simulation
        simu_res = simulator.run_simulate(self._settings.simu.horizon,
                                          self._settings.simu.model_name,
                                          self._settings.simu.init_set)

        # 3. plotting
        Plotter.make_plot(self._dim, self._settings.reach.directions, supp_matrix,
                          self._settings.plot.model_name,
                          self._settings.plot.poly_dir_path, simu_res)

    def _run_reachability(self):
        """
        Main method for reachability analysis with time scaling.
        """

        # todo encapsulate into a timer
        total_walltime = 0
        start_walltime = time.time()

        # volume of reach tube
        cur_vol = sys.maxsize
        # True: in scaling mode; False: outside scaling mode
        in_scaling_mode = False

        self._post_operator.do_init_step()

        eps = 1e-5
        # hybridization domain
        dom = self._post_operator.init_state
        dom.bloat(eps)

        supp_matrix = []

        while not self._is_finished():
            dom = self._refine_domain()
            dom.bloat(eps)

            cur_input_lb, cur_input_ub = self._hybridize(dom)
            eps *= 2

            self._post_operator.do_alpha_step(cur_input_lb, cur_input_ub)

            if self._has_lost_precision():
                print('Computation not completed after {} iterations. Abort now.'.format(self._cur_step))
                break

            if self._domain_contains_tube():
                self._post_operator.proceed_state(cur_input_lb, cur_input_ub, self._abs_dynamics.a_matrix)
                self._cur_step += 1

                self._post_operator.do_gamma_step()

                prev_vol = cur_vol
                cur_vol = self._compute_vol()

                # todo encapsulate into a timer
                if self._cur_step % 100 == 0:
                    now = time.time()
                    walltime_elapsed = now - start_walltime
                    total_walltime += walltime_elapsed
                    print('{} / {} steps ({:.2f}%) completed in {:.2f} secs. '
                          'Total time elapsed: {:.2f} secs'.format(self._cur_step, self._settings.reach.num_steps, 100 * self._cur_step / self._settings.reach.num_steps, walltime_elapsed, total_walltime))
                    start_walltime = now

                if not in_scaling_mode:
                    supp_matrix.append(self._post_operator.tube_supp)

                    if self._is_start_scaling():
                        # entering scaling mode
                        scaled_dyn = self._make_scaling_dynamics()
                        # set scaled dynamics as the target dynamics within linearizer
                        self._linearizer.set_target_dyn(scaled_dyn)
                        self._linearizer.is_scaled = True

                        in_scaling_mode = True

                else:  # staying in the scaling mode
                    if self._is_scaling_helpful(prev_vol, cur_vol):
                        supp_matrix.append(self._post_operator.tube_supp)
                        # We can decrease self.cur_step instead. By increasing reach.num_steps,
                        # we can see the num of extra steps we computed in scaling mode more clearly.
                        self._settings.reach.num_steps += 1
                    else:  # scaling is deemed as un-helpful
                        # rolling-back to the previous state
                        self._post_operator.rollback()
                        self._cur_step -= 1

                        # set to original (un-scaled) dynamics in the curr mode
                        self._linearizer.set_target_dyn(self._cur_mode.dynamics)
                        self._linearizer.is_scaled = False

                        in_scaling_mode = False

                eps /= 4

        return supp_matrix

    def _is_finished(self):
        """
        Return true if 1) reach maximal computation steps; or 2) found unsafe state.
        """
        return self._cur_step >= self._settings.reach.num_steps or self._reach_error

    def _has_lost_precision(self):
        """
        Stop the computation if blows up.
        """
        max_tolerance = 1e5

        return self._post_operator.has_lost_precision(max_tolerance)

    def _domain_contains_tube(self):
        """
        Return true if the new reachable tube remains in the current abstraction domain.
        Otherwise False.
        """
        temp_tube_bounds = self._post_operator.get_temp_tube_bounds()
        return hyperbox.contains(self._abs_domain.bounds, temp_tube_bounds)

    def _refine_domain(self):
        """
        Enlarge the domain by taking hull of current and next reachable sets.
        """
        tube_lb, tube_ub = self._post_operator.get_tube_bounds()
        temp_tube_lb, temp_tube_ub = self._post_operator.get_temp_tube_bounds()

        bbox_lb = np.amin([tube_lb, temp_tube_lb], axis=0)
        bbox_ub = np.amax([tube_ub, temp_tube_ub], axis=0)
        bbox = HyperBox([bbox_lb, bbox_ub], opt=1)

        return bbox

    def _hybridize(self, dom):
        """
        Compute linearized dynamics and returns the linearization errors.
        """
        a_matrix, w_poly, c_col = self._linearizer.gen_abs_dynamics(dom.bounds)

        self._set_abs_dynamics(a_matrix, w_poly, c_col)
        reach_params = suppfunc_utils.compute_reach_params(self._abs_dynamics, self._settings.reach.stepsize)
        self._post_operator.update_reach_params(reach_params)

        self._abs_domain = dom

        vertices = HyperBox.get_vertices_from_constr(self._abs_dynamics.u_coeff, self._abs_dynamics.u_col)

        err_lb = np.amin(vertices, axis=0)
        err_ub = np.amax(vertices, axis=0)

        # with open('/home/dxli/Desktop/offset-beta.dat', 'a') as opfile:
        #     opfile.write(str(self.reach_params.beta) + '\n')
        # lp = GlpkWrapper(2)
        # with open('/home/dxli/Desktop/no-offset-beta.dat', 'a') as opfile:
        #     opfile.write(str(SuppFuncUtils.compute_beta(self.abs_dynamics, self._tau, lp)) + '\n')

        return err_lb, err_ub

    def _set_abs_dynamics(self, a_matrix, w_poly, c_col):
        """
        Set linearized dynamics.
        """
        u = w_poly[1]
        b = np.dstack((c_col, -c_col)).reshape((self._dim * 2, -1))
        w_poly = (w_poly[0], u + b)

        lb, ub = self._post_operator.get_cur_init_set()

        init_col = np.vstack((ub, -lb)).T.reshape(1, -1).flatten()

        abs_dynamics = AffineDynamics(dim=self._dim, a_matrix=a_matrix,
                                      u_coeff=w_poly[0], u_col=w_poly[1],
                                      x0_col=init_col)
        self._abs_dynamics = abs_dynamics

    def _compute_vol(self):
        """
        Compute the volume of the reachable set.
        """
        tube_lb, tube_ub = self._post_operator.get_tube_bounds()
        widths = tube_ub - tube_lb
        return np.prod(widths)

    def _is_scaling_helpful(self, prev_vol, cur_vol):
        """
        Return true if staying in the scaling mode helps with precision.
        """
        imprv_rate = (prev_vol - cur_vol) / prev_vol
        return imprv_rate >= self._settings.reach.scaling_cutoff

    def _is_start_scaling(self):
        """
        Return true if can start scaling at the current step.

        The stepsize increases. This is to (theoretically) avoid the case where
        the number of scaling steps added each time upon entering scaling mode
        is larger than the stepsize; and consequently computation will not terminate.
        """
        stepsize = max(int(self._settings.reach.num_steps *
                           self._settings.reach.scaling_freq), 1)

        return self._cur_step % stepsize == 0

    def _make_scaling_dynamics(self):
        """
        Return dynamics scaled by scaling function.

        Scaling function has two terms: 1) distance function; 2) scaling factor.
        """
        dist_func_str = self._make_dist_func_str()
        m = self._make_scaling_factor(dist_func_str)

        scaled_dynamics_str = []
        for dyn in self._cur_mode.dynamics.sp_dynamics:
            scaled_dynamics_str.append('{}*({})*({})'.format(m, dist_func_str, dyn))

        scaled_dynamics = GeneralDynamics(self._cur_mode.id_to_vars, *scaled_dynamics_str)
        return scaled_dynamics

    def _make_dist_func_str(self):
        """
        Return a string representation of the scaling function.
        """
        lb, ub = self._post_operator.get_tube_bounds()

        # domain center
        c = np.sum(self._abs_domain.bounds, axis=0) / 2
        # compute derivative
        deriv = self._cur_mode.dynamics.eval(c)
        # get norm vector
        norm_vec = deriv / (np.dot(deriv, deriv) ** 0.5)

        # get the minimal distance between the domain center and the reachable set boundary
        # along the normal vector direction.
        pos_clip = np.where(norm_vec >= 0, 1, 0)
        neg_clip = np.where(norm_vec < 0, 1, 0)
        # the corner which maximizes the distance
        p = pos_clip * ub + neg_clip * lb

        # distance function -(a/||a||) \cdot x + b, we denote a' = a/||a||
        b = np.dot(norm_vec, p)
        a_prime = [-elem for elem in norm_vec]

        linear_term = ''
        for idx, elem in enumerate(a_prime):
            linear_term += '{}*x{}+'.format(elem, idx)

        scaling_func_str = '{}+{}'.format(linear_term, b)

        return scaling_func_str

    def _make_scaling_factor(self, dist_func_str):
        """
        Return scaling factor, a rationale number.
        """
        dyn_with_dist_str = []
        for dyn in self._cur_mode.dynamics.sp_dynamics:
            dyn_with_dist_str.append('({})*({})'.format(dist_func_str, dyn))

        # domain center
        c = np.sum(self._abs_domain.bounds, axis=0) / 2
        # compute the norm of the dynamics scaled by multiply distance
        dyn_with_dist = GeneralDynamics(self._cur_mode.id_to_vars, *dyn_with_dist_str)
        dyn_with_dist_norm = np.linalg.norm(dyn_with_dist.eval_jacobian(c), np.inf)
        # compute the norm of the original linearized dynamics
        norm = np.linalg.norm(self._abs_dynamics.a_matrix, np.inf)
        # compute scaling factor m
        m = norm / dyn_with_dist_norm

        return m
