import numpy as np

from cores.post_opt.abstract_post_opt import AbstractContinuousPostOpeartor
from utils import suppfunc_utils
from utils.containers import PostOptStateholder, ReachParams
from utils import utils


class FwdContinuousPostOpeartor(AbstractContinuousPostOpeartor):
    def __init__(self, init_state, directions):
        """
        :param init_state: a HyperBox
        """
        super().__init__()

        self.init_state = init_state
        self.tube_supp = None

        self._temp_tube_supp = None
        self._handler = PostOptStateholder(init_state.lb, init_state.ub)
        self._directions = directions
        self._dim = directions.shape[1]
        self._reach_params = ReachParams()
        self._canonical_direction_idx = utils.get_canno_dir_indices(directions)

    def update_reach_params(self, vals):
        self._reach_params.alpha, self._reach_params.beta, self._reach_params.delta, self._reach_params.tau = vals

    def do_gamma_step(self):
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
        """

        supp = np.empty(self._directions.shape[0])

        for idx, l in enumerate(self._directions):
            delta_T_l = self._handler.phi_list.get_val().dot(l).reshape(1, -1)
            pos_clip = np.clip(a=delta_T_l, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=delta_T_l, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(self._handler.input_ub_seq.get_val()) + \
                     neg_clip.dot(self._handler.input_lb_seq.get_val())

            supp[idx] = maxval

        self.set_supp = supp

        # update current initial set
        self._update_cur_init_set()

    def do_alpha_step(self, Vi_lb, Vi_ub):
        W_alpha_lb = Vi_lb * self._reach_params.tau - self._reach_params.alpha
        W_alpha_ub = Vi_ub * self._reach_params.tau + self._reach_params.alpha

        next_lb_seq = np.hstack((self._handler.input_lb_seq.get_val(), W_alpha_lb))
        next_ub_seq = np.hstack((self._handler.input_ub_seq.get_val(), W_alpha_ub))

        delta_T = self._reach_params.delta.T

        if len(self._handler.phi_list.get_val()) == 0:
            phi_list = np.array([delta_T])
        else:
            phi_list = np.tensordot(self._handler.phi_list.get_val(), delta_T, axes=(2, 0))
        phi_list = np.vstack((phi_list, [np.eye(self._dim)]))

        # compute supp values at tau step
        supp = np.empty(self._directions.shape[0])

        for idx, l in enumerate(self._directions):
            delta_T_l = phi_list.dot(l).reshape(1, -1)
            pos_clip = np.clip(a=delta_T_l, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=delta_T_l, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(next_ub_seq) + neg_clip.dot(next_lb_seq)
            supp[idx] = maxval

        # (template) convex hull
        tube_supp = np.maximum(supp, self.set_supp)

        # update temp reach tube
        self._update_temp_supp_tube(tube_supp)
        self.tube_supp = tube_supp

    def _update_inhomo_seq(self, next_lb, next_ub, a_matrix):
        """
         W_{i} = τV_{i} ⊕ τd ⊕ β_{i}·B, where d = (1/τ) * \int_{0}^{τ}[(e^(τ-s)A-I)c]ds.
         c is the center of the boxed uncertainty region.
        """
        c = (next_lb + next_ub) / 2
        A = a_matrix
        M = suppfunc_utils.mat_exp_int(A, t_min=0, t_max=self._reach_params.tau)
        tau_d = np.dot(M, c)

        next_lb = next_lb * self._reach_params.tau - self._reach_params.beta + tau_d
        next_ub = next_ub * self._reach_params.tau + self._reach_params.beta + tau_d

        lb, ub = np.append(self._handler.input_lb_seq.get_val(), next_lb), \
                 np.append(self._handler.input_ub_seq.get_val(), next_ub)

        self._handler.input_lb_seq.set_val(lb)
        self._handler.input_ub_seq.set_val(ub)

    def _update_phi_list(self):
        """
        phi_list contains the product of delta_transpose.
        After n-times update, phi_list looks like this:
        [ Φ_{n}^T Φ_{n-1}^T … Φ_{1}^T, Φ_{n-1}^T … Φ_{1}^T, ..., Φ_{1}^T]
        """
        if len(self._handler.phi_list.get_val()) == 0:
            temp_list = np.array([np.eye(self._dim)])
        else:
            delta_T = self._reach_params.delta.T

            temp_list = np.tensordot(self._handler.phi_list.get_val(), delta_T, axes=(2, 0))
            temp_list = np.vstack((temp_list, [np.eye(self._dim)]))

        self._handler.phi_list.set_val(temp_list)