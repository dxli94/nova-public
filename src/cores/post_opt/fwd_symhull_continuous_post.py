import numpy as np

from cores.post_opt.base_post_opt import BaseContinuousPostOperator
from utils import suppfunc_utils


class FwdSymhullContinuousPostOperator(BaseContinuousPostOperator):
    def __init__(self, init_state, directions):
        super().__init__(init_state, directions)

    def update_reach_params(self, vals):
        self._reach_params.alpha, _, self._reach_params.delta, self._reach_params.tau = vals

    def _update_inhomo_seq(self, next_lb, next_ub, a_matrix):
        """
        Offsetting the input set.

         W_{i} = τV_{i} ⊕ ⊡(Φ_{2}(|A|, τ) ⊡(AU)),
        """
        c = (next_lb + next_ub) / 2
        next_w_lb = next_lb - c
        next_w_ub = next_ub - c

        a_abs_matrix = np.absolute(a_matrix)
        # Φ_{1}(|A|, τ), Φ_{2}(|A|, τ)
        phi1 = suppfunc_utils.compute_phi_1(a_matrix, self._reach_params.tau)
        phi2 = suppfunc_utils.compute_phi_2(a_abs_matrix, self._reach_params.tau)

        E_U = self._compute_E_U(next_w_lb, next_w_ub, a_abs_matrix, phi2)
        phi1_c = phi1.dot(c)

        next_lb = next_w_lb * self._reach_params.tau - E_U + phi1_c
        next_ub = next_w_ub * self._reach_params.tau + E_U + phi1_c

        lb, ub = np.append(self._handler.input_lb_seq.get_val(), next_lb), \
                 np.append(self._handler.input_ub_seq.get_val(), next_ub)

        self._handler.input_lb_seq.set_val(lb)
        self._handler.input_ub_seq.set_val(ub)

    def _compute_E_U(self, next_lb, next_ub, a_matrix, phi2):
        """
        Compute ⊡(Φ_{2}(|A|, τ) ⊡ (AU))
        """

        # AU
        a_pos_clip = np.clip(a=a_matrix, a_min=0, a_max=np.inf)
        a_neg_clip = np.clip(a=a_matrix, a_min=-np.inf, a_max=0)

        AU_ub = a_pos_clip.dot(next_ub) + a_neg_clip.dot(next_lb)
        AU_lb = a_neg_clip.dot(next_ub) + a_pos_clip.dot(next_lb)

        # ⊡(AU)
        box_AU_ub = AU_ub.copy()
        box_AU_lb = AU_lb.copy()

        for i in range(a_matrix.shape[0]):
            abs_ub = abs(box_AU_ub[i])
            abs_lb = abs(box_AU_lb[i])

            if abs_ub > abs_lb:
                box_AU_lb[i] = -box_AU_ub[i]
            elif abs_ub < abs_lb:
                box_AU_ub[i] = -box_AU_lb[i]

        # phi2_box_AU = Φ_{2}(|A|, τ) ⊡ (AU)
        phi2_pos_clip = np.clip(a=phi2, a_min=0, a_max=np.inf)
        phi2_neg_clip = np.clip(a=phi2, a_min=-np.inf, a_max=0)

        phi2_box_AU_ub = phi2_pos_clip.dot(box_AU_ub) + phi2_neg_clip.dot(box_AU_lb)
        phi2_box_AU_lb = phi2_neg_clip.dot(box_AU_ub) + phi2_pos_clip.dot(box_AU_lb)

        # box_phi2_box_AU = ⊡(Φ_{2}(|A|, τ) ⊡(AU))
        # The following parts might not be necessary
        box_phi2_box_AU_ub = phi2_box_AU_ub.copy()
        box_phi2_box_AU_lb = phi2_box_AU_lb.copy()

        for i in range(a_matrix.shape[0]):
            abs_ub = abs(box_phi2_box_AU_ub[i])
            abs_lb = abs(box_phi2_box_AU_lb[i])

            if abs_ub > abs_lb:
                box_phi2_box_AU_lb[i] = -box_phi2_box_AU_ub[i]
            elif abs_ub < abs_lb:
                box_phi2_box_AU_ub[i] = -box_phi2_box_AU_lb[i]

        E_U = phi2_box_AU_ub

        return E_U

    def do_alpha_step(self, Vi_lb, Vi_ub):
        W_alpha_lb = Vi_lb * self._reach_params.tau - self._reach_params.alpha
        W_alpha_ub = Vi_ub * self._reach_params.tau + self._reach_params.alpha

        next_lb_seq = np.hstack((self._handler.input_lb_seq.get_val(), W_alpha_lb))
        next_ub_seq = np.hstack((self._handler.input_ub_seq.get_val(), W_alpha_ub))

        delta_T = self._reach_params.delta.T

        matexp_list = np.tensordot(self._handler.matexp_list.get_val(), delta_T, axes=(2, 0))
        matexp_list = np.vstack((matexp_list, [np.eye(self._dim)]))

        # compute supp values at tau step
        supp = np.empty(self._directions.shape[0])

        for idx, l in enumerate(self._directions):
            delta_T_l = matexp_list.dot(l).reshape(1, -1)
            pos_clip = np.clip(a=delta_T_l, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=delta_T_l, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(next_ub_seq) + neg_clip.dot(next_lb_seq)
            supp[idx] = maxval

        # (template) convex hull
        tube_supp = np.maximum(supp, self.set_supp)

        # update temp reach tube
        self._update_temp_supp_tube(tube_supp)
        self.tube_supp = tube_supp

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
            delta_T_l = self._handler.matexp_list.get_val().dot(l).reshape(1, -1)
            pos_clip = np.clip(a=delta_T_l, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=delta_T_l, a_min=-np.inf, a_max=0)

            maxval = pos_clip.dot(self._handler.input_ub_seq.get_val()) + \
                     neg_clip.dot(self._handler.input_lb_seq.get_val())

            supp[idx] = maxval

        self.set_supp = supp

        # update current initial set
        self._update_cur_init_set()
