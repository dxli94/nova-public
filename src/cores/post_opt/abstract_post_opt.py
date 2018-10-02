import numpy as np

from utils import utils


class AbstractContinuousPostOpeartor:
    """
    An abstract class providing definitions for basic operations
    that all continuous operators should implement.
    """

    def __init__(self):
        self._handler = None
        self._canonical_direction_idx = None

        self.set_supp = None

    def do_init_step(self):
        """
        Initial step before computing flowpipes.
        """
        self._update_phi_list()
        self.do_gamma_step()

    def update_reach_params(self, vals):
        """
        Update reachability parameters.

        Parameter set may differ on different post operator algorithm.
        """
        raise NotImplementedError

    def do_gamma_step(self):
        """
        Compute the support function for the initial reachable set in the next
        abstraction domain (mode).
        """
        raise NotImplementedError

    def do_alpha_step(self, Vi_lb, Vi_ub):
        """
        Compute the support functions for the reachable tube.
        """
        raise NotImplementedError

    def proceed_state(self, cur_input_lb, cur_input_ub, a_matrix):
        """
        Update computation components to move computation one step ahead.
        """
        self._update_phi_list()
        self._update_tube(self._temp_tube_supp)
        self._update_inhomo_seq(cur_input_lb, cur_input_ub, a_matrix)

    def _update_inhomo_seq(self, next_lb, next_ub, a_matrix):
        """
        Update the sequence of E_U, which is the inhomogeneous term.
        """
        raise NotImplementedError

    def _update_phi_list(self):
        """
        Update the sequence of the product of chained matrix exponential.
        """
        raise NotImplementedError

    def _update_tube(self, supp_vals_tube):
        """
        Update the bounds of reachable tube.

        Extract lower and upper bounds from the support functions. This is assuming that
        we at least have box directions in the template directions.
        """
        lb, ub = utils.extract_bounds_from_sf(supp_vals_tube, self._canonical_direction_idx)
        self._handler.tube_lb.set_val(lb)
        self._handler.tube_ub.set_val(ub)

    def get_tube_bounds(self):
        """
        Get the lower and upper bounds of the reachable tube.
        """
        return self._handler.tube_lb.get_val(), \
               self._handler.tube_ub.get_val()

    def get_temp_tube_bounds(self):
        """
        Get the lower and upper bounds of the temporary reachable tube.
        """
        return self._handler.temp_tube_lb.get_val(), \
               self._handler.temp_tube_ub.get_val()

    def get_cur_init_set(self):
        """
        Get the lower and upper bounds of the initial reachable set in the current
        abstraction domain (mode).
        """
        return self._handler.cur_init_set_lb.get_val(), \
               self._handler.cur_init_set_ub.get_val()

    def has_lost_precision(self, max_tolerance):
        """
        If the reachable tube starts to blow up, stop the computation.

        max_tolerance is a large fixed value. We use 1e5 which is sufficient for any
        moderately large initial set and dynamics. This can be easily changed however.
        """
        return any(np.abs(self._handler.temp_tube_lb.get_val()) >= max_tolerance) or \
               any(np.abs(self._handler.temp_tube_ub.get_val()) >= max_tolerance)

    def rollback(self):
        """
        Going back to the previous state. This happens when time-scaling is found un-helpful.
        """
        self._handler.rollback()

    def _update_cur_init_set(self):
        """
        Update the initial reachable set in the current abstraction domain (mode).
        """
        lb, ub = utils.extract_bounds_from_sf(self.set_supp, self._canonical_direction_idx)
        self._handler.cur_init_set_lb.set_val(lb)
        self._handler.cur_init_set_ub.set_val(ub)

    def _update_temp_supp_tube(self, tube_supp):
        """
        Update the temporary support function values for reachable tube.
        """
        alpha_bounds = utils.extract_bounds_from_sf(tube_supp, self._canonical_direction_idx)
        self._temp_tube_supp = tube_supp
        self._handler.temp_tube_lb.set_val(alpha_bounds[0])
        self._handler.temp_tube_ub.set_val(alpha_bounds[1])