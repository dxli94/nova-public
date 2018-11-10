"""
Nova Containers File
Dongxu Li

"""
import math

from utils.utils import Freezable
from utils.utils import TrackedVar as tvar


class AppSetting(Freezable):
    """
    All settings for NOVA.
    """
    def __init__(self, reach_setting, verif_setting=None, simu_setting=None, plot_setting=None):
        self.reach = reach_setting

        self.verif = verif_setting
        self.simu = simu_setting
        self.plot = plot_setting

        self.freeze_attrs()


class ReachabilitySetting(Freezable):
    """
    Setting for reachability analysis.
    """
    def __init__(self, horizon, stepsize, directions, error_model, scaling_freq, scaling_cutoff):
        assert horizon > 0
        assert stepsize > 0
        assert error_model > 0
        assert len(directions) > 0

        self.stepsize = stepsize
        self.num_steps = int(math.ceil(horizon / stepsize))
        self.directions = directions

        self.error_model = error_model
        self.scaling_freq = scaling_freq
        self.scaling_cutoff = scaling_cutoff

        self.freeze_attrs()


class VerificationSetting(Freezable):
    """
    A container for verification setting.
    """
    def __init__(self, a_matrix, b_col):
        """
        Unsafe region defined by Ax <= b
        """
        self.a_matrix = a_matrix
        self.b_col = b_col

        self.freeze_attrs()


class PlotSetting(Freezable):
    """
    A container for plotting setting.
    """
    def __init__(self, poly_dir_path, model_name, opdims=None):
        self.poly_dir_path = poly_dir_path
        self.model_name = model_name
        self.opdims = opdims

        self.freeze_attrs()


class SimuSetting(Freezable):
    """
    A container for simulation setting.
    """
    def __init__(self, model_name, horizon, init_set_bounds):
        self.model_name = model_name
        self.horizon = horizon
        self.init_set = init_set_bounds

        self.freeze_attrs()


class ReachParams:
    """
    A container for bloating factors, matrix exp. and sampling time.
    """
    def __init__(self, alpha=None, beta=None, delta=None, tau=None):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.tau = tau


class PostOptStateholder:
    """
    A container for continuous post state variables.
    """
    def __init__(self, init_set_lb, init_set_ub):
        self.tvars_list = []
        self.init_set_lb = tvar()
        self.init_set_ub = tvar()

        self.tube_lb = tvar()
        self.tube_ub = tvar()
        self.tvars_list.append(self.tube_lb)
        self.tvars_list.append(self.tube_ub)
        # tube_lb, tube_ub = init_set_lb, init_set_ub

        # temporary variables for reachable states in dense time
        self.temp_tube_lb = tvar()
        self.temp_tube_ub = tvar()
        self.tvars_list.append(self.temp_tube_lb)
        self.tvars_list.append(self.temp_tube_ub)

        # temp_tube_lb, temp_tube_ub = init_set_lb, init_set_ub
        # initial reachable set in discrete time in the current abstract domain
        # changes when the abstract domain is large enough to contain next image in alfa step
        self.cur_init_set_lb = tvar()
        self.cur_init_set_ub = tvar()
        self.tvars_list.append(self.cur_init_set_lb)
        self.tvars_list.append(self.cur_init_set_ub)

        # current_init_set_lb, current_init_set_ub = init_set_lb, init_set_ub

        self.input_lb_seq = tvar()
        self.input_ub_seq = tvar()
        self.tvars_list.append(self.input_lb_seq)
        self.tvars_list.append(self.input_ub_seq)
        # input_lb_seq, input_ub_seq = init_set_lb, init_set_ub

        self.matexp_list = tvar([])
        self.tvars_list.append(self.matexp_list)

        self._reset(init_set_lb, init_set_ub)

    def rollback(self):
        """
        Set all state variables to the previous value.
        """
        for tv in self.tvars_list:
            tv.rollback()

    def templify(self, init_set_lb, init_set_ub):
        """
        Bounding box the current init set. This could help
        increase the running speed but will decrease the precision.
        """
        self._reset(init_set_lb, init_set_ub)

    def _reset(self, init_set_lb, init_set_ub):
        self.init_set_lb = init_set_lb
        self.init_set_ub = init_set_ub

        self.tube_lb.reset(init_set_lb)
        self.tube_ub.reset(init_set_ub)

        self.temp_tube_lb.reset(init_set_lb)
        self.temp_tube_ub.reset(init_set_ub)

        self.cur_init_set_lb.reset(init_set_lb)
        self.cur_init_set_ub.reset(init_set_ub)

        self.input_lb_seq.reset(init_set_lb)
        self.input_ub_seq.reset(init_set_ub)

        self.matexp_list.reset([])