import numpy as np
import time

from convex_set.hyperbox import HyperBox
from cores.engine import NovaEngine
from cores.hybrid_automata import NonlinHybridAutomaton
from utils import suppfunc_utils
from utils.containers import ReachabilitySetting, VerificationSetting, AppSetting, PlotSetting, SimuSetting
from utils.pykodiak.pykodiak_interface import Kodiak
from utils.timerutil import Timers


def define_ha():
    ha = NonlinHybridAutomaton()
    ha.variables = ["x0", "x1", "x2", "x3", "x4",
                    "x5", "x6", "x7", "x8", "x9",
                    "x10", "x11", "x12", "x13", "x14",
                    "x15", "x16", "x17", "x18", "x19"
                    ]

    for var in ha.variables:
        Kodiak.add_variable(var)

    mode1 = ha.new_mode('1')
    mode1.set_dynamics(
        ["0.1*x4-3*x0+2.5*(x3+x8+x13+x18)", "10*x0-2.2*x1", "10*x1-1.5*x2", "2*x0-20*x3",
         "-5*x4**2*x2**4*(10*x1-1.5*x2)", "0.1*x9-3*x5+2.5*(x3+x8+x13+x18)", "10*x5-2.2*x6", "10*x6-1.5*x7",
         "2*x5-20*x8", "-5*x9**2*x7**4*(10*x6-1.5*x7)", "0.1*x14-3*x10+2.5*(x3+x8+x13+x18)", "10*x10-2.2*x11",
         "10*x11-1.5*x12", "2*x10-20*x13", "-5*x14**2*x12**4*(10*x11-1.5*x12)", "0.1*x19-3*x15+2.5*(x3+x8+x13+x18)",
         "10*x15-2.2*x16", "10*x16-1.5*x17", "2*x15-20*x18", "-5*x19**2*x17**4*(10*x16-1.5*x17)"]

        ,
        is_linear=(True, True, True, True, False,
                   True, True, True, True, False,
                   True, True, True, True, False,
                   True, True, True, True, False))

    return ha


def define_init_states(ha):
    """Return a list of (mode, HyperBox).
    """
    rv = list()
    rv.append(
        (ha.modes['1'], HyperBox([[-0.003, 0.197, 0.997, -0.003, 0.497, -0.001, 0.199, 0.999, -0.001, 0.499, 0.001,
                                   0.201, 1.001, 0.001, 0.501, 0.003, 0.203, 1.003, 0.003, 0.503],
                                  [-0.001, 0.199, 0.999, -0.001, 0.499, 0.001, 0.201, 1.001, 0.001, 0.501, 0.003,
                                   0.203, 1.003, 0.003, 0.503, 0.005, 0.205, 1.005, 0.005, 0.505]],
                                 opt=1)))

    return rv


def define_settings():
    sys_dim = 20
    horizon = 3
    model_name = 'coupled_osc_20d'

    dirs = suppfunc_utils.generate_directions(direction_type=0, dim=sys_dim)

    reach_setting = ReachabilitySetting(horizon=horizon, stepsize=0.0015,
                                        directions=dirs, error_model=2,
                                        scaling_freq=1, scaling_cutoff=1e-3)
    # specify unsafe region
    verif_setting = VerificationSetting(a_matrix=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                        b_col=np.array([0.085]))

    plot_setting = PlotSetting(poly_dir_path='../out/sfvals', model_name=model_name, opdims=(1, 2))
    simu_setting = SimuSetting(model_name=model_name, horizon=horizon,
                               init_set_bounds=[
                                   [-0.003, 0.197, 0.997, -0.003, 0.497, -0.001, 0.199, 0.999, -0.001,
                                    0.499, 0.001, 0.201, 1.001, 0.001, 0.501, 0.003, 0.203, 1.003, 0.003,
                                    0.503],
                                   [-0.001, 0.199, 0.999, -0.001, 0.499, 0.001, 0.201, 1.001, 0.001, 0.501, 0.003,
                                    0.203, 1.003, 0.003, 0.503, 0.005, 0.205, 1.005, 0.005, 0.505]])

    app_settings = AppSetting(reach_setting=reach_setting,
                              verif_setting=verif_setting,
                              plot_setting=plot_setting,
                              simu_setting=simu_setting)

    return app_settings


def run_nova(settings):
    Timers.tic('total')
    ha = define_ha()
    init = define_init_states(ha)

    engine = NovaEngine(ha, settings)
    engine.run(init)
    Timers.print_stats()

    # return engine.result


if __name__ == '__main__':
    settings = define_settings()
    run_nova(settings)
