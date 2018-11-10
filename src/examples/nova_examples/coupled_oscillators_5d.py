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
    ha.variables = ["x0", "x1", "x2", "x3", "x4"]

    for var in ha.variables:
        Kodiak.add_variable(var)

    mode1 = ha.new_mode('1')
    mode1.set_dynamics(
        ["0.1*x4-3*x0+10.0*(x3)",
         "10*x0-2.2*x1",
         "10*x1-1.5*x2",
         "2*x0-20*x3",
         "-5*x4**2*x2**4*(10*x1-1.5*x2)"
         ],
        is_linear=(True, True, True, True, False))

    return ha


def define_init_states(ha):
    """Return a list of (mode, HyperBox).
    """
    rv = list()

    # HSCC 16'
    # rv.append((ha.modes['1'], HyperBox([[1.25, 2.28, 1.25, 2.28], [1.55, 2.32, 1.55, 2.32]], opt=1)))
    # ARCH 19'
    rv.append((ha.modes['1'], HyperBox([[-0.003, 0.197, 0.997, -0.003, 0.497],
                                        [-0.001, 0.199, 0.999, -0.001, 0.499]], opt=1)))

    return rv


def define_settings():
    sys_dim = 5
    horizon = 3
    model_name = 'coupled_osc_5d'

    dirs = suppfunc_utils.generate_directions(direction_type=0, dim=sys_dim)

    reach_setting = ReachabilitySetting(horizon=horizon, stepsize=0.005,
                                        directions=dirs, error_model=2,
                                        scaling_freq=1, scaling_cutoff=1e-3)
    # specify unsafe region
    verif_setting = VerificationSetting(a_matrix=np.array([0, 1, 0, 0, 0]),
                                        b_col=np.array([0.08]))

    plot_setting = PlotSetting(poly_dir_path='../out/sfvals', model_name=model_name)
    # HSCC 16'
    # simu_setting = SimuSetting(model_name=model_name, horizon=horizon, init_set_bounds=[[1.25, 2.28, 1.25, 2.28], [1.55, 2.32, 1.55, 2.32]])
    # ARCH 19'
    simu_setting = SimuSetting(model_name=model_name, horizon=horizon,
                               init_set_bounds=[[-0.003, 0.197, 0.997, -0.003, 0.497],
                                                [-0.001, 0.199, 0.999, -0.001, 0.499]])

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
