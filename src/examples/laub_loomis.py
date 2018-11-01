import numpy as np

from convex_set.hyperbox import HyperBox
from cores.engine import NovaEngine
from cores.hybrid_automata import NonlinHybridAutomaton
from utils import suppfunc_utils
from utils.containers import ReachabilitySetting, VerificationSetting, AppSetting, PlotSetting, SimuSetting
from utils.pykodiak.pykodiak_interface import Kodiak
from utils.timerutil import Timers


def define_ha():
    ha = NonlinHybridAutomaton()
    ha.variables = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]

    for var in ha.variables:
        Kodiak.add_variable(var)

    mode1 = ha.new_mode('1')
    mode1.set_dynamics(["1.4 * x2 - 0.9 * x0",
                        "2.5 * x4 - 1.5 * x1",
                        "0.6 * x6 - 0.8 * x2 * x1",
                        "2 - 1.3 * x3 * x2",
                        "0.7 * x0 - x3 * x4",
                        "0.3 * x0 - 3.1 * x5",
                        "1.8 * x5 - 1.5 * x6 * x1"
                        ],
                       is_linear=(True,
                                  True,
                                  False,
                                  False,
                                  False,
                                  True,
                                  False))

    return ha


def define_init_states(ha):
    """Return a list of (mode, HyperBox).
    """
    rv = list()

    # ARCH 19', W=0.01
    rv.append((ha.modes['1'], HyperBox([[1.19, 1.04, 1.49, 2.39, 0.99, 0.09, 0.44],
                                        [1.21, 1.06, 1.51, 2.41, 1.01, 0.11, 0.46]], opt=1)))

    return rv


def define_settings():
    sys_dim = 7
    horizon = 20
    model_name = 'laub_loomis'

    dirs = suppfunc_utils.generate_directions(direction_type=1, dim=sys_dim)

    reach_setting = ReachabilitySetting(horizon=horizon, stepsize=0.02,
                                        directions=dirs, error_model=2,
                                        scaling_freq=0.1, scaling_cutoff=0.005)
    # specify unsafe region
    verif_setting = VerificationSetting(a_matrix=np.array([0, -1]),
                                        b_col=np.array([-3]))

    plot_setting = PlotSetting(poly_dir_path='../out/sfvals', model_name=model_name)
    simu_setting = SimuSetting(model_name=model_name, horizon=horizon,
                               init_set_bounds=[[1.19, 1.04, 1.49, 2.39, 0.99, 0.09, 0.44],
                                                [1.21, 1.06, 1.51, 2.41, 1.01, 0.11, 0.46]])

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
