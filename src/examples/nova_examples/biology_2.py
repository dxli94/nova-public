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
    ha.variables = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]

    for var in ha.variables:
        Kodiak.add_variable(var)

    mode1 = ha.new_mode('1')
    mode1.set_dynamics(["3*x2 - x0*x5",
                        "x3 - x1*x5",
                        "x0*x5 - 3*x2",
                        "x1*x5 - x3",
                        "3*x2 + 5*x0 - x4",
                        "5*x4 + 3*x2 + x3 - x5*(x0 + x1 + 2*x7 + 1)",
                        "5*x3 + x1 - 0.5*x6",
                        "5*x6 - 2*x5*x7 + x8 - 0.2*x7",
                        "2*x5*x7 - x8"
                        ],
                       is_linear=(False,
                                  False,
                                  False,
                                  False,
                                  True,
                                  False,
                                  True,
                                  False,
                                  False))

    return ha


def define_init_states(ha):
    """Return a list of (mode, HyperBox).
    """
    rv = list()

    rv.append((ha.modes['1'], HyperBox([[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                                        [1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]], opt=1)))

    return rv


def define_settings():
    sys_dim = 9
    horizon = 2
    model_name = 'biology_2'

    dirs = suppfunc_utils.generate_directions(direction_type=1, dim=sys_dim)

    reach_setting = ReachabilitySetting(horizon=horizon, stepsize=0.001,
                                        directions=dirs, error_model=2,
                                        scaling_freq=1, scaling_cutoff=0.005)
    # specify unsafe region
    verif_setting = VerificationSetting(a_matrix=np.array([0, -1]),
                                        b_col=np.array([-3]))

    plot_setting = PlotSetting(poly_dir_path='../out/sfvals', model_name=model_name)
    simu_setting = SimuSetting(model_name=model_name, horizon=horizon,
                               init_set_bounds=[[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                                                [1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]])

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
