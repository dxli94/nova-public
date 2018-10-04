import numpy as np

from convex_set.hyperbox import HyperBox
from cores.engine import NovaEngine
from cores.hybrid_automata import NonlinHybridAutomaton
from utils import suppfunc_utils
from utils.containers import ReachabilitySetting, VerificationSetting, AppSetting, PlotSetting, SimuSetting


def define_ha():
    ha = NonlinHybridAutomaton()
    ha.variables = ["x0", "x1"]

    mode1 = ha.new_mode('1')
    mode1.set_dynamics(["x1", "(1-x0^2)*x1-x0"], is_linear=(True, False))

    return ha


def define_init_states(ha):
    """Return a list of (mode, HyperBox).
    """
    rv = list()

    rv.append((ha.modes['1'], HyperBox([[1.25, 2.45], [1.70, 2.65]], opt=1)))

    return rv


def define_settings():
    sys_dim = 2
    horizon = 7
    model_name = 'vanderpol'

    dirs = suppfunc_utils.generate_directions(direction_type=1, dim=sys_dim)

    reach_setting = ReachabilitySetting(horizon=horizon, stepsize=0.02,
                                        directions=dirs, error_model=2,
                                        scaling_freq=0.1, scaling_cutoff=0.01)
    # specify unsafe region
    verif_setting = VerificationSetting(a_matrix=np.array([0, -1]),
                                        b_col=np.array([-3]))

    plot_setting = PlotSetting(poly_dir_path='../out/sfvals', model_name=model_name)
    simu_setting = SimuSetting(model_name=model_name, horizon=horizon, init_set_bounds=[[1.25, 2.45], [1.70, 2.65]])

    app_settings = AppSetting(reach_setting=reach_setting,
                              verif_setting=verif_setting,
                              plot_setting=plot_setting,
                              simu_setting=simu_setting)

    return app_settings


def run_nova(settings):
    ha = define_ha()
    init = define_init_states(ha)

    engine = NovaEngine(ha, settings)
    engine.run(init)

    # return engine.result


if __name__ == '__main__':
    settings = define_settings()
    run_nova(settings)
