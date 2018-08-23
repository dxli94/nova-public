import sys

import numpy as np

from AffinePostOpt import PostOperator as AffinePostOperator
import SuppFuncUtils
from Hybridisation.NonlinPostOpt import NonlinPostOpt
from SysDynamics import GeneralDynamics
from Plotter import Plotter
from utils.DataReader import JsonReader
import utils.simulator as simu
from timerutil import Timers
import time


def main():
    start_time = time.time()
    # # ============== setting up ============== #
    try:
        path = sys.argv[1]
    except IndexError:
        # path = '../instances/non_linear_instances/vanderpol.json'
        # path = '../instances/non_linear_instances/predator_prey.json'
        # path = '../instances/non_linear_instances/2d_water_tank.json'
        # path = '../instances/non_linear_instances/brusselator.json'
        # path = '../instances/non_linear_instances/jet_engine.json'
        # path = '../instances/non_linear_instances/free_ball.json'
        # path = '../instances/non_linear_instances/constant_moving.json'
        # path = '../instances/non_linear_instances/buckling_column.json'
        # path = '../instances/non_linear_instances/pbt.json'
        # path = '../instances/non_linear_instances/pbt_y.json'
        # path = '../instances/non_linear_instances/lacoperon.json'
        # path = '../instances/non_linear_instances/coupled_vanderpol.json'
        # path = '../instances/non_linear_instances/spring_pendulum.json'
        # path = '../instances/non_linear_instances/lorentz_system.json'
        # path = '../instances/non_linear_instances/biology_1.json'
        path = '../instances/non_linear_instances/biology_2.json'

    # buckling_column: d = 0.1, dwell_steps = 200, start_i = 50
    # vanderpol: d = 0.1, dwell_steps = 5, start_i = 100, time_step = 0.02
    # brusselator: d = 0.1, dwell_steps = 400, start_i = 100, time_step = 0.02

    print('reading model file: {}'.format(path.split('/')[-1]))
    data = JsonReader(path).read()
    time_horizon = data['time_horizon']
    tau = data['sampling_time']
    direction_type = data['direction_type']
    dim = data['dim']
    start_epsilon = data['start_epsilon']
    non_linear_dynamics = data['dynamics']
    state_vars = data['state_variables']
    is_linear = data['is_linear']
    init_coeff = np.array(data['init_coeff'])
    init_col = np.array(data['init_col'])
    opvars = data['opvars']
    simu_model = data['simu_model']
    pseudo_var = data['pseudo_var']

    # Timers.tic('Generate directions')
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    # Timers.toc('Generate directions')

    # Timers.tic('Generate General Dynamics')
    id_to_vars = {}
    for i, var in enumerate(state_vars):
        id_to_vars[i] = var
    non_linear_dynamics = GeneralDynamics(id_to_vars, *non_linear_dynamics)
    # Timers.toc('Generate General Dynamics')
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    # Timers.tic('flowpipe computation')
    np.set_printoptions(precision=100)
    nonlin_post_opt = NonlinPostOpt(dim, non_linear_dynamics, time_horizon, tau, directions,
                                    init_coeff, init_col, is_linear, start_epsilon, pseudo_var, id_to_vars)
    sf_mat = nonlin_post_opt.compute_post()
    # Timers.toc('flowpipe computation')

    # Timers.tic('get projection')
    images = AffinePostOperator.get_projections_new(directions=directions, opdims=opvars, sf_mat=sf_mat)
    # Timers.toc('get projection')

    # Timers.tic('Initialise Plotter')
    plotter = Plotter(images, opvars)
    # Timers.toc('Initialise Plotter')

    # Timers.tic('Save Polygons to file')
    plotter.save_polygons_to_file()
    # Timers.toc('Save Polygons to file')

    # Timers.tic('Run simulation')
    run_simulate(time_horizon, simu_model, init_coeff, init_col, opvars)
    # Timers.toc('Run simulation')
    #
    # Timers.toc('total')
    # Timers.print_stats()
    #
    # images = nonlin_post_opt.lin_post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=x_mat)
    # plotter = Plotter(images, opvars)
    # plotter.save_polygons_to_file(filename='x.out')
    print('Total running time: {:.2f}'.format(time.time() - start_time))


def run_simulate(time_horizon, model, init_coeff, init_col, opdims):
    x, y = simu.simulate(time_horizon, model, init_coeff, init_col, opdims)
    with open('../out/simu.out', 'w') as simu_op:
        for elem in zip(x, y):
            simu_op.write(str(elem[0]) + ' ' + str(elem[1]) + '\n')
    return x, y


if __name__ == '__main__':
    main()
