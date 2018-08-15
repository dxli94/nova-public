import sys

import numpy as np

from AffinePostOpt import PostOperator as AffinePostOperator
import SuppFuncUtils
from Hybridisation.NonlinPostOpt import NonlinPostOpt
from SysDynamics import GeneralDynamics
from Plotter import Plotter
from utils.DataReader import JsonReader
import utils.simulator as simu


def main():
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
        path = '../instances/non_linear_instances/buckling_column.json'

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

    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    id_to_vars = {}
    for i, var in enumerate(state_vars):
        id_to_vars[i] = var
    non_linear_dynamics = GeneralDynamics(id_to_vars, *non_linear_dynamics)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    np.set_printoptions(precision=100)
    nonlin_post_opt = NonlinPostOpt(dim, non_linear_dynamics, time_horizon, tau, directions,
                                    init_coeff, init_col, is_linear, start_epsilon, pseudo_var)
    sf_mat = nonlin_post_opt.compute_post()
    # images = nonlin_post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=bound_mat)
    # plotter = Plotter(images, opvars)
    # plotter.save_polygons_to_file()

    images = AffinePostOperator.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()
    run_simulate(time_horizon, simu_model, init_coeff, init_col)

    #
    # images = nonlin_post_opt.lin_post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=x_mat)
    # plotter = Plotter(images, opvars)
    # plotter.save_polygons_to_file(filename='x.out')


def run_simulate(time_horizon, model, init_coeff, init_col):
    x, y = simu.simulate(time_horizon, model, init_coeff, init_col)
    with open('../out/simu.out', 'w') as simu_op:
        for elem in zip(x, y):
            simu_op.write(str(elem[0]) + ' ' + str(elem[1]) + '\n')
    return x, y


if __name__ == '__main__':
    main()
