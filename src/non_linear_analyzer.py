import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import SuppFuncUtils
import utils.simulator as simu
from AffinePostOpt import PostOperator as AffinePostOperator
from Hybridisation.NonlinPostOpt import NonlinPostOpt
from Plotter import Plotter
from SysDynamics import GeneralDynamics
from utils.DataReader import JsonReader

from timerutil import Timers


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
        # path = '../instances/non_linear_instances/roessler_attractor.json'
        # path = '../instances/non_linear_instances/coupled_vanderpol.json'
        path = '../instances/non_linear_instances/spring_pendulum.json'
        # path = '../instances/non_linear_instances/lorentz_system.json'
        # path = '../instances/non_linear_instances/biology_1.json'
        # path = '../instances/non_linear_instances/biology_2.json'
        # path = '../instances/non_linear_instances/laub_loomis.json'

    model_name = path.split('/')[-1].split('.')[0]
    print('reading model file: {}'.format(model_name))
    data = JsonReader(path).read()
    print('Finished.')
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
    opdims = data['opvars']
    simu_model = data['simu_model']
    # pseudo_var = data['pseudo_var']
    pseudo_var = False
    scaling_per = data['scaling_per']
    scaling_cutoff = data['scaling_cutoff']

    directions = SuppFuncUtils.generate_directions(direction_type, dim)

    id_to_vars = {}
    for i, var in enumerate(state_vars):
        id_to_vars[i] = var
    non_linear_dynamics = GeneralDynamics(id_to_vars, *non_linear_dynamics)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    np.set_printoptions(precision=100)
    # def __init__(self, dim, nonlin_dyn, time_horizon, tau, directions,
    #              init_mat_X0, init_col_X0, is_linear, start_epsilon, pseudo_var, id_to_vars):
    nonlin_post_opt = NonlinPostOpt(dim=dim,
                                    nonlin_dyn=non_linear_dynamics,
                                    time_horizon=time_horizon,
                                    tau=tau,
                                    init_coeff=init_coeff,
                                    init_col=init_col,
                                    is_linear=is_linear,
                                    directions=directions,
                                    start_epsilon=start_epsilon,
                                    pseudo_var=pseudo_var,
                                    scaling_per=scaling_per,
                                    scaling_cutoff=scaling_cutoff,
                                    id_to_vars=id_to_vars)
    sf_mat = nonlin_post_opt.compute_post()
    # ============== Flowpipe construction done. ============== #

    pickle_path = os.path.join('../out/pickles', model_name +
                               '_T={}_t={}_dt={}_pv={}.pickle'.format(time_horizon, tau,
                                                                      direction_type, pseudo_var))
    simu_dir_path = os.path.join('../out/simu_traj', model_name + '.simu')
    poly_dir_path = '../out/sfvals'

    pickle_start_time = time.time()
    print('\nStoring support function values on disk...')
    create_pickle(pickle_path, sf_mat)
    print('Finished in {} secs.'.format(time.time() - pickle_start_time))

    xs = run_simulate(time_horizon, simu_model, init_coeff, init_col)
    simu.save_simu_traj(xs, simu_dir_path)

    print('Saving images...')
    img_start_time = time.time()
    make_plot(dim, directions, sf_mat, model_name, xs, poly_dir_path)
    print('Finished in {} secs.'.format(time.time() - img_start_time))
    print('Total running time: {:.2f}'.format(time.time() - start_time))


def run_simulate(time_horizon, model, init_coeff, init_col):
    xs = simu.simulate(time_horizon, model, init_coeff, init_col)
    return xs


def create_pickle(filename, data):
    with open(filename, 'wb') as opfile:
        pickle.dump(data, opfile)


def make_plot(dim, directions, sf_mat, model_name, xs, poly_dir):
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                continue
            opdims = (i, j)
            config_plt(opdims)
            ppl_polys = AffinePostOperator.get_projections_new(directions=directions, opdims=opdims, sf_mat=sf_mat)

            img_dir_path = os.path.join('../out/imgs', model_name)
            if not os.path.exists(img_dir_path):
                os.mkdir(img_dir_path)
            img_path = os.path.join(img_dir_path, '{}-{}.png'.format(*opdims))
            plotter = Plotter(ppl_polys, opdims)

            # plot simulation
            x, y = xs[:, opdims[0]], xs[:, opdims[1]]
            plotter.plot_points(x, y, xlabel=str(i), ylabel=str(j))

            # plot polygons
            poly_dir_path = os.path.join(poly_dir, model_name)
            if not os.path.exists(poly_dir_path):
                os.mkdir(poly_dir_path)
            poly_file_path = os.path.join(poly_dir_path, '{}-{}'.format(*opdims))
            plotter.save_polygons_to_file(filename=poly_file_path)
            plotter.plot_polygons(poly_file_path, xlabel=str(i), ylabel=str(j))

            # plot scaling points
            if None:  # has issue with transparency,
                if os.path.exists('../out/pivots.out'):
                    plotter.plot_pivots('../out/pivots.out', opdims, 'green')
                if os.path.exists('../out/sca_cent.out'):
                    plotter.plot_pivots('../out/sca_cent.out', opdims, 'yellow')

            plotter.save_plt(opfile=img_path)


def config_plt(opdims):
    plt.clf()

    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x_{}$'.format(opdims[0]))
    ax.set_ylabel('$x_{}$'.format(opdims[1]))


if __name__ == '__main__':
    main()
