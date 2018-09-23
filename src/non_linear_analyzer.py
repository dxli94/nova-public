import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import utils.Simulator as simu
from AffinePostOpt import PostOperator as AffinePostOperator
from Hybridisation.NonlinPostOpt import NonlinPostOpt
from SysDynamics import GeneralDynamics
from utils import SuppFuncUtils
from utils.DataReader import JsonReader
from utils.Plotter import Plotter


def main():
    start_time = time.time()
    # # ============== setting up ============== #
    try:
        path = sys.argv[1]
    except IndexError:
        path = '../instances/non_linear_instances/vanderpol.json'
        # path = '../instances/non_linear_instances/predator_prey.json'
        # path = '../instances/non_linear_instances/2d_water_tank.json'
        # path = '../instances/non_linear_instances/brusselator.json'
        # path = '../instances/non_linear_instances/jet_engine.json'
        # path = '../instances/non_linear_instances/free_ball.json'
        # path = '../instances/non_linear_instances/constant_moving.json'
        # path = '../instances/non_linear_instances/buckling_column.json'
        # path = '../instances/non_linear_instances/pbt.json'
        # path = '../instances/non_linear_instances/pbt_y.json'
        # path = '../instances/non_linear_instances/2d_controller.json'
        # path = '../instances/non_linear_instances/3d_controller.json'
        # path = '../instances/non_linear_instances/watt_steam.json'
        # path = '../instances/non_linear_instances/lacoperon.json'
        # path = '../instances/non_linear_instances/roessler_attractor.json'
        # path = '../instances/non_linear_instances/coupled_vanderpol_4d.json'
        # path = '../instances/non_linear_instances/coupled_vanderpol_6d.json'
        # path = '../instances/non_linear_instances/coupled_vanderpol_8d.json'
        # path = '../instances/non_linear_instances/spring_pendulum.json'
        # path = '../instances/non_linear_instances/lorentz_system.json'
        # path = '../instances/non_linear_instances/biology_1.json'
        # path = '../instances/non_linear_instances/biology_2.json'
        # path = '../instances/non_linear_instances/laub_loomis.json'
        # path = '../instances/non_linear_instances/laub_loomis_large_init.json'

    model_name = path.split('/')[-1].split('.')[0]
    print('reading model file: {}'.format(model_name))
    configs = JsonReader(path).read()
    print('Finished.')
    time_horizon = configs['time_horizon']
    tau = configs['sampling_time']
    direction_type = configs['direction_type']
    dim = configs['dim']
    start_epsilon = configs['start_epsilon']
    non_linear_dynamics = configs['dynamics']
    state_vars = configs['state_variables']
    is_linear = configs['is_linear']
    init_coeff = np.array(configs['init_coeff'])
    init_col = np.array(configs['init_col'])
    opdims = configs['opvars']
    simu_model = configs['simu_model']
    pseudo_var = False
    scaling_per = configs['scaling_per']
    scaling_cutoff = configs['scaling_cutoff']

    try:
        unsafe_coeff = np.array(configs['unsafe_coeff'])
        unsafe_col = np.array(configs['unsafe_col'])
    except KeyError:
        unsafe_coeff = unsafe_col = None

    directions = SuppFuncUtils.generate_directions(direction_type, dim)

    id_to_vars = {}
    for i, var in enumerate(state_vars):
        id_to_vars[i] = var
    non_linear_dynamics = GeneralDynamics(id_to_vars, *non_linear_dynamics)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    np.set_printoptions(precision=100)
    nonlin_post_opt = NonlinPostOpt(dim=dim,
                                    nonlin_dyn=non_linear_dynamics,
                                    time_horizon=time_horizon,
                                    tau=tau,
                                    init_coeff=init_coeff,
                                    init_col=init_col,
                                    is_linear=is_linear,
                                    directions=directions,
                                    start_epsilon=start_epsilon,
                                    scaling_per=scaling_per,
                                    scaling_cutoff=scaling_cutoff,
                                    id_to_vars=id_to_vars,
                                    unsafe_coeff=unsafe_coeff,
                                    unsafe_col=unsafe_col)
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

    print('\nStart simulations.')
    simu_traj = run_simulate(time_horizon, simu_model, init_coeff, init_col)
    # simu.save_simu_traj(simu_traj, simu_dir_path)
    print('\nSimulations done.')

    print('Saving images...')
    img_start_time = time.time()
    make_plot(dim, directions, sf_mat, model_name, simu_traj, poly_dir_path)
    print('Finished in {} secs.'.format(time.time() - img_start_time))
    print('Total running time: {:.2f}'.format(time.time() - start_time))


def run_simulate(time_horizon, model, init_coeff, init_col):
    from ConvexSet.Polyhedron import Polyhedron
    from ConvexSet.HyperBox import HyperBox
    import random
    vertices = Polyhedron(init_coeff, init_col).get_vertices()
    init_set = HyperBox(vertices)
    n = 100

    bounds = init_set.bounds.T

    simu_points = []
    for i in range(n):
        p = tuple(random.uniform(*b) for b in bounds)
        simu_points.append(p)

    simu_points.extend(list(vertices))

    simu_traj = simu.simulate(time_horizon, model, simu_points)
    return simu_traj


def create_pickle(filename, data):
    with open(filename, 'wb') as opfile:
        pickle.dump(data, opfile)


def make_plot(dim, directions, sf_mat, model_name, simu_traj, poly_dir):
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
            for xs in simu_traj:
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
