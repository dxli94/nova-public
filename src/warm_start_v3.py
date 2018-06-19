import numpy as np
import time

import SuppFuncUtils as SuppFuncUtils
from ConvexSet.Polyhedron import Polyhedron
from utils.DataReader import DataReader
from utils.GlpkWrapper import GlpkWrapper

from timerutil import Timers

def update_delta_list(sys_dynamics, tau, delta_list):
    dyn_coeff_mat = sys_dynamics.get_dyn_coeff_matrix_A()
    # if we want to be general, in the non-linear case, delta depends on
    # matrix A, which changes over time. So here do not pre-compute delta
    # or pass it as a param. Instead, recompute it every time.
    delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, tau)

    if len(delta_list) == 0:
        delta_list = np.array([delta])
    else:
        delta_list = np.tensordot(delta, delta_list, axes=((1),(1))).swapaxes(0, 1)
    delta_list = np.vstack((delta_list, [np.eye(sys_dynamics.dim)]))

    return delta_list


def update_input_bounds(ub, lb, next_ub, next_lb):
    return np.append(lb, next_lb), np.append(ub, next_ub)


def get_input_bounds(sys_dyn, beta, tau):
    return [-0.0004921390956875789, -0.09859213909568759], [0.0004921390956875789, -0.09760786090431242]


def test():
    data_reader = DataReader(path2instance="../instances/single_mode_affine_instances/free_ball.txt")
    sys_dynamics = data_reader.read_data()
    tau = 0.01
    mylp = GlpkWrapper(sys_dynamics.dim)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau, mylp)

    delta_list = []

    init_poly = Polyhedron(coeff_matrix=sys_dynamics.init_coeff_matrix,
                           col_vec=sys_dynamics.init_col_vec)
    vertices = np.array(init_poly.vertices)
    init_ub = np.amax(vertices, axis=0)
    init_lb = np.amin(vertices, axis=0)
    input_ub = init_ub
    input_lb = init_lb

    max_iters = 2000

    Timers.tic('total')

    total_val = 0

    start = time.time()
    for i in range(max_iters):
        delta_list = update_delta_list(sys_dynamics, tau, delta_list)
        # in nonlinear case, input-bound will be re-computed
        next_lb, next_ub = get_input_bounds(sys_dynamics, beta, tau)
        input_lb, input_ub = update_input_bounds(input_ub, input_lb, next_ub, next_lb)
        factors = delta_list.transpose(1, 0, 2).reshape(2, -1)

        Timers.tic('accumulate loop')
        
        for j in range(factors.shape[0]):
            #Timers.tic('list comprehension') # ~ 6.4 sec, np.dot also takes time
            #max_point = [input_ub[idx] if k >= 0 else input_lb[idx] for idx, k in enumerate(factors[j, :])]
            #min_point = [input_lb[idx] if k >= 0 else input_ub[idx] for idx, k in enumerate(factors[j, :])]
            #Timers.toc('list comprehension')

            # Timers.tic('make min/max points') # ~ 4 seconds, np.dot much faster
            row = factors[j, :]
            # max_point = np.empty(len(row))
            # min_point = np.empty(len(row))
            #
            # # for idx in range(len(row)):
            # #     k = row[idx]
            # #
            # #     if k >= 0:
            # #         max_point[idx] = input_ub[idx]
            # #         min_point[idx] = input_lb[idx]
            # #     else:
            # #         max_point[idx] = input_lb[idx]
            # #         min_point[idx] = input_ub[idx]

            Timers.tic('np.clip')
            pos_clip = np.clip(a=row, a_min=0, a_max=np.inf)
            neg_clip = np.clip(a=row, a_min=-np.inf, a_max=0)
            Timers.toc('np.clip')

            Timers.tic('np.dot')
            maxval = pos_clip.dot(input_ub) + neg_clip.dot(input_lb)
            minval = neg_clip.dot(input_ub) + pos_clip.dot(input_lb)
            Timers.toc('np.dot')

            # Timers.toc('make min/max points')
            #
            # Timers.tic('np.dot')
            # maxval = np.dot(row, max_point)
            # minval = np.dot(row, min_point)
            # Timers.toc('np.dot')

            # print('maxval_new is {}, maxval is {}.'.format(maxval_new, maxval))
            # print('minval_new is {}, minval is {}.'.format(minval_new, minval))
            #
            # exit()

            if j == 0:
                total_val += maxval

        Timers.toc('accumulate loop')

        if i % 100 == 0:
            print('100 iterations in {} secs, in total {} iterations.'.format(time.time()-start, i))
            start = time.time()

    # original 9.1 seconds, -1280053.0497008062
    Timers.toc('total')
    Timers.print_stats()
    print('in total {} iterations.'.format(max_iters))
    print(total_val)

def main():
    test()


if __name__ == '__main__':
    main()
