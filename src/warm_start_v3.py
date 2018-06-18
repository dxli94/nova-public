import numpy as np
import time

import SuppFuncUtils as SuppFuncUtils
from ConvexSet.Polyhedron import Polyhedron
from utils.DataReader import DataReader
from utils.GlpkWrapper import GlpkWrapper


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

    overall_start = time.time()

    total_val = 0

    start = time.time()
    for i in range(max_iters):
        delta_list = update_delta_list(sys_dynamics, tau, delta_list)
        # in nonlinear case, input-bound will be re-computed
        next_lb, next_ub = get_input_bounds(sys_dynamics, beta, tau)
        input_lb, input_ub = update_input_bounds(input_ub, input_lb, next_ub, next_lb)
        factors = delta_list.transpose(1, 0, 2).reshape(2, -1)

        for j in range(factors.shape[0]):
            max_point = [input_ub[idx] if k >= 0 else input_lb[idx] for idx, k in enumerate(factors[j, :])]
            min_point = [input_lb[idx] if k >= 0 else input_ub[idx] for idx, k in enumerate(factors[j, :])]

            maxval = np.dot(factors[j, :], max_point)
            minval = np.dot(factors[j, :], min_point)

            if j == 0:
                total_val += maxval

        if i % 100 == 0:
            print('100 iterations in {} secs, in total {} iterations.'.format(time.time()-start, i))
            start = time.time()

    print('Overall time is {} secs, in total {} iterations.'.format(time.time()-overall_start, max_iters))
    print(total_val)

def main():
    test()


if __name__ == '__main__':
    main()