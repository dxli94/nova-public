import numpy as np
import time
from scipy.sparse import csr_matrix, csc_matrix

import SuppFuncUtils as SuppFuncUtils
from utils.python_sparse_glpk.python_sparse_glpk import LpInstance
from utils.DataReader import DataReader
from utils.GlpkWrapper import GlpkWrapper


def add_init_constraints(lp, sys_dyn, init_coeff_mat, init_col_vec):
    # n columns for state variables
    lp.add_cols(sys_dyn.dim)
    lp.add_rows_equal_zero(sys_dyn.dim)

    init_coeff_mat = init_coeff_mat
    init_coeff_col = init_col_vec
    init_coeff_col = init_coeff_col.reshape(1, len(init_coeff_col)).flatten()

    # padding with zeros
    zeros_mat = np.zeros((init_coeff_mat.shape[0], sys_dyn.dim))
    init_csr = csr_matrix(np.hstack((zeros_mat, init_coeff_mat)), dtype=float)
    lp.add_cols(sys_dyn.dim)
    lp.add_rows_less_equal(init_coeff_col)
    lp.set_constraints_csr(init_csr, offset=(sys_dyn.dim, 0))

    eye = np.eye(sys_dyn.dim)
    eye_csr = csr_matrix(eye)
    lp.set_constraints_csr(eye_csr)


def add_input_constraints(lp, sys_dyn, beta, tau):
    num_row = lp.get_num_rows()
    num_col = lp.get_num_cols()

    coeff_mat = sys_dyn.coeff_matrix_U
    sys_col_vec = sys_dyn.col_vec_U * tau
    # system input
    col_vec_term1 = sys_col_vec.reshape(1, len(sys_col_vec)).flatten()
    # bloating
    col_vec_term2 = [beta] * sys_dyn.dim * 2
    # input and bloating can be summed up as they are all boxes.
    # The sum of minkowski sum over box1 and box 2 is equiv. to
    # the minkowski sum over sum of box1 and box2.
    col_vec = col_vec_term1 + col_vec_term2

    zero_mat = np.zeros((num_row, coeff_mat.shape[1]))
    a_csc = csc_matrix(np.vstack((zero_mat, coeff_mat)))

    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(col_vec)
    lp.set_constraints_csc(a_csc, offset=(0, num_col))


def add_bloating_constraints(lp, sys_dyn, beta):
    num_row = lp.get_num_rows()
    num_col = lp.get_num_cols()

    coeff_mat = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    col_vec = [beta] * sys_dyn.dim * 2
    zero_mat = np.zeros((num_row, sys_dyn.dim))
    a_csc = csc_matrix(np.vstack((zero_mat, coeff_mat)))

    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(col_vec)
    lp.set_constraints_csc(a_csc, offset=(0, num_col))


def update_first_n_col(lp, sys_dynamics, tau, delta_list):
    dyn_coeff_mat = sys_dynamics.get_dyn_coeff_matrix_A()
    # if we want to be general, in the non-linear case, delta depends on
    # matrix A, which changes over time. So here do not pre-compute delta
    # or pass it as a param. Instead, recompute it every time.
    delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, tau)

    # todo for better efficiency, we could try to rewrite the following lines using numpy broadcast
    temp_delta_list = []
    if len(delta_list) == 0:
        temp_delta_list.append(delta)

    for x in delta_list:
        temp_delta_list.append(np.dot(delta, x))
    temp_delta_list.append(np.eye(sys_dynamics.dim))
    delta_list = temp_delta_list

    # todo can be better written using reduce
    tail = None
    for x in delta_list:
        if tail is not None:
            tail = np.hstack((tail, -x))
        else:
            tail = -x

    temp = np.hstack((np.eye(sys_dynamics.dim), tail))
    temp_csr = csr_matrix(temp)
    lp.set_constraints_csr(temp_csr)

    return delta_list


def test():
    data_reader = DataReader(path2instance="../instances/single_mode_affine_instances/free_ball.txt")
    sys_dynamics = data_reader.read_data()

    delta_list = []

    init_coeff_mat = sys_dynamics.init_coeff_matrix
    init_coeff_col = sys_dynamics.init_col_vec
    init_coeff_col = init_coeff_col.reshape(1, len(init_coeff_col)).flatten()

    tau = 0.01
    mylp = GlpkWrapper(sys_dynamics.dim)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau, mylp)

    lp = LpInstance()

    add_init_constraints(lp, sys_dynamics, init_coeff_mat, init_coeff_col)

    # direction = np.array([1, 0])
    # direction = np.array([-1, 0])
    # direction = np.array([0, -1])
    direction = np.array([0, 1])

    start = time.time()
    for i in range(100):
        add_input_constraints(lp, sys_dynamics, beta, tau)
        delta_list = update_first_n_col(lp, sys_dynamics, tau, delta_list)

        # lp.reset_lp()

        c = np.hstack((-direction, np.zeros(lp.get_num_cols() - sys_dynamics.dim)))
        val = -np.dot(lp.minimize(c), c)
        print(val)

        if i % 100 == 0 and i != 0:
            print('100 iterations in {} secs, in total {} iterations.'.format(time.time()-start, i))
            start = time.time()


def main():
    # test()
    test()

if __name__ == '__main__':
    main()

