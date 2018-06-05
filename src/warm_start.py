from scipy.sparse import csr_matrix, csc_matrix
from utils.python_sparse_glpk.python_sparse_glpk import LpInstance

import numpy as np
import SuppFuncUtils as SuppFuncUtils

from utils.DataReader import DataReader
from utils.GlpkWrapper import GlpkWrapper


def add_init_constraints(lp, init_coeff_mat, init_col_vec):
    init_coeff_mat = init_coeff_mat
    init_coeff_col = init_col_vec
    init_coeff_col = init_coeff_col.reshape(1, len(init_coeff_col)).flatten()
    init_csr = csr_matrix(np.array(init_coeff_mat), dtype=float)

    lp.add_cols(init_csr.shape[1])
    lp.add_rows_less_equal(init_coeff_col)
    lp.set_constraints_csr(init_csr)


def add_input_constraints(lp, sys_dyn, tau):
    num_row = lp.get_num_rows()
    num_col = lp.get_num_cols()

    coeff_mat = sys_dyn.coeff_matrix_U
    col_vec = sys_dyn.col_vec_U * tau
    col_vec = col_vec.reshape(1, len(col_vec)).flatten()

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


def add_equality_constraints(lp, sys_dyn, delta):
    num_row = lp.get_num_rows()
    num_col = lp.get_num_cols()

    iden = np.identity(sys_dyn.dim)
    zero_mat = np.zeros((sys_dyn.dim, num_col - 3*sys_dyn.dim))

    a_csr = csr_matrix(np.hstack((zero_mat, -delta, -iden, -iden)))
    lp.add_rows_equal_zero(sys_dyn.dim)
    lp.add_cols(sys_dyn.dim)
    lp.set_constraints_csr(a_csr, offset=(num_row, 0))

    zero_mat = np.zeros((num_row, sys_dyn.dim))
    a_csc = csc_matrix(np.vstack((zero_mat, iden)))
    lp.set_constraints_csc(a_csc, offset=(0, num_col))


def add_constraints(lp, sys_dyn, tau, beta, delta):
    add_bloating_constraints(lp, sys_dyn, beta)
    add_input_constraints(lp, sys_dyn, tau)
    add_equality_constraints(lp, sys_dyn, delta)


def test():
    data_reader = DataReader(path2instance="../instances/single_mode_affine_instances/free_ball.txt")
    sys_dynamics = data_reader.read_data()

    dyn_coeff_mat = sys_dynamics.get_dyn_coeff_matrix_A()

    init_coeff_mat = sys_dynamics.init_coeff_matrix
    init_coeff_col = sys_dynamics.init_col_vec
    init_coeff_col = init_coeff_col.reshape(1, len(init_coeff_col)).flatten()
    init_csr = csr_matrix(np.array(init_coeff_mat), dtype=float)

    tau = 0.01
    delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, tau)
    mylp = GlpkWrapper(sys_dynamics.dim)
    # alfa = SuppFuncUtils.compute_alpha(sys_dynamics, tau, temp_lp)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau, mylp)

    # import time
    # start = time.time()

    lp = LpInstance()

    lp.add_cols(init_csr.shape[1])
    lp.add_rows_less_equal(init_coeff_col)
    lp.set_constraints_csr(init_csr)

    import time

    direction = np.array([1, 0])
    # direction = np.array([-1, 0])
    # direction = np.array([0, -1])
    # direction = np.array([0, 1])

    start = time.time()
    for i in range(2000):
        c = np.hstack(([0.0]*6*(i+1), -direction))
        # print(c)
        add_bloating_constraints(lp, sys_dynamics, beta)
        add_input_constraints(lp, sys_dynamics, tau)
        add_equality_constraints(lp, sys_dynamics, delta)
        val = -np.dot(lp.minimize(c), c)
        if i % 100 == 0 and i != 0:
            print('100 iterations in {} secs, in total {} iterations.'.format(time.time()-start, i))
            start = time.time()

    # end = time.time()
    # print(end-start)


def main():
    test()

if __name__ == '__main__':
    main()

