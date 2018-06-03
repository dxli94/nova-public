from scipy.sparse import csr_matrix, csc_matrix
from src.utils.python_sparse_glpk.python_sparse_glpk import LpInstance

import numpy as np
import SuppFuncUtils

import src.utils.plot_constraints as plotter
from utils.DataReader import DataReader
from utils.GlpkWrapper import GlpkWrapper


def add_input_constraints(lp, sys_dyn, tau, shape_info):
    num_row = shape_info['numRow']
    num_col = shape_info['numCol']

    coeff_mat = sys_dyn.coeff_matrix_U
    col_vec = sys_dyn.col_vec_U * tau
    col_vec = col_vec.reshape(1, len(col_vec)).flatten()

    zero_mat = np.zeros((num_row, coeff_mat.shape[1]))
    a_csc = csc_matrix(np.vstack((zero_mat, coeff_mat)))

    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(col_vec)
    lp.set_constraints_csc(a_csc, offset=(0, num_col))

    shape_info['numRow'] += len(col_vec)
    shape_info['numCol'] += a_csc.shape[1]


def add_bloating_constraints(lp, sys_dyn, beta, shape_info):
    num_row = shape_info['numRow']
    num_col = shape_info['numCol']

    coeff_mat = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    col_vec = [beta] * sys_dyn.dim * 2
    zero_mat = np.zeros((num_row, sys_dyn.dim))
    a_csc = csc_matrix(np.vstack((zero_mat, coeff_mat)))

    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(col_vec)
    lp.set_constraints_csc(a_csc, offset=(0, num_col))

    shape_info['numRow'] += len(col_vec)
    shape_info['numCol'] += sys_dyn.dim


def add_equality_constraints(lp, sys_dyn, delta, shape_info):
    num_row = shape_info['numRow']
    num_col = shape_info['numCol']

    iden = np.identity(sys_dyn.dim)
    zero_mat = np.zeros((sys_dyn.dim, num_col - 3*sys_dyn.dim))

    a_csr = csr_matrix(np.hstack((zero_mat, -delta, -iden, -iden)))
    lp.add_rows_equal_zero(sys_dyn.dim)
    lp.add_cols(sys_dyn.dim)
    lp.set_constraints_csr(a_csr, offset=(num_row, 0))

    zero_mat = np.zeros((num_row, sys_dyn.dim))
    a_csc = csc_matrix(np.vstack((zero_mat, iden)))
    lp.set_constraints_csc(a_csc, offset=(0, num_col))

    shape_info['numRow'] += sys_dyn.dim
    shape_info['numCol'] += sys_dyn.dim


def test(shape_info):
    data_reader = DataReader(path2instance="../instances/single_mode_affine_instances/free_ball.txt")
    sys_dynamics = data_reader.read_data()

    dyn_coeff_mat = sys_dynamics.get_dyn_coeff_matrix_A()

    init_coeff_mat = sys_dynamics.init_coeff_matrix
    init_coeff_col = sys_dynamics.init_col_vec
    init_coeff_col = init_coeff_col.reshape(1, len(init_coeff_col)).flatten()
    init_csr = csr_matrix(np.array(init_coeff_mat), dtype=float)

    shape_info['numCol'] += sys_dynamics.dim
    shape_info['numRow'] += len(init_coeff_col)

    tau = 0.01
    delta = SuppFuncUtils.mat_exp(dyn_coeff_mat, tau)
    temp_lp = GlpkWrapper(sys_dynamics.dim)
    # alfa = SuppFuncUtils.compute_alpha(sys_dynamics, tau, temp_lp)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau, temp_lp)

    # import time
    # start = time.time()

    lp = LpInstance()

    lp.add_cols(init_csr.shape[1])
    lp.add_rows_less_equal(init_coeff_col)
    lp.set_constraints_csr(init_csr)

    for i in range(1000):
        direction = np.hstack(([0.0]*6*(i+1), -1, 0))
        add_bloating_constraints(lp, sys_dynamics, beta, shape_info)
        add_input_constraints(lp, sys_dynamics, tau, shape_info)
        add_equality_constraints(lp, sys_dynamics, delta, shape_info)
        print(-np.dot(lp.minimize(direction), direction))

    # end = time.time()
    # print(end-start)


def main():
    shape_info = dict()
    shape_info['numCol'] = 0
    shape_info['numRow'] = 0

    test(shape_info)

if __name__ == '__main__':
    main()

