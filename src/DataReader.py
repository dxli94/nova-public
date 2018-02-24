import numpy as np

from src.SysDynamics import SysDynamics


def read_data():
    directions = [
        np.array([-1, 0]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([0, 1]),
        # np.array([1, 1]),
        # np.array([1, -1]),
        # np.array([-1, -1]),
        # np.array([-1, 1])
    ]
    init_coeff_matrix_X0 = [[-1, 0],  # -x1 <= 0
                            [1, 0],  # x1 <= 2
                            [0, -1],  # -x2 <= 2
                            [0, 1]]  # x2 <= 0

    init_col_vec_X0 = [[0],
                       [2],
                       [0],
                       [2]]

    # dynamics_matrix_A = [[0, 1],
    #                      [-2, 0]]

    dynamics_matrix_A = [[0, 0],
                         [0, 0]]

    dynamics_matrix_B = np.identity(2)

    # U is a square with inf_norm = 2
    dynamics_coeff_matrix_U = [[-1, 0],  # u1 >= 0
                               [1, 0],  # u1 <= 1
                               [0, -1],  # u2 >= 0
                               [0, 1]]  # u2 <= 1
    dynamics_col_vec_U = [[-1],
                          [1],
                          [-1],
                          [1]]

    sys_dynamics = SysDynamics(init_coeff_matrix_X0=np.array(init_coeff_matrix_X0),
                               init_col_vec_X0=np.array(init_col_vec_X0),
                               dynamics_matrix_A=np.array(dynamics_matrix_A),
                               dynamics_matrix_B=np.array(dynamics_matrix_B),
                               dynamics_coeff_matrix_U=np.array(dynamics_coeff_matrix_U),
                               dynamics_col_vec_U=np.array(dynamics_col_vec_U))

    return directions, sys_dynamics
