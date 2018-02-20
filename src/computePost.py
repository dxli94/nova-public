import numpy as np

import SuppFuncUtils
from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from Plotter import Plotter
from SysDynamics import SysDynamics


def compute_initial_sf(poly_init, trans_poly_U, l, sys_dynamics, tau):
    delta_tp = np.transpose(SuppFuncUtils.mat_exp(dynamics_matrix_A, 1 * tau))
    sf_X0 = poly_init.compute_support_function(np.matmul(delta_tp, l))

    sf_V = tau * trans_poly_U.compute_support_function(l)
    alpha = SuppFuncUtils.compute_alpha(sys_dynamics, tau)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    sf_omega0 = max(sf_X0, sf_X0 + sf_V + alpha * sf_ball)

    return sf_omega0


def compute_sf_w(l, trans_poly_U, tau):
    sf_V = tau * trans_poly_U.compute_support_function(l)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    sf_omega = tau * sf_V + beta * sf_ball
    return sf_omega


def compute_post(sys_dynamics, tau):
    ret = []
    init = sys_dynamics.get_dyn_init_X0()
    poly_init = Polyhedron(init[0], init[1])

    dyn_matrix_B = np.matrix(sys_dynamics.get_dyn_matrix_B())
    dyn_coeff_matrix_U = np.matrix(sys_dynamics.get_dyn_coeff_matrix_U())
    dyn_col_vec_U = np.matrix(sys_dynamics.get_dyn_col_vec_U())
    trans_poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                             coeff_matrix=dyn_coeff_matrix_U,
                             col_vec_U=dyn_col_vec_U)

    delta_tp = np.transpose(SuppFuncUtils.mat_exp(dynamics_matrix_A, tau))

    for idx in range(len(directions)):
        for n in time_frames:
            # delta_tp = np.transpose(mat_exp(A, n * time_interval))
            if n == 0:
                prev_r = directions[idx]
                prev_s = compute_initial_sf(poly_init, trans_poly_U, directions[idx], sys_dynamics, tau)
                ret.append([prev_s])
            else:
                r = np.matmul(delta_tp, prev_r)
                s = prev_s + compute_sf_w(r, trans_poly_U, tau)
                sf = poly_init.compute_support_function(r) + s

                ret[-1].append(sf)

                prev_r = r
                prev_s = s

    return np.matrix(ret)


def get_images(sf_mat, directions):
    ret = []

    d_mat = np.matrix(directions)
    sf_mat = np.transpose(np.matrix(sf_mat))
    for sf_row in sf_mat:
        ret.append(Polyhedron(d_mat, np.transpose(sf_row)))
    return ret


def main():
    pass


if __name__ == '__main__':
    TIME_EPLASE = 1.5
    TIME_INTERVAL = 0.3
    

    main()

    directions = [
        np.array([-1, 0]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([1, -1]),
        np.array([-1, -1]),
        np.array([-1, 1]),
    ]
    init_coeff_matrix_X0 = [[-1, 0],  # -x1 <= 0
                    [1, 0],  # x1 <= 2
                    [0, -1],  # -x2 <= 0.5
                    [0, 1]]  # x2 <= 1

    init_col_vec_X0 = [[0],
                       [2],
                       [0.5],
                       [0]]

    dynamics_matrix_A = [[0, 1],
                         [-2, 0]]
    dynamics_matrix_B = np.identity(2)

    # U is a square with inf_norm = 2
    dynamics_coeff_matrix_U = [[-1, 0],  # u1 >= 0
                               [1, 0],   # u1 <= 1
                               [0, -1],  # u2 >= 0
                               [0, 1]]   # u2 <= 1
    dynamics_col_vec_U = [[0],
                          [1],
                          [0],
                          [1]]

    sys_dynamics = SysDynamics(dynamics_matrix_A=dynamics_matrix_A,
                               init_coeff_matrix_X0=init_coeff_matrix_X0,
                               init_col_vec_X0=init_col_vec_X0,
                               dynamics_matrix_B=dynamics_matrix_B,
                               dynamics_coeff_matrix_U=dynamics_coeff_matrix_U,
                               dynamics_col_vec_U=dynamics_col_vec_U)

    time_frames = range(int(np.ceil(TIME_EPLASE / TIME_INTERVAL)) + 1)

    # sfp = SupportFunctionProvider(poly)
    sf_mat = compute_post(sys_dynamics, tau=TIME_INTERVAL)
    # print(sf_mat)
    images_by_time = get_images(sf_mat, directions)
    for image in images_by_time:
        print(image.vertices)
    # polygon_plotter(images_by_time)
    plHelper = Plotter(images_by_time)
    plHelper.Print()
