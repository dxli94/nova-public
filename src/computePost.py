import numpy as np

import SuppFuncUtils
import DataReader

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from Plotter import Plotter


def compute_initial_sf(poly_init, trans_poly_U, l, sys_dynamics, tau):
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()
    delta_tp = np.transpose(SuppFuncUtils.mat_exp(dyn_matrix_A, 1 * tau))
    sf_X0 = poly_init.compute_support_function(np.matmul(delta_tp, l))

    sf_V = tau * trans_poly_U.compute_support_function(l)
    alpha = SuppFuncUtils.compute_alpha(sys_dynamics, tau)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    sf_omega0 = max(sf_X0, sf_X0 + sf_V + alpha * sf_ball)
    return sf_omega0


def compute_sf_w(sys_dynamics, l, trans_poly_U, tau):
    sf_V = tau * trans_poly_U.compute_support_function(l)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    sf_omega = tau * sf_V + beta * sf_ball
    return sf_omega


def compute_post(sys_dynamics, directions, tau):
    ret = []
    init = sys_dynamics.get_dyn_init_X0()
    poly_init = Polyhedron(init[0], init[1])

    dyn_matrix_B = sys_dynamics.get_dyn_matrix_B()
    dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
    dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

    trans_poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                             coeff_matrix_U=dyn_coeff_matrix_U,
                             col_vec_U=dyn_col_vec_U)

    delta_tp = np.transpose(SuppFuncUtils.mat_exp(dyn_matrix_A, tau))

    for idx in range(len(directions)):
        for n in time_frames:
            # delta_tp = np.transpose(mat_exp(A, n * time_interval))
            if n == 0:
                prev_r = directions[idx]
                prev_s = compute_initial_sf(poly_init, trans_poly_U, directions[idx], sys_dynamics, tau)
                ret.append([prev_s])
            else:
                r = np.matmul(delta_tp, prev_r)
                s = prev_s + compute_sf_w(sys_dynamics, r, trans_poly_U, tau)
                sf = poly_init.compute_support_function(r) + s

                ret[-1].append(sf)

                prev_r = r
                prev_s = s

    return np.array(ret)


def get_images(sf_mat, directions):
    ret = []

    d_mat = np.array(directions)
    sf_mat = np.transpose(sf_mat)
    for sf_row in sf_mat:
        ret.append(Polyhedron(d_mat, np.reshape(sf_row, (len(sf_row), 1))))
    return ret


def main():
    directions, sys_dynamics = DataReader.read_data()

    sf_mat = compute_post(sys_dynamics, directions, tau=SAMP_FREQ)
    # print(sf_mat)
    images_by_time = get_images(sf_mat, directions)
    for image in images_by_time:
        print(image.vertices)
    # polygon_plotter(images_by_time)
    plHelper = Plotter(images_by_time)
    plHelper.Print()


if __name__ == '__main__':
    TIME_HORIZON = 10
    SAMP_FREQ = 1
    time_frames = range(int(np.floor(TIME_HORIZON / SAMP_FREQ)))

    main()

