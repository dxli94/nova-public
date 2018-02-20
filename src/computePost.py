import numpy as np

import SuppFuncUtils
from Plotter import Plotter
from Polyhedron import Polyhedron
from SysDynamics import SysDynamics


def compute_initial_sf(init_poly, l, sys_dynamics, tau):
    delta_tp = np.transpose(SuppFuncUtils.mat_exp(dynamics_matrix_A, 1 * tau))
    v_matrix = np.matmul(sys_dynamics.get_dyn_matrix_b(), sys_dynamics.get_dyn_input_matrix())
    v_poly = Polyhedron(v_matrix.tolist())

    sf_X0 = init_poly.compute_support_function(np.matmul(delta_tp, l))

    sf_V = tau * v_poly.compute_support_function(l)
    alpha = SuppFuncUtils.compute_alpha(sys_dynamics, tau)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    sf_omega0 = max(sf_X0, sf_X0 + sf_V + alpha * sf_ball)

    return sf_omega0


def compute_sf_w(l, tau):
    v_matrix = np.matmul(sys_dynamics.get_dyn_matrix_b(), sys_dynamics.get_dyn_input_matrix())
    v_poly = Polyhedron(v_matrix)
    sf_V = tau * v_poly.compute_support_function(l)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    sf_omega = tau * sf_V + beta * sf_ball
    return sf_omega


def compute_post(sys_dynamics, tau):
    ret = []
    init = sys_dynamics.get_dyn_init()
    init_poly = Polyhedron(init[0], init[1])

    delta_tp = np.transpose(SuppFuncUtils.mat_exp(dynamics_matrix_A, tau))

    for idx in range(len(directions)):
        for n in time_frames:
            # delta_tp = np.transpose(mat_exp(A, n * time_interval))
            if n == 0:
                prev_r = directions[idx]
                prev_s = compute_initial_sf(init_poly, directions[idx], sys_dynamics, tau)
                ret.append([prev_s])
            else:
                r = np.matmul(delta_tp, prev_r)
                s = prev_s + compute_sf_w(r, tau)
                sf = init_poly.compute_support_function(r) + s

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
    TIME_EPLASE = 1
    TIME_INTERVAL = 0.5
    

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
    coeff_matrix = [[-1, 0],  # -x1 <= 1
                    [1, 0],  # x1 <= 2
                    [0, -1],  # -x2 <= 0.5
                    [0, 1]  # x2 <= 1
                    ]
    col_vec = [[0], [2], [0.5], [0]]
    dynamics_matrix_A = [[0, 1],
                         [-2, 0]]
    dynamics_matrix_B = np.identity(4)
    dynamics_matrix_input = [[1, 0],
                             [0, 1],
                             [0, -1],
                             [-1, 0]]

    # print(np.matmul(dynamics_matrix_B, dynamics_matrix_input))
    # exit()

    sys_dynamics = SysDynamics(matrix_a=dynamics_matrix_A,
                               matrix_init_coeff=coeff_matrix,
                               matrix_init_col=col_vec,
                               matrix_b=dynamics_matrix_B,
                               matrix_input=dynamics_matrix_input)

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
