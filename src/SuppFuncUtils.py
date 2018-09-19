import numpy as np
from scipy.linalg import expm
import cvxopt as cvx

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from SysDynamics import AffineDynamics


def mat_exp(A, tau):
    return expm(np.multiply(A, tau))


def support_unitball_infnorm(direction):
    # unit ball for f
    return sum([abs(elem) for elem in direction])


def compute_log_infnorm(A):
    return max(np.sum(A, axis=1))


def compute_log_2norm(A):
    eigns, _ = np.linalg.eig((A + A.T)/2)
    max_eign = max(eigns)

    return max_eign


def compute_alpha(sys_dynamics, tau, lp):
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

    # if dyn_matrix_A is a zero-matrix, no need to perform bloating
    if not np.any(dyn_matrix_A):
        return 0

    dyn_matrix_B = sys_dynamics.get_dyn_matrix_B()
    dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
    dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()
    dyn_matrix_init = sys_dynamics.get_dyn_init_X0()

    norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    # two_norm_a = compute_log_2norm(dyn_matrix_A)
    # log_norm_a = compute_log_infnorm(dyn_matrix_A)
    # norm_a = log_norm_a
    # norm_a = two_norm_a

    tt1 = np.exp(tau * norm_a)

    I_max_norm = Polyhedron(coeff_matrix=dyn_matrix_init[0],
                            col_vec=dyn_matrix_init[1]).compute_max_norm(lp)
    poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                       coeff_matrix_U=dyn_coeff_matrix_U,
                       col_vec_U=dyn_col_vec_U)

    v_max_norm = poly_U.compute_max_norm(lp)

    return (tt1 - 1 - tau * norm_a) * (I_max_norm + (v_max_norm / norm_a))


def compute_beta(sys_dynamics, tau, lp):
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

    # if dyn_matrix_A is a zero-matrix, no need to perform bloating
    if not np.any(dyn_matrix_A):
        return 0

    dyn_matrix_B = sys_dynamics.get_dyn_matrix_B()
    dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
    dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()

    norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    # two_norm_a = compute_log_2norm(dyn_matrix_A)
    # log_norm_a = compute_log_infnorm(dyn_matrix_A)
    # norm_a = log_norm_a
    # norm_a = two_norm_a

    # ========
    tt1 = np.exp(tau * norm_a)

    poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                       coeff_matrix_U=dyn_coeff_matrix_U,
                       col_vec_U=dyn_col_vec_U)

    v_max_norm = poly_U.compute_max_norm(lp)

    return (tt1 - 1 - tau * norm_a) * (v_max_norm / norm_a)


def compute_beta_no_offset(sys_dynamics, tau):
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

    # if dyn_matrix_A is a zero-matrix, no need to perform bloating
    if not np.any(dyn_matrix_A):
        return 0

    norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
    dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()

    tt1 = np.exp(tau * norm_a)

    lb, ub = [], []

    for i in range(dyn_coeff_matrix_U.shape[0]):
        coeff_row = dyn_coeff_matrix_U[i]
        col_row = dyn_col_vec_U[i][0]

        if sum(coeff_row) == 1:
            ub.append(col_row)
        else:
            lb.append(-col_row)

    diff_lb, diff_ub = [], []

    for l, u in zip(lb, ub):
        diff_lb.append(l - u)
        diff_ub.append(u - l)

    D_v = max(np.amax(np.abs(diff_lb), axis=0), np.amax(np.abs(diff_ub), axis=0))
    R_w = D_v / 2

    return (tt1 - 1 - tau * norm_a) * (R_w / norm_a)


def compute_reach_params(sys_dynamics, tau):
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

    # if dyn_matrix_A is a zero-matrix, no need to perform bloating
    if not np.any(dyn_matrix_A):
        return 0, 0

    norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
    dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()
    dyn_col_init_X0 = sys_dynamics.get_dyn_init_X0()[1]

    # === e^(\norm{A} tau)
    tt1 = np.exp(tau * norm_a)

    lb, ub = [], []

    I_max_norm = -1
    v_max_norm = -1

    # ==== I_max_norm =====
    for col_val in dyn_col_init_X0:
        I_max_norm = max(I_max_norm, abs(col_val))

    # ==== v_max_norm  & D_v =====
    for i in range(dyn_coeff_matrix_U.shape[0]):
        coeff_row = dyn_coeff_matrix_U[i]
        col_val = dyn_col_vec_U[i][0]
        v_max_norm = max(v_max_norm, abs(col_val))

        if sum(coeff_row) == 1:
            ub.append(col_val)
        else:
            lb.append(-col_val)

    diff_lb, diff_ub = [], []

    for l, u in zip(lb, ub):
        diff_lb.append(l - u)
        diff_ub.append(u - l)

    D_v = max(np.amax(np.abs(diff_lb), axis=0), np.amax(np.abs(diff_ub), axis=0))
    R_w = D_v / 2

    tt2 = tt1 - 1 - tau * norm_a

    alpha = tt2 * (I_max_norm + (v_max_norm / norm_a))
    beta = tt2 * (R_w / norm_a)
    delta = mat_exp(dyn_matrix_A, tau)

    return alpha, beta, delta


def generate_directions(direction_type, dim):
    direction_generator = []

    if direction_type == 0:  # box
        for i in range(dim):
            direction = np.zeros(dim)
            direction[i] = 1
            direction_generator.append(direction)

            direction = np.zeros(dim)
            direction[i] = -1
            direction_generator.append(direction)

        return np.array(direction_generator)

    elif direction_type == 1:  # octagonal
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    direction = np.zeros(dim)
                    direction[i] = 1
                    direction_generator.append(direction)

                    direction = np.zeros(dim)
                    direction[i] = -1
                    direction_generator.append(direction)
                else:
                    direction = np.zeros(dim)
                    direction[i] = 1
                    direction[j] = 1
                    direction_generator.append(direction)

                    direction = np.zeros(dim)
                    direction[i] = 1
                    direction[j] = -1
                    direction_generator.append(direction)

                    direction = np.zeros(dim)
                    direction[i] = -1
                    direction[j] = 1
                    direction_generator.append(direction)

                    direction = np.zeros(dim)
                    direction[i] = -1
                    direction[j] = -1
                    direction_generator.append(direction)

    elif direction_type == 2:  # uniform
        assert dim == 2  # for now only supports 2D, n-D is a bit trickier

        n = 8  # set as a user option later

        direction_generator = []
        theta = 2 * np.pi / n
        for k in range(n):
            l = np.array([np.cos(k*theta), np.sin(k*theta)])
            direction_generator.append(l)

    return np.array(direction_generator)


def compute_support_function_singular(c, l):
    return np.dot(l, c)


def mat_exp_int(A, t_min, t_max, nbins=5):
    iden = np.identity(A.shape[0])
    f = lambda x: expm(A*x)-iden
    xv = np.linspace(t_min, t_max, nbins)
    result = np.apply_along_axis(f, 0, xv.reshape(1, -1))
    return np.trapz(result, xv)


if __name__ == '__main__':
    # direction_type = 1
    # dim = 3
    #
    # print(len(generate_directions(direction_type, dim)))
    A = np.array([[1, 0], [0, 1]])
    T = 0.01
    rv = mat_exp_int(A, T)
    print(rv)

    inv_A = np.linalg.inv(A)

    print(np.dot((mat_exp(A, T) - mat_exp(A, 0)), inv_A))

    # init_mat_X0 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    # init_col_X0 = np.array([[1], [1], [1], [1]])
    #
    # coeff_matrix_B = np.array([[1, 0], [0, 1]])
    #
    # matrix_A = np.array([[1, 0], [0, 1]])
    # coeff_col = np.array([[1], [-1]])
    #
    # poly_w_0 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    # poly_w_1 = np.array([[1], [1], [1], [1]])
    #
    # abs_dynamics = AffineDynamics(dim=2,
    #                               init_coeff_matrix_X0=init_mat_X0,
    #                               init_col_vec_X0=init_col_X0,
    #                               dynamics_matrix_A=matrix_A,
    #                               dynamics_matrix_B=coeff_matrix_B,
    #                               dynamics_coeff_matrix_U=poly_w_0,
    #                               dynamics_col_vec_U=poly_w_1)
    #
    # print(compute_beta_no_offset(abs_dynamics, 0.1))
