import numpy as np
from scipy.linalg import expm

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly


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

    # norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    # two_norm_a = compute_log_2norm(dyn_matrix_A)
    log_norm_a = compute_log_infnorm(dyn_matrix_A)
    norm_a = log_norm_a
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

    # norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    # two_norm_a = compute_log_2norm(dyn_matrix_A)
    log_norm_a = compute_log_infnorm(dyn_matrix_A)
    norm_a = log_norm_a
    # norm_a = two_norm_a

    # ========
    tt1 = np.exp(tau * norm_a)

    poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                       coeff_matrix_U=dyn_coeff_matrix_U,
                       col_vec_U=dyn_col_vec_U)

    v_max_norm = poly_U.compute_max_norm(lp)

    return (tt1 - 1 - tau * norm_a) * (v_max_norm / norm_a)


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


if __name__ == '__main__':
    direction_type = 1
    dim = 3

    print(len(generate_directions(direction_type, dim)))
