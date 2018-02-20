import numpy as np
from scipy.linalg import expm
from Polyhedron import Polyhedron


def mat_exp(A, tau):
    return expm(np.multiply(A, tau))


def support_unitball_infnorm(direction):
    return sum([abs(elem) for elem in direction])


def compute_alpha(sys_dynamics, tau):
    dyn_matrix_a = np.matrix(sys_dynamics.get_dyn_matrix_a())
    dyn_matrix_b = np.matrix(sys_dynamics.get_dyn_matrix_b())
    dyn_matrix_input = np.matrix(sys_dynamics.get_dyn_input_matrix())
    dyn_matrix_init = sys_dynamics.get_dyn_init()

    norm_a = np.linalg.norm(dyn_matrix_a, np.inf)
    tt1 = np.exp(tau * norm_a)

    I_max_norm = Polyhedron(coeff_matrix=dyn_matrix_init[0],
                            vec_col=dyn_matrix_init[1]).compute_max_norm()
    matrix_v = np.matmul(dyn_matrix_b, dyn_matrix_input)
    v_max_norm = Polyhedron(coeff_matrix=matrix_v).compute_max_norm()

    return (tt1 - 1 - tau * norm_a) * (I_max_norm + (v_max_norm / norm_a))


def compute_beta(sys_dynamics, tau):
    dyn_matrix_a = np.matrix(sys_dynamics.get_dyn_matrix_a())
    dyn_matrix_b = np.matrix(sys_dynamics.get_dyn_matrix_b())
    dyn_matrix_input = np.matrix(sys_dynamics.get_dyn_input_matrix())

    norm_a = np.linalg.norm(dyn_matrix_a, np.inf)
    tt1 = np.exp(tau * norm_a)

    matrix_v = np.matmul(dyn_matrix_b, dyn_matrix_input)
    v_max_norm = Polyhedron(coeff_matrix=matrix_v).compute_max_norm()

    return (tt1 - 1 - tau * norm_a) * (v_max_norm / norm_a)
