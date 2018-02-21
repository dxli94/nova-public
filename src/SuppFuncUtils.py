import numpy as np
from scipy.linalg import expm

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly


def mat_exp(A, tau):
    return expm(np.multiply(A, tau))


def support_unitball_infnorm(direction):
    return sum([abs(elem) for elem in direction])


def compute_alpha(sys_dynamics, tau):
    dyn_matrix_A = np.matrix(sys_dynamics.get_dyn_coeff_matrix_A())
    dyn_matrix_B = np.matrix(sys_dynamics.get_dyn_matrix_B())
    dyn_coeff_matrix_U = np.matrix(sys_dynamics.get_dyn_coeff_matrix_U())
    dyn_col_vec_U = np.matrix(sys_dynamics.get_dyn_col_vec_U())
    dyn_matrix_init = sys_dynamics.get_dyn_init_X0()

    # norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    norm_a = np.linalg.norm(dyn_matrix_A, 1)

    tt1 = np.exp(tau * norm_a)

    I_max_norm = Polyhedron(coeff_matrix=dyn_matrix_init[0],
                            col_vec=dyn_matrix_init[1]).compute_max_norm()
    poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                       coeff_matrix=dyn_coeff_matrix_U,
                       col_vec_U=dyn_col_vec_U)

    v_max_norm = poly_U.compute_max_norm()

    return (tt1 - 1 - tau * norm_a) * (I_max_norm + (v_max_norm / norm_a))


def compute_beta(sys_dynamics, tau):
    dyn_matrix_A = np.matrix(sys_dynamics.get_dyn_coeff_matrix_A())
    dyn_matrix_B = np.matrix(sys_dynamics.get_dyn_matrix_B())
    dyn_coeff_matrix_U = np.matrix(sys_dynamics.get_dyn_coeff_matrix_U())
    dyn_col_vec_U = np.matrix(sys_dynamics.get_dyn_col_vec_U())

    norm_a = np.linalg.norm(dyn_matrix_A, np.inf)
    tt1 = np.exp(tau * norm_a)

    poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                       coeff_matrix=dyn_coeff_matrix_U,
                       col_vec_U=dyn_col_vec_U)

    v_max_norm = poly_U.compute_max_norm()

    return (tt1 - 1 - tau * norm_a) * (v_max_norm / norm_a)
