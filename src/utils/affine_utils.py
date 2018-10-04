import numpy as np

from convex_set.polyhedron import Polyhedron
from convex_set.transpoly import TransPoly


def support_unitball_infnorm(direction):
    """
    Compute infinity norm of a unit ball on a direction.
    """
    # unit ball for f
    return sum([abs(elem) for elem in direction])


def compute_alpha(sys_dynamics, tau, lp):
    """
    Compute bloating factor \alpha. See LGG paper for reference.
    """
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
    """
    Compute bloating factor \beta. See LGG paper for reference.
    """
    dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

    # if dyn_matrix_A is a zero-matrix, no need to perform bloating
    if not np.any(dyn_matrix_A):
        return 0

    dyn_matrix_B = sys_dynamics.get_dyn_matrix_B()
    dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
    dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()

    norm_a = np.linalg.norm(dyn_matrix_A, np.inf)

    tt1 = np.exp(tau * norm_a)

    poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                       coeff_matrix_U=dyn_coeff_matrix_U,
                       col_vec_U=dyn_col_vec_U)

    v_max_norm = poly_U.compute_max_norm(lp)

    return (tt1 - 1 - tau * norm_a) * (v_max_norm / norm_a)