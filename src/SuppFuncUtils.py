import numpy as np
from scipy.linalg import expm


def mat_exp(A, tau):
    return expm(np.multiply(A, tau))


def support_unitball_infnorm(direction):
    return sum([abs(elem) for elem in direction])


def compute_alpha(poly, tau):
    coeff_matrix = np.matrix(poly.coeff_matrix)
    col_vec = np.matrix(poly.col_vec)

    norm_A = np.linalg.norm(coeff_matrix, np.inf)
    tt1 = np.exp(tau * norm_A)

    I_max_norm = poly.compute_max_norm()
    V_max_norm = compute_v_max_norm()

    return (tt1 - 1 - tau * norm_A) * (I_max_norm + (V_max_norm / norm_A))


def compute_v_max_norm():
    return -1


def compute_beta(poly, tau):
    coeff_matrix = np.matrix(poly.coeff_matrix)
    col_vec = np.matrix(poly.col_vec)

    norm_A = np.linalg.norm(coeff_matrix, np.inf)
    tt1 = np.exp(tau * norm_A)

    V_max_norm = compute_v_max_norm()

    return (tt1 - 1 - tau * norm_A) * (V_max_norm / norm_A)
