import numpy as np
from scipy.linalg import expm


def mat_exp(A, tau=1):
    """
    Compute matrix exponential e^{A \tau}.
    """
    return expm(np.multiply(A, tau))


def compute_log_infnorm(A):
    return max(np.sum(A, axis=1))


def compute_log_2norm(A):
    eigns, _ = np.linalg.eig((A + A.T)/2)
    max_eign = max(eigns)

    return max_eign


def compute_beta_no_offset(sys_dynamics, tau):
    """
    Compute bloating factor while offsetting the input set with a constant.
    """
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
    """
    Returns the reachability parameters, including bloating factors (\alpha, \beta)
    and transposed matrix exponential and stepsize.
    """
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

    return alpha, beta, delta, tau


def generate_directions(direction_type, dim):
    """
    Generate template directions for a n-dimensional space.

    direction_type = 0: Box
    direction_type = 1: Octagon
    n = dim
    """
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

    return np.array(direction_generator, dtype=float)


def compute_support_function_singular(c, l):
    """
    Compute support function on direction l for a singular point (vector) c.
    """
    return np.dot(l, c)


def mat_exp_int(A, t_min, t_max, nbins=5):
    """
    Compute integration of matrix exponential minus identity matrix using numerical method.
    """
    iden = np.identity(A.shape[0])
    f = lambda x: expm(A*x)-iden
    xv = np.linspace(t_min, t_max, nbins)
    result = np.apply_along_axis(f, 0, xv.reshape(1, -1))
    return np.trapz(result, xv)

#
# def compute_phi_matrices(a_matrix, tau):
#     """
#     See SpaceEx CAV 11' (Goran. F. et al) Eq. (8) for Φ_{1}(|A|, τ) and Φ_{2}(|A|, τ).
#     """
#     assert len(a_matrix.shape) == 2
#     assert a_matrix.shape[0] == a_matrix.shape[1]
#
#     dim = a_matrix.shape[0]
#     a_matrix_abs = np.absolute(a_matrix)
#
#     try:
#         inv_a = np.linalg.inv(a_matrix)
#         inv_abs_a = np.linalg.inv(a_matrix_abs)
#
#         I = np.identity(dim)
#         # Φ_{1}(A, τ) = A^-1 *(e^{tau*A}-I)
#         phi1 = np.dot(inv_a, (mat_exp(a_matrix, tau)-I))
#         # Φ_{2}(|A|, τ) = |A|^-2 *(e^{tau*|A|}-I-tau*|A|)
#         term1 = np.dot(inv_abs_a, inv_abs_a)
#         phi2 = np.dot(term1, (mat_exp(a_matrix_abs, tau)-I-tau*a_matrix_abs))
#     except np.linalg.LinAlgError:  # A not invertible
#         phi_base_matrix = make_base_phi_matrix(a_matrix, tau)
#         phi_base_exp = mat_exp(phi_base_matrix)
#
#         phi1 = phi_base_exp[0:dim, 1*dim:2*dim]
#         phi2 = phi_base_exp[0:dim, 2*dim:3*dim]
#
#     return phi1, phi2


def compute_phi_1(a_matrix, tau):
    """
    See SpaceEx CAV 11' (Goran. F. et al) Eq. (8) for Φ_{1}(|A|, τ) and Φ_{2}(|A|, τ).
    """
    assert len(a_matrix.shape) == 2
    assert a_matrix.shape[0] == a_matrix.shape[1]

    dim = a_matrix.shape[0]

    try:
        inv_a = np.linalg.inv(a_matrix)

        I = np.identity(dim)
        # Φ_{1}(A, τ) = A^-1 *(e^{tau*A}-I)
        phi1 = np.dot(inv_a, (mat_exp(a_matrix, tau) - I))
    except np.linalg.LinAlgError:  # A not invertible
        phi_base_matrix = make_base_phi_matrix(a_matrix, tau)
        phi_base_exp = mat_exp(phi_base_matrix)

        phi1 = phi_base_exp[0:dim, 1 * dim:2 * dim]

    return phi1


def compute_phi_2(a_matrix, tau):
    """
    See SpaceEx CAV 11' (Goran. F. et al) Eq. (8) for Φ_{1}(|A|, τ) and Φ_{2}(|A|, τ).
    """
    assert len(a_matrix.shape) == 2
    assert a_matrix.shape[0] == a_matrix.shape[1]

    dim = a_matrix.shape[0]

    try:
        inv_a = np.linalg.inv(a_matrix)

        I = np.identity(dim)
        # Φ_{2}(|A|, τ) = |A|^-2 *(e^{tau*|A|}-I-tau*|A|)
        term1 = np.dot(inv_a, inv_a)
        phi2 = np.dot(term1, (mat_exp(a_matrix, tau)-I-tau*a_matrix))
    except np.linalg.LinAlgError:  # A not invertible
        phi_base_matrix = make_base_phi_matrix(a_matrix, tau)
        phi_base_exp = mat_exp(phi_base_matrix)

        phi2 = phi_base_exp[0:dim, 2*dim:3*dim]

    return phi2


def make_base_phi_matrix(a_matrix, tau):
    """
    base_phi_matrix = [ [A\tau, I\tau,     0],
                        [    0,     0, I\tau],
                        [    0,     0,     0]
                      ]

    See SpaceEx CAV 11' (Goran. F. et al).
    """

    dim = a_matrix.shape[0]

    A_tau = a_matrix * tau
    I_tau = np.identity(dim) * tau

    base_matrix = np.zeros(shape=(3*dim, 3*dim))
    base_matrix[0:dim, 0:dim] = A_tau
    base_matrix[0:dim, dim:2*dim] = I_tau
    base_matrix[dim:2*dim, 2*dim:3*dim] = I_tau

    return base_matrix


if __name__ == '__main__':
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tau = 0.1
    dim = A.shape[1]

    I = np.identity(dim)
    phi_base = make_base_phi_matrix(A, tau)
    phi_base_exp = mat_exp(phi_base)
    phi_1 = phi_base_exp[0:dim, 1*dim:2*dim]
    res = phi_1 - tau * I

    print(res)
    print(mat_exp_int(A, t_min=0, t_max=tau, nbins=5000))

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
