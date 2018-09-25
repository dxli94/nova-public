import numpy as np
from ConvexSet.polyhedron import Polyhedron
from ConvexSet.transpoly import TransPoly

from cores.sys_dynamics import AffineDynamics
from misc.affine_post_opt import PostOperator
from utils import suppfunc_utils

dynamics_matrix_A = np.array([[0, 1], [0, 0]])
dynamics_matrix_B = np.array([[1, 0], [0, 1]])

dynamics_coeff_matrix_U = np.array([[-1, 0],  # u1 >= 0
                                    [1, 0],  # u1 <= 0
                                    [0, -1],  # u2 >= 9.81
                                    [0, 1]])  # u2 <= 9.81
dynamics_col_vec_U = np.array([[0], [0], [9.81], [-9.81]])

dynamics_init_coeff_matrix_X0 = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
dynamics_init_col_vec_X0 = np.array([[-10], [10.2], [0], [0]])
sys_dynamics = AffineDynamics(x0_matrix=dynamics_init_coeff_matrix_X0,
                              x0_col=dynamics_init_col_vec_X0,
                              a_matrix=dynamics_matrix_A,
                              b_matrix=dynamics_matrix_B,
                              u_coeff=dynamics_coeff_matrix_U,
                              u_col=dynamics_col_vec_U,
                              dim=2)
directions = suppfunc_utils.generate_directions(direction_type=0, dim=2)
post_opt = PostOperator(sys_dynamics, directions)


def test_compute_initial_sf():
    init = sys_dynamics.get_dyn_init_X0()
    poly_init = Polyhedron(init[0], init[1])
    tau = 0.1

    delta_tp = np.transpose(
        suppfunc_utils.mat_exp(sys_dynamics.get_dyn_coeff_matrix_A(), 1 * tau))

    trans_poly_U = TransPoly(trans_matrix_B=sys_dynamics.get_dyn_matrix_B(),
                             coeff_matrix_U=sys_dynamics.get_dyn_coeff_matrix_U(),
                             col_vec_U=sys_dynamics.get_dyn_col_vec_U())

    true_sf_X0 = [10.2, -10, 0.0, 0]
    true_sf_tp_X0 = [10.2, -10, 0.0, 0]
    true_sf_V = [0.0, 0, -9.81, 9.81]

    sf_omega0 = [10.3034700707,
                 -9.89652992931,
                 0,
                 1.08447007069]

    # max(sf_X0, sf_tp_X0 + self.tau * sf_V + alpha * sf_ball)
    alpha = suppfunc_utils.compute_alpha(sys_dynamics, tau)
    np.testing.assert_almost_equal(alpha, 0.1034700706937106)

    for elem in zip(post_opt.directions, true_sf_X0, true_sf_tp_X0, true_sf_V, sf_omega0):
        l = elem[0]
        sf_X0 = poly_init.compute_support_function(l)
        sf_tp_X0 = poly_init.compute_support_function(np.dot(delta_tp, l))
        sf_V = trans_poly_U.compute_support_function(l)
        sf_ball = suppfunc_utils.support_unitball_infnorm(l)
        sf_omega0 = max(sf_X0, sf_tp_X0 + tau * sf_V + alpha * sf_ball)

        np.testing.assert_almost_equal(sf_X0, elem[1])
        np.testing.assert_almost_equal(sf_tp_X0, elem[2])
        np.testing.assert_almost_equal(sf_V, elem[3])
        np.testing.assert_almost_equal(sf_omega0, elem[4])
