import numpy as np

import SuppFuncUtils

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from PostOperator import PostOperator
from SysDynamics import SysDynamics

dynamics_matrix_A = np.array([[0, 1], [0, 0]])
dynamics_matrix_B = np.array([[1, 0], [0, 1]])

dynamics_coeff_matrix_U = np.array([[-1, 0],  # u1 >= 0
                                    [1, 0],  # u1 <= 0
                                    [0, -1],  # u2 >= 9.81
                                    [0, 1]])  # u2 <= 9.81
dynamics_col_vec_U = np.array([[0], [0], [9.81], [-9.81]])

dynamics_init_coeff_matrix_X0 = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
dynamics_init_col_vec_X0 = np.array([[-10], [10.2], [0], [0]])
sys_dynamics = SysDynamics(init_coeff_matrix_X0=dynamics_init_coeff_matrix_X0,
                           init_col_vec_X0=dynamics_init_col_vec_X0,
                           dynamics_matrix_A=dynamics_matrix_A,
                           dynamics_matrix_B=dynamics_matrix_B,
                           dynamics_coeff_matrix_U=dynamics_coeff_matrix_U,
                           dynamics_col_vec_U=dynamics_col_vec_U)
directions = SuppFuncUtils.generate_directions(direction_type=0, dim=2)
post_opt = PostOperator(sys_dynamics, directions, time_horizon=0.1, samp_freq=0.1)


def test_compute_initial_sf():
    init = sys_dynamics.get_dyn_init_X0()
    poly_init = Polyhedron(init[0], init[1])
    delta_tp = np.transpose(
        SuppFuncUtils.mat_exp(sys_dynamics.get_dyn_coeff_matrix_A(), 1 * post_opt.tau))

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
    alpha = SuppFuncUtils.compute_alpha(sys_dynamics, post_opt.tau)
    np.testing.assert_almost_equal(alpha, 0.1034700706937106)

    for elem in zip(post_opt.directions, true_sf_X0, true_sf_tp_X0, true_sf_V, sf_omega0):
        l = elem[0]
        sf_X0 = poly_init.compute_support_function(l)
        sf_tp_X0 = poly_init.compute_support_function(np.dot(delta_tp, l))
        sf_V = trans_poly_U.compute_support_function(l)
        sf_ball = SuppFuncUtils.support_unitball_infnorm(l)
        sf_omega0 = max(sf_X0, sf_tp_X0 + post_opt.tau * sf_V + alpha * sf_ball)

        np.testing.assert_almost_equal(sf_X0, elem[1])
        np.testing.assert_almost_equal(sf_tp_X0, elem[2])
        np.testing.assert_almost_equal(sf_V, elem[3])
        np.testing.assert_almost_equal(sf_omega0, elem[4])
