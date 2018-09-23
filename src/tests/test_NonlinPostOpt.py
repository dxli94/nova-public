import numpy as np

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly
from SysDynamics import AffineDynamics
from utils import SuppFuncUtils
from utils.GlpkWrapper import GlpkWrapper


def compute_initial_sf(delta_tp, poly_init, trans_poly_U, l, alpha, tau, lp):
    sf_X0 = poly_init.compute_support_function(l, lp)
    sf_tp_X0 = poly_init.compute_support_function(np.dot(delta_tp, l), lp)

    sf_V = trans_poly_U.compute_support_function(l, lp)
    sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

    print(tau * sf_V + alpha * sf_ball)

    sf_omega0 = max(sf_X0, sf_tp_X0 + tau * sf_V + alpha * sf_ball)

    return sf_omega0


def compute_beta_step(delta_tp, poly_init, trans_poly_U, l, beta, tau, lp):
    delta_tp_l = np.dot(delta_tp, l)
    term1 = poly_init.compute_support_function(delta_tp_l, lp)
    term2 = tau * trans_poly_U.compute_support_function(l, lp)
    term3 = beta * SuppFuncUtils.support_unitball_infnorm(l)

    val = term1 + term2 + term3

    print(val)
    return val


def test_A():
    mat_A = np.array([[0., 1.],
             [-7.359545618728333, -0.9599999999999997]])
    mat_B = np.identity(2)
    mat_U = np.array([[1, 0],
                [-1, 0],
                [0, 1],
                [0, -1]
                ])
    col_U = np.array([[0.],
             [0.],
             [8.903363866219667],
             [-8.751075622179807]
             ])
    mat_init = np.array([[1, 0],
                [-1, 0],
                [0, 1],
                [0, -1]
                ])
    col_init = np.array([[1.55], [-1.25], [2.32], [-2.28]])
    poly_init = Polyhedron(mat_init, col_init)

    tau = 0.01
    delta_tp = np.transpose(SuppFuncUtils.mat_exp(mat_A, tau))

    trans_poly_U = TransPoly(trans_matrix_B=mat_B,
                             coeff_matrix_U=mat_U,
                             col_vec_U=col_U)
    directions = SuppFuncUtils.generate_directions(direction_type=1, dim=2)

    sys_dynamics = AffineDynamics(dim=2,
                                  init_coeff_matrix_X0=mat_init,
                                  init_col_vec_X0=col_init,
                                  dynamics_matrix_A=mat_A,
                                  dynamics_matrix_B=mat_B,
                                  dynamics_coeff_matrix_U=mat_U,
                                  dynamics_col_vec_U=col_U)

    lp = GlpkWrapper(sys_dim=2)

    # alpha = SuppFuncUtils.compute_alpha(sys_dynamics, tau, lp)
    beta = SuppFuncUtils.compute_beta(sys_dynamics, tau, lp)

    print('alpha is {}'.format(beta))

    sf_0 = [compute_beta_step(delta_tp, poly_init, trans_poly_U, l, beta, tau, lp) for l in directions]

    print(sf_0)


if __name__ == '__main__':
    test_A()

