import numpy as np

import SuppFuncUtils

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly


class PostOperator:
    def __init__(self, sys_dynamics, directions, time_horizon, samp_freq):
        self.sys_dynamics = sys_dynamics
        self.directions = directions
        self.time_horizon = time_horizon
        self.tau = samp_freq

    def compute_initial_sf(self, poly_init, trans_poly_U, l, alpha):
        dyn_matrix_A = self.sys_dynamics.get_dyn_coeff_matrix_A()
        delta_tp = np.transpose(SuppFuncUtils.mat_exp(dyn_matrix_A, 1 * self.tau))

        sf_X0 = poly_init.compute_support_function(l)
        sf_tp_X0 = poly_init.compute_support_function(np.dot(delta_tp, l))

        sf_V = trans_poly_U.compute_support_function(l)
        sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

        sf_omega0 = max(sf_X0, sf_tp_X0 + self.tau * sf_V + alpha * sf_ball)
        return sf_omega0

    def compute_sf_w(self, l, trans_poly_U, beta):
        sf_V = trans_poly_U.compute_support_function(l)
        sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

        sf_omega = self.tau * sf_V + beta * sf_ball
        return sf_omega

    def compute_post(self):
        ret = []
        init = self.sys_dynamics.get_dyn_init_X0()
        poly_init = Polyhedron(init[0], init[1])

        dyn_matrix_B = self.sys_dynamics.get_dyn_matrix_B()
        dyn_coeff_matrix_U = self.sys_dynamics.get_dyn_coeff_matrix_U()
        dyn_col_vec_U = self.sys_dynamics.get_dyn_col_vec_U()
        dyn_matrix_A = self.sys_dynamics.get_dyn_coeff_matrix_A()

        trans_poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                                 coeff_matrix_U=dyn_coeff_matrix_U,
                                 col_vec_U=dyn_col_vec_U)

        delta_tp = np.transpose(SuppFuncUtils.mat_exp(dyn_matrix_A, self.tau))
        alpha = SuppFuncUtils.compute_alpha(self.sys_dynamics, self.tau)
        beta = SuppFuncUtils.compute_beta(self.sys_dynamics, self.tau)

        sf_0 = [self.compute_initial_sf(poly_init, trans_poly_U, l, alpha) for l in self.directions]
        poly_omega0 = Polyhedron(np.array(self.directions), np.reshape(sf_0, (len(sf_0), 1)))

        time_frames = range(int(np.floor(self.time_horizon / self.tau)))

        for idx in range(len(self.directions)):
            ret.append([])
            for n in time_frames:
                # delta_tp = np.transpose(mat_exp(A, n * time_interval))
                if n == 0:
                    prev_r = self.directions[idx]
                    prev_s = 0
                    ret[-1].append(sf_0[idx])
                else:
                    r = np.dot(delta_tp, prev_r)
                    s = prev_s + self.compute_sf_w(r, trans_poly_U, beta)
                    sf = poly_omega0.compute_support_function(r) + s

                    ret[-1].append(sf)

                    prev_r = r
                    prev_s = s
        return np.array(ret)

    def get_images(self, sf_mat):
        ret = []

        d_mat = np.array(self.directions)
        sf_mat = np.transpose(sf_mat)
        for sf_row in sf_mat:
            ret.append(Polyhedron(d_mat, np.reshape(sf_row, (len(sf_row), 1))))
        return ret
