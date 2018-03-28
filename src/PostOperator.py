import numpy as np

import SuppFuncUtils

from ConvexSet.Polyhedron import Polyhedron
from ConvexSet.TransPoly import TransPoly


class PostOperator:
    def __init__(self, sys_dynamics, directions):
        self.sys_dynamics = sys_dynamics
        self.directions = directions

    def compute_initial_sf(self, poly_init, trans_poly_U, l, alpha, tau):
        dyn_matrix_A = self.sys_dynamics.get_dyn_coeff_matrix_A()
        delta_tp = np.transpose(SuppFuncUtils.mat_exp(dyn_matrix_A, 1 * tau))

        sf_X0 = poly_init.compute_support_function(l)
        sf_tp_X0 = poly_init.compute_support_function(np.dot(delta_tp, l))

        sf_V = trans_poly_U.compute_support_function(l)
        sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

        sf_omega0 = max(sf_X0, sf_tp_X0 + tau * sf_V + alpha * sf_ball)
        return sf_omega0

    def compute_sf_w(self, l, trans_poly_U, beta, tau):
        sf_V = trans_poly_U.compute_support_function(l)
        sf_ball = SuppFuncUtils.support_unitball_infnorm(l)

        sf_omega = tau * sf_V + beta * sf_ball
        return sf_omega

    def compute_post(self, time_horizon, tau):
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

        delta_tp = np.transpose(SuppFuncUtils.mat_exp(dyn_matrix_A, tau))
        alpha = SuppFuncUtils.compute_alpha(self.sys_dynamics, tau)
        beta = SuppFuncUtils.compute_beta(self.sys_dynamics, tau)

        sf_0 = [self.compute_initial_sf(poly_init, trans_poly_U, l, alpha, tau) for l in self.directions]
        poly_omega0 = Polyhedron(np.array(self.directions), np.reshape(sf_0, (len(sf_0), 1)))

        time_frames = range(int(np.floor(time_horizon / tau)))

        for idx in range(len(self.directions)):
            for tf in time_frames:
                # delta_tp = np.transpose(mat_exp(A, n * time_interval))
                if tf == 0:
                    prev_r = self.directions[idx]
                    prev_s = 0

                    if idx == 0:
                        ret.append([sf_0[idx]])
                    else:
                        ret[tf].append(sf_0[idx])
                else:
                    r = np.dot(delta_tp, prev_r)
                    s = prev_s + self.compute_sf_w(r, trans_poly_U, beta, tau)
                    sf = poly_omega0.compute_support_function(r) + s

                    if idx == 0:
                        ret.append([sf])
                    else:
                        ret[tf].append(sf)

                    prev_r = r
                    prev_s = s
        return np.array(ret)

    def get_projections(self, opdims, sf_mat):
        ret = []

        d_mat = []
        d_mat_idx = []
        close_list = {}
        for i in range(len(self.directions)):
            if any(self.directions[i][list(opdims)]): # != 0 or self.directions[i][opdims[1]] != 0:
                projection_dir = self.directions[i][list(opdims)]
                projection_dir_tuple = tuple(projection_dir.tolist())

                if projection_dir_tuple not in close_list:
                    d_mat.append(projection_dir)
                    d_mat_idx.append(i)
                    close_list[projection_dir_tuple] = True

        for sf_row in sf_mat:
            sf_row_col = np.reshape(sf_row, (len(sf_row), 1))
            sf_row_dir = sf_row_col[d_mat_idx]
            ret.append(Polyhedron(np.array(d_mat), sf_row_dir))
        return ret

    def get_general_projections(self, directions, sf_mat, opdims, sys_dims):
        ret = []

        transform_matrix_pi = np.zeros((len(opdims), sys_dims))
        transform_matrix_pi[0][opdims[0]] = 1
        transform_matrix_pi[1][opdims[1]] = 1

        for sf_row in sf_mat:
            # print(len(sf_row))
            # print(len(self.directions))
            # exit()
            poly_x = Polyhedron(np.array(self.directions), np.reshape(sf_row, (len(sf_row), 1)))
            sf_vec = [poly_x.compute_support_function(transform_matrix_pi.T.dot(l)) for l in directions]
            sf_vec = np.reshape(sf_vec, (len(sf_vec), 1))

            ret.append(Polyhedron(np.array(directions), sf_vec))

        return ret
