import numpy as np

import utils.ppl_helper as PPLHelper
from convex_set.polyhedron import Polyhedron
from convex_set.transpoly import TransPoly
from utils import affine_utils
from utils import suppfunc_utils


class PostOperator:
    @staticmethod
    def compute_initial_sf(delta_tp, poly_init, trans_poly_U, l, alpha, tau, lp):
        sf_X0 = poly_init.compute_support_function(l, lp)
        sf_tp_X0 = poly_init.compute_support_function(np.dot(delta_tp, l), lp)

        sf_V = trans_poly_U.compute_support_function(l, lp)
        sf_ball = affine_utils.support_unitball_infnorm(l)

        sf_omega0 = max(sf_X0, sf_tp_X0 + tau * sf_V + alpha * sf_ball)

        return sf_omega0

    @staticmethod
    def compute_sf_w(l, trans_poly_U, beta, tau, lp):
        sf_V = trans_poly_U.compute_support_function(l, lp)
        sf_ball = affine_utils.support_unitball_infnorm(l)
        sf_omega = tau * sf_V + beta * sf_ball
        return sf_omega

    def compute_post(self, sys_dynamics, directions, time_horizon, tau, lp):
        ret = []
        init = sys_dynamics.get_dyn_init_X0()
        poly_init = Polyhedron(init[0], init[1])

        dyn_matrix_B = sys_dynamics.get_dyn_matrix_B()
        dyn_coeff_matrix_U = sys_dynamics.get_dyn_coeff_matrix_U()
        dyn_col_vec_U = sys_dynamics.get_dyn_col_vec_U()
        dyn_matrix_A = sys_dynamics.get_dyn_coeff_matrix_A()

        trans_poly_U = TransPoly(trans_matrix_B=dyn_matrix_B,
                                 coeff_matrix_U=dyn_coeff_matrix_U,
                                 col_vec_U=dyn_col_vec_U)

        delta_tp = np.transpose(suppfunc_utils.mat_exp(dyn_matrix_A, tau))
        alpha = affine_utils.compute_alpha(sys_dynamics, tau, lp)
        beta = affine_utils.compute_beta(sys_dynamics, tau, lp)

        sf_0 = [self.compute_initial_sf(delta_tp, poly_init, trans_poly_U, l, alpha, tau, lp) for l in directions]

        time_frames = range(int(np.floor(time_horizon / tau)))

        for idx in range(len(directions)):
            for tf in time_frames:
                # delta_tp = np.transpose(mat_exp(A, n * time_interval))
                if tf == 0:
                    prev_r = directions[idx]
                    prev_s = 0

                    if idx == 0:
                        ret.append([sf_0[idx]])
                    else:
                        ret[tf].append(sf_0[idx])
                else:
                    r = np.dot(delta_tp, prev_r)
                    s = prev_s + self.compute_sf_w(prev_r, trans_poly_U, beta, tau, lp)
                    sf = self.compute_initial_sf(delta_tp, poly_init, trans_poly_U, r, alpha, tau, lp) + s

                    if idx == 0:
                        ret.append([sf])
                    else:
                        ret[tf].append(sf)

                    prev_r = r
                    prev_s = s
        return np.array(ret)

    @staticmethod
    def get_projections(directions, opdims, sf_mat):
        # todo might be buggy, try get_projections_new() instead
        ret = []

        d_mat = []
        d_mat_idx = []
        close_list = {}
        for i in range(len(directions)):
            if any(directions[i][list(opdims)]):
                    projection_dir = directions[i][list(opdims)]
                    projection_dir_tuple = tuple(projection_dir.tolist())

                    if projection_dir_tuple not in close_list:
                        d_mat.append(projection_dir)
                        d_mat_idx.append(i)
                        close_list[projection_dir_tuple] = True

        for sf_row in sf_mat:
            sf_row_col = np.reshape(sf_row, (len(sf_row), 1))
            sf_row_dir = sf_row_col[d_mat_idx]
            ret.append(PPLHelper.create_polytope(np.array(d_mat), sf_row_dir, len(opdims)))

        return ret


    @staticmethod
    def get_projections_new(directions, opdims, sf_mat):
        assert len(opdims) == 2, 'Support projection on 2d space only.'

        donot_opdims = []
        for i in range(directions.shape[1]):
            if i not in opdims:
                donot_opdims.append(i)
        donot_opdims = tuple(donot_opdims)

        ret = []

        d_mat = []
        d_mat_idx = []
        close_list = {}
        for i, d in enumerate(directions):
            if any(d[list(opdims)]) and not any(d[list(donot_opdims)]):
                projection_dir = d[list(opdims)]
                projection_dir_tuple = tuple(projection_dir.tolist())

                if projection_dir_tuple not in close_list:
                    d_mat.append(projection_dir)
                    d_mat_idx.append(i)
                    close_list[projection_dir_tuple] = True

        for sf_row in sf_mat:
            sf_row_col = np.reshape(sf_row, (len(sf_row), 1))
            sf_row_dir = sf_row_col[d_mat_idx]
            ret.append(PPLHelper.create_polytope(np.array(d_mat), sf_row_dir, len(opdims)))

        return ret

    @staticmethod
    def get_general_projections(directions, sf_mat, opdims, sys_dims, lp):
        """Pi is of shape (len(projection_dimensions), sys_dims)
            E.g. project on x1, x2 with a 3 dimensional system,
            \Pi is
            [[1, 0, 0],
             [0, 1, 0]
            ]

        \rho_Y(d) = \rho_X(\Pi^T \cdot d)

        This mapping is a general approach to project X on any direction d.
        However, this methods calls extra LPs (compared with get_projections()).

        For projection on the dimensions of the system (not a random direction,
        user get_projections() instead.
        """

        ret = []

        transform_matrix_pi = np.zeros((len(opdims), sys_dims))
        transform_matrix_pi[0][opdims[0]] = 1
        transform_matrix_pi[1][opdims[1]] = 1

        for sf_row in sf_mat:
            poly_x = Polyhedron(np.array(directions), np.reshape(sf_row, (len(sf_row), 1)))
            sf_vec = [poly_x.compute_support_function(transform_matrix_pi.T.dot(l), lp) for l in directions]
            sf_vec = np.reshape(sf_vec, (len(sf_vec), 1))

            ret.append(PPLHelper.create_polytope(np.array(directions), sf_vec, len(opdims)))

        return ret