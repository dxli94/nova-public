import PPLHelper
import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.Hybridiser import Hybridiser

import numpy as np
import cvxopt as cvx


def generate_bounding_box(poly):
    bounding_box = HyperBox(poly.vertices)
    return bounding_box


def compute_support_functions_for_polyhedra(poly, directions):
    vec = np.array([poly.compute_support_function(l) for l in directions])
    return vec.reshape(len(vec), 1)


def main():
    # ============== setting up ============== #
    cvx.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

    tau = 0.01
    time_horizon = 8
    direction_type = 0
    starting_epsilon = 1e-4
    dim = 2
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    # f: Vanderpol oscillator
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    is_linear = [True, False]
    # init_set = HyperBox(np.array([[1.75705, -0.34387], [1.75705, -0.54896], [2.06926, -0.54896], [2.06926, -0.34387]]))
    init_set = HyperBox(np.array([[1, -2], [1, -2.0001], [1.0001, -2], [1.0001, -2.0001]]))
    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    time_frames = int(np.floor(time_horizon / tau))
    epsilon = starting_epsilon
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    hybridiser = Hybridiser(dim, non_linear_dynamics, tau, directions,
                            init_matrix_X0, init_col_vec_X0, is_linear)
    hybridiser.X = compute_support_functions_for_polyhedra(init_poly, directions)
    hybridiser.init_X = hybridiser.X
    # B := \bb(X0)
    bbox = generate_bounding_box(init_poly)
    # (A, V) := L(f, B), s.t. f(x) = (A, V) over-approx. g(x)
    hybridiser.hybridise(bbox, starting_epsilon)
    # P_{0} := \alpha(X_{0})
    hybridiser.compute_alpha_step()
    i = 0

    # initialise support function matrix, [r], [s]
    sf_mat = [hybridiser.P]
    s_on_each_direction = [0] * len(directions)
    r_on_each_direction = directions

    flag = False  # whether we have a new abstraction domain
    isalpha = False
    while i < time_frames:
        if flag:
            # P_{i+1} := \alpha(X_{i})
            hybridiser.compute_alpha_step()
            s_temp = [0] * len(directions)
            r_temp = directions

            isalpha = True
        else:
            # P_{i+1} := \beta(P_{i})
            s_temp, r_temp = hybridiser.compute_beta_step(s_on_each_direction, r_on_each_direction)

        # if P_{i+1} \subset B
        ppl_poly_next = PPLHelper.create_ppl_polyhedra_from_support_functions(hybridiser.P, hybridiser.directions, dim)
        if PPLHelper.contains(hybridiser.abs_domain.to_ppl(), ppl_poly_next):
            sf_mat.append(hybridiser.P)
            if isalpha:
                hybridiser.init_X = hybridiser.X
                isalpha = False
            hybridiser.compute_gamma_step()
            s_on_each_direction, r_on_each_direction = s_temp, r_temp
            i += 1
            if i % 100 == 0:
                print(i)

            epsilon = starting_epsilon
            flag = False
        else:
            bbox = generate_bounding_box(Polyhedron(hybridiser.directions, hybridiser.P.reshape(len(hybridiser.P), 1)))
            hybridiser.hybridise(bbox, epsilon)
            epsilon *= 2
            flag = True

    opvars = (0, 1)
    images = hybridiser.post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)

    from Plotter import Plotter
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()


if __name__ == '__main__':
    main()
