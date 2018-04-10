import PPLHelper
import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from ConvexSet.Polyhedron import Polyhedron
from Hybridization.Hybridiser import Hybridiser

import numpy as np
import cvxopt as cvx


def generate_bounding_box(poly):
    bounding_box = HyperBox(poly.vertices)
    return bounding_box


def main():
    # ============== setting up ============== #
    cvx.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

    tau = 0.001
    time_horizon = 0.01
    direction_type = 0
    starting_epsilon = 1e-4
    dim = 2
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    # f: Vanderpol dynamics
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    is_linear = [True, False]
    init_set = HyperBox(np.array([[-1, -2], [-1, -2.00001], [-1.00001, -2], [-1.00001, -2.000001]]))
    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    i = 0

    # B := \beta(P)
    bbox = generate_bounding_box(init_poly)

    # (A, V) = L(f, B),
    # P0 = R_[0,r](P)
    hybridiser = Hybridiser(dim, non_linear_dynamics, tau, directions,
                            init_matrix_X0, init_col_vec_X0, is_linear)
    hybridiser.hybridise(bbox, starting_epsilon)
    hybridiser.compute_initial_image()
    hybridiser.prev_sf = hybridiser.sf

    # initialise support function matrix
    sf_mat = [hybridiser.sf]

    time_frames = int(np.floor(time_horizon / tau))
    epsilon = starting_epsilon
    s_on_each_direction = [0] * len(directions)
    r_on_each_direction = directions

    isChanged = False
    # Remember to comment this section in for functioning
    while i < time_frames:
        # if P_{i+1} \subset B
        s_temp, r_temp = hybridiser.compute_next_image(s_on_each_direction, r_on_each_direction)
        ppl_poly_next = PPLHelper.create_ppl_polyhedra_from_support_functions(hybridiser.sf, hybridiser.directions, dim)
        if PPLHelper.contains(hybridiser.abs_domain.to_ppl(), ppl_poly_next):
            i += 1
            # print(i)
            if i % 100 == 0:
                print(str(i) + ' in ' + str(time_frames))

            if isChanged:
                isChanged = False
            else:
                s_on_each_direction, r_on_each_direction = s_temp, r_temp

            epsilon = starting_epsilon
            sf_mat.append(hybridiser.sf)
            hybridiser.prev_sf = hybridiser.sf
        else:
            if not isChanged:
                hybridiser.update_init_image(hybridiser.directions, hybridiser.prev_sf.reshape(len(hybridiser.prev_sf), 1))
                s_on_each_direction = [0] * len(directions)
                r_on_each_direction = directions
                isChanged = True
                bbox = generate_bounding_box(Polyhedron(hybridiser.directions, hybridiser.prev_sf.reshape(len(hybridiser.prev_sf), 1)))

            hybridiser.hybridise(bbox, epsilon)
            epsilon *= 2

    opvars = (0, 1)
    images = hybridiser.post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)

    from Plotter import Plotter
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()


if __name__ == '__main__':
    main()
