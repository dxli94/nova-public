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

    tau = 0.01
    time_horizon = 0.01
    direction_type = 0
    starting_epsilon = 1e-5
    dim = 2
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    # f: Vanderpol dynamics
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    is_linear = [True, False]
    # non_linear_dynamics = ['x[1]', '-1*x[0]-x[1]']
    # is_linear = [True, True]
    # init_set = HyperBox(np.array([[2, 2], [2, 2], [2, 2], [2, 2]]))
    init_set = HyperBox(np.array([[-2, 1]*4]))
    # init_set = HyperBox(np.array([[-1, -4]*4]))
    # init_set = HyperBox(np.array([[-1, -4], [-1, -4], [-1, -4], [-1, -4]]))
    # init_set = HyperBox(np.array([[2, -0.5], [2, -0.5], [2, -0.5], [2, -0.5]]))
    # init_set = HyperBox(np.array([[-1.3, -4], [-1.3, -4], [-1.3, -4], [-1.3, -4]]))
    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    i = 0

    # B := \beta(P)
    bbox = generate_bounding_box(init_poly)

    # (A, V) = L(f, B),
    # P0 = R_[0,r](P)
    hybridiser = Hybridiser(dim, non_linear_dynamics, starting_epsilon, tau, directions,
                            init_matrix_X0, init_col_vec_X0, is_linear)
    hybridiser.set_init_image(init_matrix_X0, init_col_vec_X0)
    hybridiser.hybridise(bbox, hybridiser.epsilon)
    hybridiser.compute_initial_image()
    hybridiser.prev_sf = hybridiser.sf

    # initialise support function matrix
    sf_mat = [hybridiser.sf]

    time_frames = int(np.floor(time_horizon / tau))
    epsilon = starting_epsilon

    # isContain = True
    # Remember to comment this section in for functioning
    while i < time_frames:
        # if P_{i+1} \subset B
        hybridiser.compute_next_image()
        ppl_poly_next = PPLHelper.create_ppl_polyhedra_from_support_functions(hybridiser.sf, hybridiser.directions, dim)
        if PPLHelper.contains(hybridiser.abs_domain.to_ppl(), ppl_poly_next):
            i += 1
            print(i)
            epsilon = starting_epsilon

            sf_mat.append(hybridiser.sf)
            hybridiser.prev_sf = hybridiser.sf
        else:
            bbox = generate_bounding_box(Polyhedron(hybridiser.directions, hybridiser.prev_sf.reshape(len(hybridiser.prev_sf), 1)))
            hybridiser.hybridise(bbox, epsilon)

            epsilon *= 2

    opvars = (0, 1)
    images = hybridiser.post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)

    # for i in range(len(images)):
    #     for v in images[i].vertices:
    #         print('%.5f %.5f' % v)
    #     print('\n', end='')
    from Plotter import Plotter
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()


if __name__ == '__main__':
    main()
