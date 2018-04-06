import PPLHelper
import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from ConvexSet.Polyhedron import Polyhedron
from Hybridization.Hybridizer import Hybridizer

import numpy as np
import cvxopt as cvx


def generate_bounding_box(poly):
    bounding_box = HyperBox(poly.vertices)
    return bounding_box


def main():
    # ============== setting up ============== #
    cvx.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

    tau = 0.01
    time_horizon = 1
    direction_type = 0
    starting_epsilon = 1e-1
    dim = 2
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    # f: Vanderpol dynamics
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    init_set = HyperBox(np.array([[2, 2], [2, 2], [2, 2], [2, 2]]))
    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    i = 0

    # B := \beta(P)
    bbox = generate_bounding_box(init_poly)

    # (A, V) = L(f, B),
    # P0 = R_[0,r](P)
    hybridiser = Hybridizer(dim, non_linear_dynamics, starting_epsilon, tau, directions)
    hybridiser.set_current_image(init_matrix_X0, init_col_vec_X0)
    hybridiser.hybridise(bbox)

    # initialise support function matrix
    sf_mat = [hybridiser.current_image_col_vec]

    hybridiser.compute_next_image()

    time_frames = int(np.floor(time_horizon / tau))

    while i < time_frames:
        # if P_{i+1} \subset B
        ppl_poly_next = PPLHelper.create_ppl_polyhedra_from_support_functions(hybridiser.current_image_col_vec,
                                                                              hybridiser.current_image_coff_mat,
                                                                              dim)
        if PPLHelper.contains(hybridiser.abs_domain.to_ppl(), ppl_poly_next):
            sf_mat.append(hybridiser.current_image_col_vec)
            i += 1

            hybridiser.compute_next_image()
            sf_mat.append(hybridiser.current_image_col_vec)
        else:
            # print(i)
            bbox = generate_bounding_box(Polyhedron(hybridiser.current_image_coff_mat,
                                                    hybridiser.current_image_col_vec))
            hybridiser.reset_abs_domain(bbox, starting_epsilon)

    sf_mat = np.array(sf_mat)
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
