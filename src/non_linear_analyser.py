import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from ConvexSet.HyperBox import hyperbox_contain
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.Hybridiser import Hybridiser
from GlpkWrapper import GlpkWrapper

import numpy as np


def generate_bounding_box(poly):
    bounding_box = HyperBox(poly.vertices)
    return bounding_box


def compute_support_functions_for_polyhedra(poly, directions, lp):
    vec = np.array([poly.compute_support_function(l, lp) for l in directions])
    return vec.reshape(len(vec), 1)


def main():
    # ============== setting up ============== #
    tau = 0.01
    time_horizon = 7
    time_frames = int(np.floor(time_horizon / tau))
    direction_type = 0
    dim = 2
    glpk_wrapper = GlpkWrapper(dim)
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    start_epsilon = 1e-9
    # f: Vanderpol oscillator
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    # non_linear_dynamics = ['x[1]', '-x[0]-x[1]']
    is_linear = [True, False]

    # ============== initial state set ==========#
    # init_set = HyperBox(np.array([[1.25, -2.3]]*4))
    # large Init
    init_set = HyperBox(np.array([[1.25, 2.28], [1.55, 2.28], [1.25, 2.32], [1.55, 2.32]]))
    # larger Init
    # init_set = HyperBox(np.array([[0, 0.7], [0, 1.7], [1, 1.7], [1, 1.7]]))
    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    hybridiser = Hybridiser(dim, non_linear_dynamics, tau, directions,
                            init_matrix_X0, init_col_vec_X0, is_linear)
    hybridiser.X = compute_support_functions_for_polyhedra(init_poly, directions, glpk_wrapper)
    hybridiser.init_X_in_each_domain = hybridiser.X
    hybridiser.init_X = hybridiser.X
    hybridiser.poly_init = Polyhedron(hybridiser.directions, hybridiser.init_X)
    # B := \bb(X0)
    bbox = HyperBox(init_poly.vertices)
    # (A, V) := L(f, B), s.t. f(x) = (A, V) over-approx. g(x)
    hybridiser.hybridise(bbox, 1e-6, glpk_wrapper)
    # P_{0} := \alpha(X_{0})
    hybridiser.P_temp = hybridiser.X
    hybridiser.P = hybridiser.X
    i = 0

    # initialise support function matrix, [r], [s]
    sf_mat = []
    bbox_mat = []
    x_mat = [hybridiser.X]

    s_on_each_direction = [0] * len(directions)
    r_on_each_direction = directions

    x_s_on_each_direction = [0] * len(directions)

    flag = True  # whether we have a new abstraction domain
    isalpha = False
    epsilon = start_epsilon
    delta_product = 1

    while i < time_frames:
        if flag:
            # P_{i+1} := \alpha(X_{i})
            hybridiser.compute_alpha_step(glpk_wrapper)
            s_temp = [0] * len(directions)
            r_temp = directions
            isalpha = True
        else:
            # P_{i+1} := \beta(P_{i})
            s_temp, r_temp = hybridiser.compute_beta_step(s_on_each_direction, r_on_each_direction, glpk_wrapper)

        # if P_{i+1} \subset B
        # Todo P_temp is not a hyperbox rather an rotated rectangon. Checking the bounding box is sufficient but not necessary. Needs to be refine
        if hyperbox_contain(hybridiser.abs_domain.to_constraints()[1], hybridiser.P_temp):
            hybridiser.P = hybridiser.P_temp
            prev_delta_product = delta_product
            delta_product = np.dot(delta_product, hybridiser.reach_params.delta_tp)

            sf_mat.append(hybridiser.P)
            bbox_mat.append(bbox.to_constraints()[1])
            x_mat.append(hybridiser.X)

            if isalpha:
                hybridiser.init_X_in_each_domain = hybridiser.X
                isalpha = False
            x_s_on_each_direction = hybridiser.compute_gamma_step(x_s_on_each_direction, prev_delta_product,
                                                                  delta_product, glpk_wrapper)
            s_on_each_direction, r_on_each_direction = s_temp, r_temp
            i += 1
            if i % 100 == 0:
                print(i)

            flag = False
            epsilon = start_epsilon
        else:
            bbox = hybridiser.refine_domain()
            hybridiser.hybridise(bbox, epsilon, glpk_wrapper)
            epsilon *= 2
            flag = True

    opvars = (0, 1)
    images = hybridiser.post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)

    from Plotter import Plotter
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()

    images = hybridiser.post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=bbox_mat)
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file(filename='bbox.out')

    images = hybridiser.post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=x_mat)
    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file(filename='x.out')


if __name__ == '__main__':
    main()
