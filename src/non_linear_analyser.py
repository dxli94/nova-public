import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from ConvexSet.HyperBox import hyperbox_contain
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.Hybridiser import Hybridiser
from glpkWrapper import glpkWrapper

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
    time_horizon = 10
    direction_type = 0
    dim = 2
    glpk_wrapper = glpkWrapper(dim)
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    start_epsilon = 1e-9
    # f: Vanderpol oscillator
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    is_linear = [True, False]

    init_set = HyperBox(np.array([[1.765, 1.75]]*4))
    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    time_frames = int(np.floor(time_horizon / tau))
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    hybridiser = Hybridiser(dim, non_linear_dynamics, tau, directions,
                            init_matrix_X0, init_col_vec_X0, is_linear)
    hybridiser.X = compute_support_functions_for_polyhedra(init_poly, directions, glpk_wrapper)
    hybridiser.init_X = hybridiser.X
    # B := \bb(X0)
    bbox = HyperBox(init_poly.vertices)
    # print(init_set.to_constraints()[1])
    # exit()
    # (A, V) := L(f, B), s.t. f(x) = (A, V) over-approx. g(x)
    hybridiser.hybridise(bbox, 1e-6, glpk_wrapper)
    # P_{0} := \alpha(X_{0})
    # hybridiser.compute_alpha_step(glpk_wrapper)
    hybridiser.P_temp = hybridiser.X
    hybridiser.P = hybridiser.X
    i = 0

    # initialise support function matrix, [r], [s]
    sf_mat = []
    bbox_mat = []

    s_on_each_direction = [0] * len(directions)
    r_on_each_direction = directions

    flag = True  # whether we have a new abstraction domain
    isalpha = False
    epsilon = start_epsilon
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
        if hyperbox_contain(hybridiser.abs_domain.to_constraints()[1], hybridiser.P_temp):
            hybridiser.P = hybridiser.P_temp
            # print(hybridiser.P)
            sf_mat.append(hybridiser.P)

            bbox_mat.append(bbox.to_constraints()[1])
            if isalpha:
                hybridiser.init_X = hybridiser.X
                isalpha = False
            hybridiser.compute_gamma_step(glpk_wrapper)
            s_on_each_direction, r_on_each_direction = s_temp, r_temp
            i += 1
            if i % 100 == 0:
                print(i)
            flag = False
            # print(epsilon)
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


if __name__ == '__main__':
    main()
