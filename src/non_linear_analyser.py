import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from ConvexSet.HyperBox import hyperbox_contain
from ConvexSet.Polyhedron import Polyhedron
from Hybridisation.Hybridiser import Hybridiser
from glpkWrapper import glpkWrapper

import numpy as np
import cvxopt as cvx


def generate_bounding_box(poly):
    bounding_box = HyperBox(poly.vertices)
    return bounding_box


def compute_support_functions_for_polyhedra(poly, directions, lp):
    vec = np.array([poly.compute_support_function(l, lp) for l in directions])
    return vec.reshape(len(vec), 1)


def main():
    # ============== setting up ============== #
    tau = 0.01
    time_horizon = 4
    direction_type = 0
    dim = 2
    lp = glpkWrapper(dim)
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    # f: Vanderpol oscillator
    non_linear_dynamics = ['x[1]', '(1-x[0]^2)*x[1]-x[0]']
    is_linear = [True, False]
    init_set = HyperBox(np.array([[1.765, 1.935]]*4))

    init_matrix_X0, init_col_vec_X0 = init_set.to_constraints()
    init_poly = Polyhedron(init_matrix_X0, init_col_vec_X0)
    time_frames = int(np.floor(time_horizon / tau))
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    hybridiser = Hybridiser(dim, non_linear_dynamics, tau, directions,
                            init_matrix_X0, init_col_vec_X0, is_linear)
    hybridiser.X = compute_support_functions_for_polyhedra(init_poly, directions, lp)
    hybridiser.init_X = hybridiser.X
    # B := \bb(X0)
    bbox = generate_bounding_box(init_poly)
    # (A, V) := L(f, B), s.t. f(x) = (A, V) over-approx. g(x)
    hybridiser.hybridise(bbox, 1e-9, lp)
    # P_{0} := \alpha(X_{0})
    hybridiser.compute_alpha_step(lp)
    hybridiser.P = hybridiser.P_temp
    i = 0

    # initialise support function matrix, [r], [s]
    sf_mat = []

    bbox_mat = []

    s_on_each_direction = [0] * len(directions)
    r_on_each_direction = directions

    flag = False  # whether we have a new abstraction domain
    isalpha = False
    while i < time_frames:
        if flag:
            # P_{i+1} := \alpha(X_{i})
            hybridiser.compute_alpha_step(lp)
            s_temp = [0] * len(directions)
            r_temp = directions
            isalpha = True
        else:
            # P_{i+1} := \beta(P_{i})
            s_temp, r_temp = hybridiser.compute_beta_step(s_on_each_direction, r_on_each_direction, lp)

        # if P_{i+1} \subset B
        if hyperbox_contain(hybridiser.abs_domain.to_constraints()[1], hybridiser.P_temp):
            hybridiser.P = hybridiser.P_temp
            sf_mat.append(hybridiser.P)

            bbox_mat.append(bbox.to_constraints()[1])

            if isalpha:
                hybridiser.init_X = hybridiser.X
                isalpha = False
            hybridiser.compute_gamma_step(lp)
            s_on_each_direction, r_on_each_direction = s_temp, r_temp
            i += 1
            if i % 100 == 0:
                print(i)
            flag = False
        else:
            bbox = hybridiser.refine_domain()
            hybridiser.hybridise(bbox, 0, lp)
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
