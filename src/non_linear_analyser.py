import sys

import numpy as np

import SuppFuncUtils
from AffinePostOpt import PostOperator
from ConvexSet.HyperBox import HyperBox
from ConvexSet.HyperBox import hyperbox_contain
from ConvexSet.Polyhedron import Polyhedron
from DataReader import JsonReader
from GlpkWrapper import GlpkWrapper
from Hybridisation.Hybridiser import Hybridiser
from SysDynamics import GeneralDynamics


def generate_bounding_box(poly):
    bounding_box = HyperBox(poly.vertices)
    return bounding_box


def compute_support_functions_for_polyhedra(poly, directions, lp):
    vec = np.array([poly.compute_support_function(l, lp) for l in directions])
    return vec.reshape(len(vec), 1)


def main():
    # # ============== setting up ============== #
    try:
        path = sys.argv[1]
    except IndexError:
        path = '../instances/non_linear_instances/vanderpol.json'
        # path = '../instances/non_linear_instances/predator_prey.json'
        # path = '../instances/non_linear_instances/2d_water_tank.json'
        # path = '../instances/non_linear_instances/free_ball.json'

    data = JsonReader(path).read()
    time_horizon = data['time_horizon']
    tau = data['sampling_time']
    direction_type = data['direction_type']
    dim = data['dim']
    start_epsilon = data['start_epsilon']
    non_linear_dynamics = data['dynamics']
    state_vars = data['state_variables']
    is_linear = data['is_linear']
    time_frames = int(np.ceil(time_horizon / tau))
    init_coeff = np.array(data['init_coeff'])
    init_col = np.array(data['init_col'])

    glpk_wrapper = GlpkWrapper(dim)
    directions = SuppFuncUtils.generate_directions(direction_type, dim)
    id_to_vars = {}
    for i, var in enumerate(state_vars):
        id_to_vars[i] = var
    non_linear_dynamics = GeneralDynamics(id_to_vars, *non_linear_dynamics)
    init_poly = Polyhedron(init_coeff, init_col)
    # ============== setting up done ============== #

    # ============== start flowpipe construction. ============== #
    hybridiser = Hybridiser(dim, non_linear_dynamics, tau, directions,
                            init_coeff, init_col, is_linear, start_epsilon)
    hybridiser.X = compute_support_functions_for_polyhedra(init_poly, directions, glpk_wrapper)
    hybridiser.init_X = hybridiser.X
    hybridiser.init_X_in_each_domain = hybridiser.X
    hybridiser.init_poly = Polyhedron(hybridiser.directions, hybridiser.init_X)

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

    trans_poly_U_list = []
    beta_list = []

    flag = True  # whether we have a new abstraction domain
    isalpha = False
    epsilon = start_epsilon
    delta_product = 1
    delta_product_list_without_first_one = [1]

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
            if i != 0:
                temp = []
                for elem in delta_product_list_without_first_one:
                    temp.append(np.dot(elem, hybridiser.reach_params.delta_tp))
                temp.append(1)
                delta_product_list_without_first_one = temp
            delta_product = np.dot(delta_product, hybridiser.reach_params.delta_tp)

            sf_mat.append(hybridiser.P)
            bbox_mat.append(bbox.to_constraints()[1])
            x_mat.append(hybridiser.X)

            if isalpha:
                hybridiser.init_X_in_each_domain = hybridiser.X
                isalpha = False

            hybridiser.compute_gamma_step(i, trans_poly_U_list, beta_list,
                                          delta_product, delta_product_list_without_first_one,
                                          glpk_wrapper)

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
