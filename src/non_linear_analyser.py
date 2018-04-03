import itertools
from ppl import Variable, Constraint_System, C_Polyhedron
import numpy as np
import pyibex
import cvxopt as cvx
import math

import SuppFuncUtils
from ConvexSet.HyperBox import HyperBox
from LookupTables import instance_1
from Plotter import Plotter
from PostOperator import PostOperator
from SysDynamics import SysDynamics

dim = 2

def generate_invariants_from_table(table):
    axes = [sorted(list(set(key[i] for key in table))) for i in range(dim)]
    intervals = [[(ax[i], ax[i + 1]) for i in range(len(ax) - 1)] for ax in axes]

    # Cartesian product
    return list(itertools.product(*intervals))


def normalised(n):
    return n * normalised_factor


def interpolate_dynamics(table, inv, idx):
    mu = (idx + 1) * 0.5
    # return 'y', str(mu) + '*(1-x^2)*y-x'
    return 'y', '(1-x^2)*y-x'


def generate_modes(invariants, table):
    polyhedron_map = {}

    inv_id = 0
    for inv in invariants:
        cs = Constraint_System()
        variables = [Variable(idx) for idx in range(dim)]

        for idx in range(len(inv)):
            cs.insert(normalised(variables[idx]) >= normalised(inv[idx][0]))
            cs.insert(normalised(variables[idx]) <= normalised(inv[idx][1]))
        inv_poly = C_Polyhedron(cs)
        polyhedron_map[inv_id] = [interpolate_dynamics(table, inv, inv_id), inv_poly, inv]
        inv_id += 1

    return polyhedron_map


def find_relevant_modes(modes, image):
    return [(idx, modes[idx]) for idx in modes if not modes[idx][1].is_disjoint_from(image)]


def generate_abstraction_dynamics(relevant_modes, table):
    relevant_vertices = []
    for rel_md in relevant_modes:
        # print(rel_md[1][-1])
        relevant_vertices.extend(list(itertools.product(*rel_md[1][-1])))

    for i in range(len(relevant_vertices)):
        for j in range(i + 1, len(relevant_vertices)):
            try:
                x = np.array([relevant_vertices[i], relevant_vertices[j]])
                b_x = np.array([table[x[0][0], x[0][1]][0], table[x[1][0], x[1][1]][0]])
                a_x = np.linalg.solve(x, b_x)
            except np.linalg.LinAlgError:
                continue

    for i in range(len(relevant_vertices)):
        for j in range(i + 1, len(relevant_vertices)):
            try:
                x = np.array([relevant_vertices[i], relevant_vertices[j]])
                b_y = np.array([table[x[0][0], x[0][1]][1], table[x[1][0], x[1][1]][1]])
                a_y = np.linalg.solve(x, b_y)
            except np.linalg.LinAlgError:
                continue

    abstract_dynamics_x = str(a_x[0]) + '*x + ' + str(a_x[1]) + '*y'
    abstract_dynamics_y = str(a_y[0]) + '*x + ' + str(a_y[1]) + '*y'

    ux_max_abs = -1
    uy_max_abs = -1

    for rel_md in relevant_modes:
        rel_x_dynamics, rel_y_dynamics = rel_md[1][0]
        inv_x, inv_y = rel_md[1][-1]

        diff_x_dynamics = str(rel_x_dynamics) + '-(' + abstract_dynamics_x + ')'
        diff_y_dynamics = str(rel_y_dynamics) + '-(' + abstract_dynamics_y + ')'

        f_diff_x = pyibex.Function("x", "y", diff_x_dynamics)
        f_diff_y = pyibex.Function("x", "y", diff_y_dynamics)

        xy = pyibex.IntervalVector([list(inv_x), list(inv_y)])
        f_diff_x_eval = f_diff_x.eval(xy)
        f_diff_y_eval = f_diff_y.eval(xy)

        ux_max_abs = max(ux_max_abs, max(abs(f_diff_x_eval[0]), abs(f_diff_x_eval[1])))
        uy_max_abs = max(uy_max_abs, max(abs(f_diff_y_eval[0]), abs(f_diff_y_eval[1])))

    return a_x, a_y, ux_max_abs, uy_max_abs


def packup_dynamics(hyperbox, a_x, a_y, ux_max_abs, uy_max_abs):
    coeff_matix_A = np.array([a_x, a_y])

    coeff_matrix_U = np.array([[-1, 0],
                               [1, 0],
                               [0, -1],
                               [0, 1]])
    col_vec_U = np.array([ux_max_abs, ux_max_abs, uy_max_abs, uy_max_abs])
    matrix_B = np.identity(dim)

    init_coeff_matrix_X0, init_col_vec_X0 = hyperbox.to_constraints()

    return SysDynamics(dim=dim,
                       init_coeff_matrix_X0=init_coeff_matrix_X0,
                       init_col_vec_X0=init_col_vec_X0,
                       dynamics_matrix_A=coeff_matix_A,
                       dynamics_matrix_B=matrix_B,
                       dynamics_coeff_matrix_U=coeff_matrix_U,
                       dynamics_col_vec_U=col_vec_U)


def main():
    # read data 1) initial states; 2) look-up table dynamics
    init_hyperbox = HyperBox(instance_1.get_init())
    table = instance_1.get_table()

    # generate invariants from table
    invariants = generate_invariants_from_table(table)
    modes = generate_modes(invariants, table)

    # bloat the initial, epsilon is the bloating factor
    epsilon = 0.05
    init_hyperbox.bloat(epsilon)

    directions = SuppFuncUtils.generate_directions(direction_type=0,
                                                   dim=dim)

    # find initial relevant modes
    relevant_modes = find_relevant_modes(modes, init_hyperbox.to_ppl())

    # generate initial abstraction dynamics
    abstraction_dynamics = generate_abstraction_dynamics(relevant_modes, table)
    sys_dynamics = packup_dynamics(init_hyperbox, *abstraction_dynamics)
    opvars = (0, 1)

    cvx.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
    post_opt = PostOperator()

    sf_mat = post_opt.compute_post(sys_dynamics, directions, 0.01, 0.01)
    images = post_opt.get_projections(directions=directions, opdims=opvars, sf_mat=sf_mat)

    plotter = Plotter(images, opvars)
    plotter.save_polygons_to_file()

    # print(relevant_dynamics)


if __name__ == '__main__':
    normalised_factor = 1e9

    main()
