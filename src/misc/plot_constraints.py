'''
Stanley Bak
March 2018

Polytope convex hull testing code
'''

import math
import matplotlib.pyplot as plt

import numpy as np

import cvxopt

plot_vecs = None  # list of vectors to optimize in for plotting, assigned in Star.init_plot_vecs


def init_plot_vecs(num_angles=2048):
    'initialize plot_vecs'

    global plot_vecs

    plot_vecs = []

    step = 2.0 * math.pi / num_angles

    for theta in np.arange(0.0, 2.0 * math.pi, step):
        x = math.cos(theta)
        y = math.sin(theta)

        vec = np.array([x, y], dtype=float)

        plot_vecs.append(vec)


def get_verts(a_mat, b_vec):
    'get the verticies of the polygon projection of the star used for plotting'

    pts = find_boundary_pts(a_mat, b_vec)

    verts = [[pt[0], pt[1]] for pt in pts]

    # wrap polygon back to first point
    verts.append(verts[0])

    return verts


def minimize(a_mat, b_vec, direction):
    'minimize a_mat and b_vec... returning the 2-d point of the first 2 coordinates of the minimum'

    a_ub = [[float(x) for x in row] for row in a_mat]
    b_ub = [float(x) for x in b_vec]

    num_vars = len(a_ub[0])

    c = [float(direction[i]) if i < len(direction) else 0.0 for i in range(num_vars)]

    # solve it with cvxopt
    options = {'show_progress': False}
    sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(a_ub).T, cvxopt.matrix(b_ub), options=options)

    if sol['status'] != 'optimal':
        raise RuntimeError("cvxopt LP failed: {}".format(sol['status']))

    res_cvxopt = [float(n) for n in sol['x']]

    return np.array([res_cvxopt[0], res_cvxopt[1]])


def find_boundary_pts(a_mat, b_vec):
    '''
    find a constaint-star's boundaries using Star.plot_vecs. This solves several LPs and
    returns a list of points on the boundary (in the standard basis) which maximize each
    of the passed-in directions
    '''

    global plot_vecs

    direction_list = plot_vecs
    rv = []

    assert len(direction_list) >= 2

    # optimized approach: do binary search to find changes
    point = minimize(a_mat, b_vec, direction_list[0])
    rv.append(point.copy())

    # add it in thirds, to ensure we don't miss anything
    third = len(direction_list) // 3

    # 0 to 1/3
    point = minimize(a_mat, b_vec, direction_list[third])

    if not np.array_equal(point, rv[-1]):
        rv += binary_search_boundaries(a_mat, b_vec, 0, third, rv[-1], point)
        rv.append(point.copy())

    # 1/3 to 2/3
    point = minimize(a_mat, b_vec, direction_list[2 * third])

    if not np.array_equal(point, rv[-1]):
        rv += binary_search_boundaries(a_mat, b_vec, third, 2 * third, rv[-1], point)
        rv.append(point.copy())

    # 2/3 to end
    point = minimize(a_mat, b_vec, direction_list[-1])

    if not np.array_equal(point, rv[-1]):
        rv += binary_search_boundaries(a_mat, b_vec, 2 * third, len(direction_list) - 1, rv[-1], point)
        rv.append(point.copy())

    # pop last point if it's the same as the first point
    if len(rv) > 1 and np.array_equal(rv[0], rv[-1]):
        rv.pop()

    return rv


def binary_search_boundaries(a_mat, b_vec, start, end, start_point, end_point):
    '''
    return all the optimized points in the star for the passed-in directions, between
    the start and end indices, exclusive

    points which match start_point or end_point are not returned
    '''

    global plot_vecs

    dirs = plot_vecs
    rv = []

    if start + 1 < end:
        mid = (start + end) // 2

        mid_point = minimize(a_mat, b_vec, dirs[mid])

        not_start = not np.allclose(start_point, mid_point, atol=1e-3)
        not_end = not np.allclose(end_point, mid_point, atol=1e-3)

        if not_start:
            rv += binary_search_boundaries(a_mat, b_vec, start, mid, start_point, mid_point)

        if not_start and not_end:
            rv.append(mid_point)

        if not_end:
            rv += binary_search_boundaries(a_mat, b_vec, mid, end, mid_point, end_point)

    return rv


def plot_constraints(a_mat, b_vec, col='k-', lw=1):
    '''
    plot a polytope given in constraint form (by taking a fine-grained support function outer-approximation)

    This calls plt.plot(), so call plt.show() afterwards
    '''

    verts = get_verts(a_mat, b_vec)

    # wrap polygon back to first point
    verts.append(verts[0])

    xs = [ele[0] for ele in verts]
    ys = [ele[1] for ele in verts]

    plt.plot(xs, ys, col, lw=lw)


def main():
    'main function'

    a1_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    a1_vec = np.array([5, -4.5, 2, -1], dtype=float)

    plot_constraints(a1_mat, a1_vec)

    a2_mat = np.array([[1, 1], [-1, 1], [0, -1]], dtype=float)
    a2_vec = np.array([0.5, 0.5, 0], dtype=float)

    plot_constraints(a2_mat, a2_vec, col='r')

    o_mat = np.zeros(a2_mat.shape)
    a1_colvec = a1_vec.copy()
    a1_colvec.shape = (a1_vec.shape[0], 1)

    a2_colvec = a2_vec.copy()
    a2_colvec.shape = (a2_vec.shape[0], 1)
    o_vec = np.zeros(a2_vec.shape)

    row1 = np.concatenate((a1_mat, a1_mat, a1_colvec), axis=1)
    row2 = np.concatenate((o_mat, -a2_mat, -a2_colvec), axis=1)

    chull_mat = np.concatenate((row1, row2))
    chull_vec = np.concatenate((a1_vec, o_vec))

    print_on = False

    if print_on:
        print("a1_mat:\n{}".format(a1_mat))
        print("a1_vec: {}".format(a1_vec))

        print("a2_mat:\n{}".format(a2_mat))
        print("a2_vec: {}".format(a2_vec))

        print("----------")
        print("row1:\n{}".format(row1))
        print("row2:\n{}".format(row2))

        print("----------")
        print("chull_mat:\n{}".format(chull_mat))
        print("chull_vec: {}".format(chull_vec))

    plot_constraints(chull_mat, chull_vec, col='g:', lw=2)

    for frac in [0.25, 0.5, 0.75]:
        row3 = [[0, 0, 0, 0, 1]]
        row4 = [[0, 0, 0, 0, -1]]
        fixed_mat = np.concatenate((chull_mat, row3, row4))
        fixed_vec = np.concatenate((chull_vec, [frac], [-frac]))
        plot_constraints(fixed_mat, fixed_vec, col='b-', lw=2)

    plt.show()


if __name__ == '__main__':
    init_plot_vecs()
    # main()
    a1_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    a1_vec = np.array([5, -4.5, 2, -1], dtype=float)

    plot_constraints(a1_mat, a1_vec)
    plt.show()
