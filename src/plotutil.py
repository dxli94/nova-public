'plot utilities'

import math
import numpy as np
import matplotlib.pyplot as plt

plot_vecs = None # list of vectors to optimize in for plotting, assigned in Star.init_plot_vecs

def init_plot_vecs(num_angles=512):
    'initialize plot_vecs'

    global plot_vecs

    plot_vecs = []

    step = 2.0 * math.pi / num_angles

    for theta in np.arange(0.0, 2.0*math.pi, step):
        x = math.cos(theta)
        y = math.sin(theta)

        vec = np.array([x, y], dtype=float)

        plot_vecs.append(vec)

def get_verts(lpi):
    'get the verticies of the polygon projection of the star used for plotting'

    pts = find_boundary_pts(lpi)

    verts = [[pt[0], pt[1]] for pt in pts]

    # wrap polygon back to first point
    verts.append(verts[0])

    return verts



def minimize(lpi, direction):
    'minimize to lp... returning the 2-d point of the first 2 coordinates of the minimum'

    num_cols = lpi.get_num_cols()
    
    direction_full_vec = [direction[0], direction[1]] + ([0] * (num_cols - 2))

    return lpi.minimize(direction_full_vec)


def find_boundary_pts(lpi):
    '''
    find a constaint-star's boundaries using Star.plot_vecs. This solves several LPs and
    returns a list of points on the boundary (in the standard basis) which maximize each
    of the passed-in directions
    '''

    global plot_vecs

    if plot_vecs is None:
        init_plot_vecs()

    direction_list = plot_vecs
    rv = []

    assert len(direction_list) >= 2

    # optimized approach: do binary search to find changes
    point = minimize(lpi, direction_list[0])
    rv.append(point.copy())

    # add it in thirds, to ensure we don't miss anything
    third = len(direction_list) // 3

    # 0 to 1/3
    point = minimize(lpi, direction_list[third])

    if not np.array_equal(point, rv[-1]):
        rv += binary_search_boundaries(lpi, 0, third, rv[-1], point)
        rv.append(point.copy())

    # 1/3 to 2/3
    point = minimize(lpi, direction_list[2*third])

    if not np.array_equal(point, rv[-1]):
        rv += binary_search_boundaries(lpi, third, 2*third, rv[-1], point)
        rv.append(point.copy())

    # 2/3 to end
    point = minimize(lpi, direction_list[-1])

    if not np.array_equal(point, rv[-1]):
        rv += binary_search_boundaries(lpi, 2*third, len(direction_list) - 1, rv[-1], point)
        rv.append(point.copy())

    # pop last point if it's the same as the first point
    if len(rv) > 1 and np.array_equal(rv[0], rv[-1]):
        rv.pop()

    return rv

def binary_search_boundaries(lpi, start, end, start_point, end_point):
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

        mid_point = minimize(lpi, dirs[mid])

        not_start = not np.allclose(start_point, mid_point, atol=1e-3)
        not_end = not np.allclose(end_point, mid_point, atol=1e-3)

        if not_start:
            rv += binary_search_boundaries(lpi, start, mid, start_point, mid_point)

        if not_start and not_end:
            rv.append(mid_point)

        if not_end:
            rv += binary_search_boundaries(lpi, mid, end, mid_point, end_point)

    return rv

def add_plot(lpi, col='k-', lw=1):
    '''
    plot a polytope given in constraint form (by taking a fine-grained support function outer-approximation)

    This calls plt.plot(), so call plt.show() afterwards
    '''

    verts = get_verts(lpi)

    # wrap polygon back to first point
    verts.append(verts[0])

    xs = [ele[0] for ele in verts]
    ys = [ele[1] for ele in verts]

    plt.plot(xs, ys, col, lw=lw)
