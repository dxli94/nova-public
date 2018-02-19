import numpy as np

import Utils
from ConvexSet.Polyhedron import Polyhedron
from Plotter import Plotter


def compute_initial_sf(poly, l, time_interval):
    initial_sf = poly.compute_support_function(l, poly.A, poly.b)[1]
    delta_tp = np.transpose(Utils.mat_exp(A, 1 * time_interval))
    first_sf = poly.compute_support_function(np.matmul(delta_tp, l), poly.A, poly.b)[1]
    return max(initial_sf, first_sf)


def compute_next_sf(poly, l):
    return poly.compute_support_function(l, poly.A, poly.b)[1]


def compute_post(poly, time_interval):

    ret = []

    delta_tp = np.transpose(Utils.mat_exp(A, time_interval))
    for idx in range(len(directions)):
        for n in time_frames:
            # delta_tp = np.transpose(mat_exp(A, n * time_interval))
            if n == 0:
                prev_r = directions[idx]
                prev_sf = compute_initial_sf(poly, directions[idx], time_interval)
                ret.append([prev_sf])
            else:
                r = np.matmul(delta_tp, prev_r)
                sf = compute_next_sf(poly, r)
                ret[-1].append(sf)
                prev_r = r

    return np.matrix(ret)


def get_images(sf_mat, directions):
    ret = []

    d_mat = np.matrix(directions)
    sf_mat = np.transpose(np.matrix(sf_mat))
    for sf_row in sf_mat:
        ret.append(Polyhedron(d_mat, np.transpose(sf_row)))
    return ret


def main():
    pass

if __name__ == '__main__':
    TIME_EPLASE = 20
    TIME_INTERVAL = 1

    main()

    directions = [
        np.array([-1, 0]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([1, -1]),
        np.array([-1, -1]),
        np.array([-1, 1]),
    ]
    # here comes a rectangle
    constr_A = [[-1, 0],  # -x1 <= 1
              [1, 0],   # x1 <= 2
              [0, -1],  # -x2 <= 0.5
              [0, 1]  # x2 <= 1
              ]
    constr_b = [[0], [2], [0.5], [0]]
    A = [[0, 1],
         [-2, 0]
         ]
    poly = Polyhedron(constr_A, constr_b)
    time_frames = range(int(np.ceil(TIME_EPLASE / TIME_INTERVAL)))

    # sfp = SupportFunctionProvider(poly)
    sf_mat = compute_post(poly, time_interval=TIME_INTERVAL)
    # print(sf_mat)
    images_by_time = get_images(sf_mat, directions)
    for image in images_by_time:
        print(image.vertices)
    # polygon_plotter(images_by_time)
    plHelper = Plotter(images_by_time)
    plHelper.Print()
