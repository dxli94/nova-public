from scipy.linalg import expm
from Polyhedron import Polyhedron
from SupportFunctionProvider import SupportFunctionProvider
from PlottingHelper import PlottingHelper
import numpy as np


def mat_exp(A, t):
    # return expm(A)
    return expm(np.multiply(A, t))


def compute_initial_sf(sf_provider, l, time_interval):
    initial_sf = sf_provider.compute_support_function(l)[1]
    delta_tp = np.transpose(mat_exp(A, 1 * time_interval))
    first_sf = sf_provider.compute_support_function(np.matmul(delta_tp, l))[1]
    return max(initial_sf, first_sf)


def compute_next_sf(sf_provider, l):
    return sf_provider.compute_support_function(l)[1]


def compute_post(sf_provider, time_interval):

    ret = []

    delta_tp = np.transpose(mat_exp(A, time_interval))
    for idx in range(len(directions)):
        for n in time_frames:
            # delta_tp = np.transpose(mat_exp(A, n * time_interval))
            if n == 0:
                prev_r = directions[idx]
                prev_sf = compute_initial_sf(sf_provider, directions[idx], time_interval)
                ret.append([prev_sf])
            else:
                r = np.matmul(delta_tp, prev_r)
                sf = compute_next_sf(sf_provider, r)
                ret[-1].append(sf)
                prev_r = r

    return np.matrix(ret)


def get_images(sf_mat, directions):
    ret = []

    d_mat = np.matrix(directions)
    sf_mat = np.transpose(np.matrix(sf_mat))
    for sf_row in sf_mat:
        ret.append(Polyhedron(np.hstack((d_mat, np.transpose(sf_row)))))
    return ret


def polygon_plotter(images):
    import matplotlib.pyplot as plt
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    # print(images[0].vertices)
    for im in images:
        poly1patch = plt.Polygon(np.transpose(np.matrix([[elem[0] for elem in im.vertices],
                                  [elem[1] for elem in im.vertices]
                                  ])), fill=False)
        ax.add_patch(poly1patch)
    xrange = [-5, 5]
    yrange = [-2, 2]
    ax.set_xlim(*xrange)
    # ax.set_xticks(range(*xrange) + [xrange[-1]])
    ax.set_ylim(*yrange)
    # ax.set_yticks(range(*yrange) + [yrange[-1]])
    # ax.set_aspect(1)

    plt.show()


if __name__ == '__main__':
    TIME_EPLASE = 20
    TIME_INTERVAL = 1

    # directions = [
    #     np.array([1, 0]),
    #     np.array([0, 1]),
    #     np.array([-1, 0]),
    #     np.array([0, -1]),
    # ]
    directions = [
        np.array([-1, 0]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([0, 1]),
        # np.array([1, 1]),
        # np.array([1, -1]),
        # np.array([-1, -1]),
        # np.array([-1, 1]),
    ]
    # here comes a rectangle
    constr = [[-1, 0, 0],  # -x1 <= 1
              [1, 0, 2],   # x1 <= 2
              [0, -1, 0.5],  # -x2 <= 0.5
              [0, 1, 0]  # x2 <= 1
              ]
    A = [[0, 1],
         [-2, 0]
         ]
    poly = Polyhedron(constr)
    time_frames = range(int(np.ceil(TIME_EPLASE / TIME_INTERVAL)))

    sfp = SupportFunctionProvider(poly)
    sf_mat = compute_post(sfp, time_interval=TIME_INTERVAL)
    # print(sf_mat)
    images_by_time = get_images(sf_mat, directions)
    for image in images_by_time:
        print(image.vertices)
    # polygon_plotter(images_by_time)
    plHelper = PlottingHelper(images_by_time)
    plHelper.Print()