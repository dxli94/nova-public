# a random example just to
import numpy as np


def get_init():
    vertices = [[0, 0],
                [0, 0.1],
                [0.1, 0],
                [0.1, 0.1]]
    return np.array(vertices)


def get_table():
    #         x     y  x'= y   y'= (1-x^2)*y-x
    data = [(0.0, 0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5, 0.5),
            (0.0, 1.0, 1.0, 1.0),
            (0.5, 0.0, 0.0, -0.5),
            (0.5, 0.5, 0.5, -0.125),
            (0.5, 1.0, 1.0, 0.25),
            (1.0, 0.0, 0.0, -1.0),
            (1.0, 0.5, 0.5, -1.0),
            (1.0, 1.0, 1.0, -1.0)]

    lookup_table = dict()
    for elem in data:
        lookup_table[tuple(elem[:2])] = tuple(elem[2:])

    return lookup_table
