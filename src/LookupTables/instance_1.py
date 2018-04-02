# a random example just to
from ConvexSet.HyperBox import HyperBox
import numpy as np


def get_init():
    bounds = [[0, 0.1],
              [0, 0.1]]
    return HyperBox(np.array(bounds))


def get_table():
    #    x     y  x'=     y'=
    #             x(y-1)  y(x-1)
    data = [(0.0, 0.0, -0.0, -0.0),
            (0.0, 0.5, -0.0, -0.5),
            (0.0, 1.0, 0.0, -1.0),
            (0.5, 0.0, -0.5, -0.0),
            (0.5, 0.5, -0.25, -0.25),
            (0.5, 1.0, 0.0, -0.5),
            (1.0, 0.0, -1.0, 0.0),
            (1.0, 0.5, -0.5, 0.0),
            (1.0, 1.0, 0.0, 0.0)]

    lookup_table = dict()
    for elem in data:
        lookup_table[tuple(elem[:2])] = tuple(elem[2:])

    return lookup_table
