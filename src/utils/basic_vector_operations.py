import numpy as np


def normalize_vector(u):
    u = np.array(u)
    sqr_sum = (sum(ele ** 2 for ele in u))**0.5

    return u / sqr_sum


def is_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    if np.isclose(x1, x2):
        if np.isclose(x2, x3):
            return True
        else:
            m = (y2 - y3) / (x2 - x3)
            c = y3 - m * x3
            return np.isclose(m*x1+c, y1)
    else:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        return np.isclose(m*x3+c, y3)
