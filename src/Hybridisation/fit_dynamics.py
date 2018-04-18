import numpy as np
from scipy.optimize import leastsq
from ConvexSet.HyperBox import HyperBox


def evaluate_exp(nonli_dyn, x, y):
    # for now just hard-code it
    # return np.array([y, -1*x-y])
    return np.array([y, (1 - x * x) * y - x])


def approx_func(x, p):
    return np.dot(x, p.reshape(2, 1).flatten())


def sampling_around_centre(abs_domain, abs_centre, n):
    coord = {}
    for i, bd in enumerate(abs_domain.bounds):
        k = bd[1] - bd[0]
        b = bd[0]

        x = k * np.random.random_sample(n) + b
        coord[i] = x

    points = list(zip(*(coord[i] for i in range(len(abs_domain.bounds)))))
    points = np.add(np.subtract(points, abs_centre) / 5, abs_centre)

    return points


def residuals(p, y, x):
    return (y - approx_func(x, p)).tolist()


def _fit_linear(points, values):
    """
        Least Square Regression cannot converge perfectly in linear case.
        We solve a linear equation instead.
    """
    pairs = list(zip(points, values))
    for i in range(len(pairs)):
        for j in range(i, len(pairs)):
            x = [pairs[i][0], pairs[j][0]]
            b_x = [pairs[i][1], pairs[j][1]]
            try:
                a_x = np.linalg.solve(x, b_x)
            except np.linalg.LinAlgError:
                continue
    return a_x


def _fit_non_linear(points, values):
    return leastsq(residuals, [0, 0], args=(values, points))[0]


def fit(abs_domain, abs_centre, n, p0, islinear):
    points = sampling_around_centre(abs_domain, abs_centre, n)
    y0 = evaluate_exp('', *points.T)

    mat_a = []
    for i in range(len(y0)):
        if islinear[i]:
            mat_a.append(_fit_linear(points, y0[i]))
        else:
            mat_a.append(_fit_non_linear(points, y0[i]))

    return mat_a


if __name__ == '__main__':
    pass