import numpy as np
from scipy.optimize import leastsq


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


def _fit_non_linear_least_sqr(points, values, p0):
    return leastsq(residuals, p0, args=(values, points))[0]


def least_sqr_fit(abs_domain, abs_centre, n, p0, islinear, eval_func):
    points = sampling_around_centre(abs_domain, abs_centre, n)
    y0 = eval_func(points.T)

    mat_a = []
    for i in range(len(y0)):
        if islinear[i]:
            mat_a.append(_fit_linear(points, y0[i]))
        else:
            mat_a.append(_fit_non_linear_least_sqr(points, y0[i], p0))

    return mat_a


def jacobian_linearise(abs_centre, jacobian_func, variables, eval_func):
    # f(x0, g0) = J(x - x0, y - y0) * [x - x0, y - y0].T + g(x0, y0)
    mat_a = np.array(jacobian_func.subs(list(zip(variables, abs_centre)))).astype(np.float64)
    b = eval_func('', *abs_centre) - mat_a.dot(abs_centre)

    # mat_a = np.array([[0, 1], [-1, -1]])
    return mat_a, b

if __name__ == '__main__':
    from ConvexSet.HyperBox import HyperBox
    import sympy
    abs_domain = HyperBox(np.array([[0, 0.7], [0, 0.71], [0.1, 0.7], [0.1, 0.71]]))
    # print(abs_domain.vertices)
    abs_centre = np.average(abs_domain.vertices, axis=1)[0]
    sym_vars = sympy.symbols('x,y', real=True)
    x, y = sym_vars
    non_linear_dynamics = [y, (1 - x ** 2) * y - x]

    sym_dyn = sympy.Matrix(non_linear_dynamics).jacobian(sym_vars)

    mat_a, u = jacobian_linearise(abs_centre, sym_dyn, sym_vars)

    print(mat_a, u)
