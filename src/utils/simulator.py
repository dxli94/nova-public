import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu = 1


def van_der_pol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 - 1.0) * x[1] - x[0]
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1])
    return res


def predator_prey_deriv(x, t):
    nx0 = x[0] - x[0] * x[1]
    nx1 = -x[1] + x[0] * x[1]
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1])
    return res


def two_dim_water_tank_deriv(x, t):
    nx0 = 0.1+0.01*(4-x[1])+0.015*((2*9.81*x[0])**0.5)
    nx1 = 0.015*(2*9.81*x[0])**0.5-0.015*(2*9.81*x[1])**0.5

    res = np.array([nx0, nx1])
    return res


def free_ball_deriv(x, t):
    nx0 = x[1]
    nx1 = -9.81

    res = np.array([nx0, nx1])
    return res


def constant_moving(x, t):
    nx0 = 1
    nx1 = 0

    res = np.array([nx0, nx1])
    return res


def simulate_one_run(horizon, model, init_point):
    ts = np.linspace(0, horizon, horizon*200)

    if model == 'vanderpol':
        xs = odeint(van_der_pol_oscillator_deriv, init_point, ts)
    elif model == 'free_ball':
        xs = odeint(free_ball_deriv, init_point, ts)
    elif model == 'predator_prev':
        xs = odeint(predator_prey_deriv, init_point, ts)
    elif model == 'constant_moving':
        xs = odeint(constant_moving, init_point, ts)
    else:
        raise ValueError('Simulate eigen: invalid model name!')
    return xs[:, 0], xs[:, 1]


def simulate(horizon, model, init_coeff, init_col):
    from ConvexSet.Polyhedron import Polyhedron
    vertices = Polyhedron(init_coeff, init_col).vertices
    center = np.average(vertices, axis=0)
    print('simulate starting point is: {}'.format(center))
    return simulate_one_run(horizon, model, center)


def main(horizon):
    ts = np.linspace(0, horizon, horizon*100)

    xs = odeint(van_der_pol_oscillator_deriv, [1.25, 2.28], ts)
    # xs = odeint(two_dim_water_tank_deriv, [0, 8], ts)
    # xs = odeint(predator_prey_deriv, [3.44, 2.3], ts)
    # print('\n'.join(str(x) for x in list(enumerate(xs))))
    plt.plot(xs[:, 0], xs[:, 1])
    plt.gca().set_aspect('equal')
    # plt.savefig('vanderpol_oscillator.png')
    plt.autoscale()
    plt.show()

if __name__ == '__main__':
    main(10)