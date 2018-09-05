import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

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


def brusselator_deriv(x, t):
    nx0 = 1 + x[0]**2*x[1] - 1.5*x[0] - x[0]
    nx1 = 1.5 * x[0] - x[0]**2*x[1]

    res = np.array([nx0, nx1])
    return res


def jet_engine_deriv(x, t):
    # "-x1-1.5*x0^2-0.5*x0^3-0.5",
    # "3*x0-x1"
    nx0 = -x[1] - 1.5 * x[0]**2 - 0.5*x[0]**3 - 0.5
    nx1 = 3 * x[0] - x[1]

    res = np.array([nx0, nx1])
    return res


def buckling_column_deriv(x, t):
    # "x1",
    # "2*x0-x0^3-0.2*x1+0.1"
    nx0 = x[1]
    nx1 = 2*x[0] - x[0] ** 3 - 0.2 * x[1] + 0.1

    res = np.array([nx0, nx1])
    return res


def pbt_deriv(x, t):
    # "x1",
    # "2*x0-x0^3-0.2*x1+0.1"
    nx0 = x[1]
    nx1 = x[0] ** 2

    res = np.array([nx0, nx1])
    return res


def lacoperon_deriv(x, t):
    nx0 = -0.4*x[0]**2 *((0.0003*x[1]**2 + 0.008) / (0.2*x[0]**2 + 2.00001) ) + 0.012 + (0.0000003 * (54660 - 5000.006*x[0]) * (0.2*x[0]**2 + 2.00001)) / (0.00036*x[1]**2 + 0.00960018 + 0.000000018*x[0]**2)
    nx1 = -0.0006*x[1]**2 + (0.000000006*x[1]**2 + 0.00000016) / (0.2*x[0]**2 + 2.00001) + (0.0015015*x[0]*(0.2*x[0]**2 + 2.00001)) / (0.00036*x[1]**2 + 0.00960018 + 0.000000018*x[0]**2)

    res = np.array([nx0, nx1])
    return res


def constant_moving_deriv(x, t):
    nx0 = 1
    nx1 = 0

    res = np.array([nx0, nx1])
    return res


def coupled_vanderpol_deriv(x, t):
    nx0 = x[1]
    nx1 = -(x[0] ** 2.0 - 1.0) * x[1] - x[0] - (x[2] - x[0])
    nx2 = x[3]
    nx3 = -(x[2] ** 2.0 - 1.0) * x[3] - x[2] - (x[0] - x[2])
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1, nx2, nx3])
    return res


def spring_pendulum_deriv(x, t):
    # "x2",
    # "x3",
    # "x0*x3^2+9.8*cos(x1)-2*(x0-1)",
    # "-(1/x0)*(2*x2*x3+9.8*sin(x1))"
    nx0 = x[2]
    nx1 = x[3]
    nx2 = x[0]*x[3]**2+9.8*math.cos(x[1])-2*(x[0]-1)
    nx3 = -(1/x[0])*(2*x[2]*x[3]+9.8*math.sin(x[1]))

    res = np.array([nx0, nx1, nx2, nx3])
    return res


def lorentz_system_deriv(x, t):
    # "10*(x1-x0)",
    # "x0*(8/3-x2)-x1",
    # "x0*x1-28*x2"
    nx0 = 10*(x[1]-x[0])
    nx1 = x[0]*(28-x[2])-x[1]
    nx2 = x[0]*x[1]-2.6667*x[2]

    res = np.array([nx0, nx1, nx2])
    return res


def roessler_attractor_deriv(x, t):
    # "-x1-x2",
    # "x0+0.2*x1",
    # "0.2+x2*(x0-5.7)"
    nx0 = -x[1]-x[2]
    nx1 = x[0]+0.2*x[1]
    nx2 = 0.2+x[2]*(x[0]-5.7)

    res = np.array([nx0, nx1, nx2])
    return res


def biology_1_deriv(x, t):
    nx0 = -0.4*x[0] + 5*x[2]*x[3]
    nx1 = 0.4*x[0] - x[1]
    nx2 = x[1] - 5*x[2]*x[3]
    nx3 = 5*x[4]*x[5] - 5*x[2]*x[3]
    nx4 = -5*x[4]*x[5] + 5*x[2]*x[3]
    nx5 = 0.5*x[6] - 5*x[4]*x[5]
    nx6 = -0.5*x[6] + 5*x[4]*x[5]

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6])
    return res


def biology_2_deriv(x, t):
    # "3*x2 - x0*x5",
    # "x3 - x1*x5",
    # "x0*x5 - 3*x2",
    # "x1*x5 - x3",
    # "3*x2 + 5*x0 - x4",
    # "5*x4 + 3*x2 + x3 - x5*(x0 + x1 + 2*x7 + 1)",
    # "5*x3 + x1 - 0.5*x6",
    # "5*x6 - 2*x5*x7 + x8 - 0.2*x7",
    # "2*x5*x7 - x8"
    nx0 = 3*x[2] - x[0]*x[5]
    nx1 = x[3] - x[1]*x[5]
    nx2 = x[0]*x[5] - 3*x[2]
    nx3 = x[1]*x[5] - x[3]
    nx4 = 3*x[2] + 5*x[0] - x[4]
    nx5 = 5*x[4] + 3*x[2] + x[3] - x[5]*(x[0] + x[1] + 2*x[7] + 1)
    nx6 = 5*x[3] + x[1] - 0.5*x[6]
    nx7 = 5*x[6] - 2*x[5]*x[7] + x[8] - 0.2*x[7]
    nx8 = 2*x[5]*x[7] - x[8]

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8])
    return res


def laub_loomis_deriv(x, t):
    nx0 = 1.4 * x[2] - 0.9 * x[0]
    nx1 = 2.5 * x[4] - 1.5 * x[1]
    nx2 = 0.6 * x[6] - 0.8 * x[2] * x[1]
    nx3 = 2 - 1.3 * x[3] * x[2]
    nx4 = 0.7 * x[0] - x[3] * x[4]
    nx5 = 0.3 * x[0] - 3.1 * x[5]
    nx6 = 1.8 * x[5] - 1.5 * x[6] * x[1]

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6])
    return res


def controller_2d_deriv(x, t):
    # "x0*x1+x1^3+2",
    # "x0^2+2*x0-3*x1"
    nx0 = x[0]*x[1]+x[1]**3+2
    nx1 = x[0]**2+2*x[0]-3*x[1]

    res = np.array([nx0, nx1])
    return res


def controller_3d_deriv(x, t):
    # "10*(x1-x0)",
    # "x0^3",
    # "x0*x1-2.667*x2"
    nx0 = 10*(x[1]-x[0])
    nx1 = x[0]**3
    nx2 = x[0]*x[1]-2.6667*x[2]

    res = np.array([nx0, nx1, nx2])
    return res


def wattsteam_deriv(x, t):
    # "x1",
    # "(x2^2*cos(x0)-1)*sin(x0)-3*x1",
    # "cos(x0)-1"

    nx0 = x[1]
    nx1 = (x[2]**2*math.cos(x[0])-1)*math.sin(x[0])-3*x[1]
    nx2 = math.cos(x[0])-1

    res = np.array([nx0, nx1, nx2])
    return res


def simulate_one_run(horizon, model, init_point):
    ts = np.linspace(0, horizon, horizon*5000)

    if model == 'vanderpol':
        xs = odeint(van_der_pol_oscillator_deriv, init_point, ts)
    elif model == 'free_ball':
        xs = odeint(free_ball_deriv, init_point, ts)
    elif model == 'predator_prev':
        xs = odeint(predator_prey_deriv, init_point, ts)
    elif model == 'constant_moving':
        xs = odeint(constant_moving_deriv, init_point, ts)
    elif model == 'brusselator':
        xs = odeint(brusselator_deriv, init_point, ts)
    elif model == 'jet_engine':
        xs = odeint(jet_engine_deriv, init_point, ts)
    elif model == 'buckling_column':
        xs = odeint(buckling_column_deriv, init_point, ts)
    elif model == 'pbt':
        xs = odeint(pbt_deriv, init_point, ts)
    elif model == '2d_controller':
        xs = odeint(controller_2d_deriv, init_point, ts)
    elif model == '3d_controller':
        xs = odeint(controller_3d_deriv, init_point, ts)
    elif model == 'lacoperon':
        xs = odeint(lacoperon_deriv, init_point, ts)
    elif model == 'watt_steam':
        xs = odeint(wattsteam_deriv, init_point, ts)
    elif model == 'coupled_vanderpol':
        xs = odeint(coupled_vanderpol_deriv, init_point, ts)
    elif model == 'spring_pendulum':
        xs = odeint(spring_pendulum_deriv, init_point, ts)
    elif model == 'lorentz_system':
        xs = odeint(lorentz_system_deriv, init_point, ts)
    elif model == 'roessler_attractor':
        xs = odeint(roessler_attractor_deriv, init_point, ts)
    elif model == 'biology_1':
        xs = odeint(biology_1_deriv, init_point, ts)
    elif model == 'biology_2':
        xs = odeint(biology_2_deriv, init_point, ts)
    elif model == 'laub_loomis':
        xs = odeint(laub_loomis_deriv, init_point, ts)
    else:
        raise ValueError('Simulate eigen: invalid model name!')
    return xs
    # return xs[:, opdims[0]], xs[:, opdims[1]]


def simulate(horizon, model, init_coeff, init_col):
    from ConvexSet.Polyhedron import Polyhedron
    vertices = Polyhedron(init_coeff, init_col).get_vertices()
    # center = np.average(vertices, axis=0)
    center = vertices[-1]
    print('simulate starting point is: {}'.format(center))
    return simulate_one_run(horizon, model, center)


def save_simu_traj(xs, filename):
    with open(filename, 'w') as simu_op:
        for row in xs:
        # with open('../out/simu.out', 'w') as simu_op:
            for elem in row:
                simu_op.write(str(elem) + ' ')
            simu_op.write('\n')

def main(horizon):
    ts = np.linspace(0, 2, 1500)

    import time
    # start_time = time.time()
    # for i in range(1000000):
    #     xs = odeint(brusselator_deriv, [2, 0.28], ts)
    # print(time.time() - start_time)
    # exit()
    xs = odeint(biology_2_deriv, [1, 1, 1, 1, 1, 1, 1, 1, 1], ts)
    # xs = odeint(van_der_pol_oscillator_deriv, [1.25, 2.28], ts)
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