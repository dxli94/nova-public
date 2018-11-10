import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import sys


def _vanderpol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -1 * (x[0] ** 2.0 - 1.0) * x[1] - x[0]
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1])
    return res


def _predator_prey_deriv(x, t):
    nx0 = 1.5*x[0] - x[0] * x[1]
    nx1 = -3*x[1] + x[0] * x[1]
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1])
    return res


def _2d_tank_deriv(x, t):
    nx0 = 0.1+0.01*(4-x[1])+0.015*((2*9.81*x[0])**0.5)
    nx1 = 0.015*(2*9.81*x[0])**0.5-0.015*(2*9.81*x[1])**0.5

    res = np.array([nx0, nx1])
    return res


def _free_ball_deriv(x, t):
    nx0 = x[1]
    nx1 = -9.81

    res = np.array([nx0, nx1])
    return res


def _brusselator_deriv(x, t):
    nx0 = 1 + x[0]**2*x[1] - 1.5*x[0] - x[0]
    nx1 = 1.5 * x[0] - x[0]**2*x[1]

    res = np.array([nx0, nx1])
    return res


def _jet_engine_deriv(x, t):
    # "-x1-1.5*x0^2-0.5*x0^3-0.5",
    # "3*x0-x1"
    nx0 = -x[1] - 1.5 * x[0]**2 - 0.5*x[0]**3 - 0.5
    nx1 = 3 * x[0] - x[1]

    res = np.array([nx0, nx1])
    return res


def _buckling_column_deriv(x, t):
    # "x1",
    # "2*x0-x0^3-0.2*x1+0.1"
    nx0 = x[1]
    nx1 = 2*x[0] - x[0] ** 3 - 0.2 * x[1] + 0.1

    res = np.array([nx0, nx1])
    return res


def _pbt_deriv(x, t):
    # "x1",
    # "2*x0-x0^3-0.2*x1+0.1"
    nx0 = x[1]
    nx1 = x[0] ** 2

    res = np.array([nx0, nx1])
    return res


def _lacoperon_deriv(x, t):
    nx0 = -0.4*x[0]**2 *((0.0003*x[1]**2 + 0.008) / (0.2*x[0]**2 + 2.00001) ) + 0.012 + (0.0000003 * (54660 - 5000.006*x[0]) * (0.2*x[0]**2 + 2.00001)) / (0.00036*x[1]**2 + 0.00960018 + 0.000000018*x[0]**2)
    nx1 = -0.0006*x[1]**2 + (0.000000006*x[1]**2 + 0.00000016) / (0.2*x[0]**2 + 2.00001) + (0.0015015*x[0]*(0.2*x[0]**2 + 2.00001)) / (0.00036*x[1]**2 + 0.00960018 + 0.000000018*x[0]**2)

    res = np.array([nx0, nx1])
    return res


def _constant_moving_deriv(x, t):
    nx0 = 1
    nx1 = 0

    res = np.array([nx0, nx1])
    return res


def _coupled_vanderpol_deriv(x, t):
    nx0 = x[1]
    nx1 = -(x[0] ** 2.0 - 1.0) * x[1] - x[0] + (x[2] - x[0])
    nx2 = x[3]
    nx3 = -(x[2] ** 2.0 - 1.0) * x[3] - x[2] + (x[0] - x[2])
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1, nx2, nx3])
    return res

def _coupled_vanderpol_6d_deriv(x, t):
    # "x1",
    # "(1 - x0^2)*x1 - x0 + (x2 - x0)",
    # "x3",
    # "(1 - x2^2)*x3 - x2 + (x0 - x2) + (x4 - x2)",
    # "x5",
    # "(1 - x4^2)*x5 - x4 + (x2 - x4)"
    nx0 = x[1]
    nx1 = -(x[0] ** 2.0 - 1.0) * x[1] - x[0] + (x[2] - x[0])
    nx2 = x[3]
    nx3 = -(x[2] ** 2.0 - 1.0) * x[3] - x[2] + (x[0] - x[2]) + (x[4] - x[2])
    nx4 = x[5]
    nx5 = -(x[4] ** 2.0 - 1.0) * x[5] - x[4] + (x[2] - x[4])
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5])
    return res


def _coupled_vanderpol_8d_deriv(x, t):
    # "x1",
    # "(1 - x0^2)*x1 - x0 + (x2 - x0)",
    # "x3",
    # "(1 - x2^2)*x3 - x2 + (x0 - x2) + (x4 - x2)",
    # "x5",
    # "(1 - x4^2)*x5 - x4 + (x2 - x4) + (x6 - x4)",
    # "x7",
    # "(1 - x6^2)*x7 - x6 + (x4 - x6)"
    nx0 = x[1]
    nx1 = -(x[0] ** 2.0 - 1.0) * x[1] - x[0] + (x[2] - x[0])
    nx2 = x[3]
    nx3 = -(x[2] ** 2.0 - 1.0) * x[3] - x[2] + (x[0] - x[2]) + (x[4] - x[2])
    nx4 = x[5]
    nx5 = -(x[4] ** 2.0 - 1.0) * x[5] - x[4] + (x[2] - x[4]) + (x[6] - x[4])
    nx6 = x[7]
    nx7 = -(x[6] ** 2.0 - 1.0) * x[7] - x[6] + (x[4] - x[6])
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7])
    return res


def _spring_pendulum_deriv(x, t):
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


def _lorentz_system_deriv(x, t):
    # "10*(x1-x0)",
    # "x0*(8/3-x2)-x1",
    # "x0*x1-28*x2"
    nx0 = 10*(x[1]-x[0])
    nx1 = x[0]*(28-x[2])-x[1]
    nx2 = x[0]*x[1]-2.6667*x[2]

    res = np.array([nx0, nx1, nx2])
    return res


def _roessler_attractor_deriv(x, t):
    # "-x1-x2",
    # "x0+0.2*x1",
    # "0.2+x2*(x0-5.7)"
    nx0 = -x[1]-x[2]
    nx1 = x[0]+0.2*x[1]
    nx2 = 0.2+x[2]*(x[0]-5.7)

    res = np.array([nx0, nx1, nx2])
    return res


def _biology_1_deriv(x, t):
    nx0 = -0.4*x[0] + 5*x[2]*x[3]
    nx1 = 0.4*x[0] - x[1]
    nx2 = x[1] - 5*x[2]*x[3]
    nx3 = 5*x[4]*x[5] - 5*x[2]*x[3]
    nx4 = -5*x[4]*x[5] + 5*x[2]*x[3]
    nx5 = 0.5*x[6] - 5*x[4]*x[5]
    nx6 = -0.5*x[6] + 5*x[4]*x[5]

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6])
    return res


def _biology_2_deriv(x, t):
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


def _laub_loomis_deriv(x, t):
    nx0 = 1.4 * x[2] - 0.9 * x[0]
    nx1 = 2.5 * x[4] - 1.5 * x[1]
    nx2 = 0.6 * x[6] - 0.8 * x[2] * x[1]
    nx3 = 2 - 1.3 * x[3] * x[2]
    nx4 = 0.7 * x[0] - x[3] * x[4]
    nx5 = 0.3 * x[0] - 3.1 * x[5]
    nx6 = 1.8 * x[5] - 1.5 * x[6] * x[1]

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6])
    return res


def _controller_2d_deriv(x, t):
    # "x0*x1+x1^3+2",
    # "x0^2+2*x0-3*x1"
    nx0 = x[0]*x[1]+x[1]**3+2
    nx1 = x[0]**2+2*x[0]-3*x[1]

    res = np.array([nx0, nx1])
    return res


def _controller_3d_deriv(x, t):
    # "10*(x1-x0)",
    # "x0^3",
    # "x0*x1-2.667*x2"
    nx0 = 10*(x[1]-x[0])
    nx1 = x[0]**3
    nx2 = x[0]*x[1]-2.6667*x[2]

    res = np.array([nx0, nx1, nx2])
    return res


def _wattsteam_deriv(x, t):
    # "x1",
    # "(x2^2*cos(x0)-1)*sin(x0)-3*x1",
    # "cos(x0)-1"

    nx0 = x[1]
    nx1 = (x[2]**2*math.cos(x[0])-1)*math.sin(x[0])-3*x[1]
    nx2 = math.cos(x[0])-1

    res = np.array([nx0, nx1, nx2])
    return res


def coupled_osc_5d(x, t):
    nx0 = 0.1*x[4]-3*x[0]+10.0*(x[3])
    nx1 = 10*x[0]-2.2*x[1]
    nx2 = 10*x[1]-1.5*x[2]
    nx3 = 2*x[0]-20*x[3]
    nx4 = -5*x[4]**2*x[2]**4*(10*x[1]-1.5*x[2])

    res = np.array([nx0, nx1, nx2, nx3, nx4])
    return res


def coupled_osc_10d(x, t):
    nx0 = 0.1*x[4]-3*x[0]+5.0*(x[3] + x[8])
    nx1 = 10*x[0]-2.2*x[1]
    nx2 = 10*x[1]-1.5*x[2]
    nx3 = 2*x[0]-20*x[3]
    nx4 = -5*x[4]**2*x[2]**4*(10*x[1]-1.5*x[2])

    nx5 = 0.1*x[9]-3*x[5]+5.0*(x[3]+x[8])
    nx6 = 10*x[5]-2.2*x[6]
    nx7 = 10*x[6]-1.5*x[7]
    nx8 = 2*x[5]-20*x[8]
    nx9 = -5*x[9]**2*x[7]**4*(10*x[6]-1.5*x[7])

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8, nx9])
    return res


def coupled_osc_15d(x, t):
    nx0 = 0.1*x[4]-3*x[0]+3.33*(x[3] + x[8] + x[13])
    nx1 = 10*x[0]-2.2*x[1]
    nx2 = 10*x[1]-1.5*x[2]
    nx3 = 2*x[0]-20*x[3]
    nx4 = -5*x[4]**2*x[2]**4*(10*x[1]-1.5*x[2])

    nx5 = 0.1*x[9]-3*x[5]+3.33*(x[3] + x[8] + x[13])
    nx6 = 10*x[5]-2.2*x[6]
    nx7 = 10*x[6]-1.5*x[7]
    nx8 = 2*x[5]-20*x[8]
    nx9 = -5*x[9]**2*x[7]**4*(10*x[6]-1.5*x[7])

    nx10 = 0.1*x[14]-3*x[10]+3.33*(x[3]+x[8]+x[13])
    nx11 = 10*x[10]-2.2*x[11]
    nx12 = 10*x[11]-1.5*x[12]
    nx13 = 2*x[10]-20*x[13]
    nx14 = -5*x[14]**2*x[12]**4*(10*x[11]-1.5*x[12])

    res = np.array([nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8, nx9, nx10, nx11, nx12, nx13, nx14])
    return res


def coupled_osc_20d(x, t):
    nx0 = 0.1*x[4]-3*x[0]+2.5*(x[3] + x[8] + x[13] + x[18])
    nx1 = 10*x[0]-2.2*x[1]
    nx2 = 10*x[1]-1.5*x[2]
    nx3 = 2*x[0]-20*x[3]
    nx4 = -5*x[4]**2*x[2]**4*(10*x[1]-1.5*x[2])

    nx5 = 0.1*x[9]-3*x[5]+2.5*(x[3] + x[8] + x[13] + x[18])
    nx6 = 10*x[5]-2.2*x[6]
    nx7 = 10*x[6]-1.5*x[7]
    nx8 = 2*x[5]-20*x[8]
    nx9 = -5*x[9]**2*x[7]**4*(10*x[6]-1.5*x[7])

    nx10 = 0.1*x[14]-3*x[10]+2.5*(x[3]+x[8]+x[13]+ x[18])
    nx11 = 10*x[10]-2.2*x[11]
    nx12 = 10*x[11]-1.5*x[12]
    nx13 = 2*x[10]-20*x[13]
    nx14 = -5*x[14]**2*x[12]**4*(10*x[11]-1.5*x[12])

    nx15 = 0.1*x[19]-3*x[15]+2.5*(x[3]+x[8]+x[13]+x[18])
    nx16 = 10*x[15]-2.2*x[16]
    nx17 = 10*x[16]-1.5*x[17]
    nx18 = 2*x[15]-20*x[18]
    nx19 = -5*x[19]**2*x[17]**4*(10*x[16]-1.5*x[17])

    res = np.array([nx0, nx1, nx2, nx3, nx4,
                    nx5, nx6, nx7, nx8, nx9,
                    nx10, nx11, nx12, nx13, nx14,
                    nx15, nx16, nx17, nx18, nx19])
    return res


def coupled_osc_25d(x, t):
    nx0 = 0.1*x[4]-3*x[0]+2*(x[3]+x[8]+x[13]+x[18]+x[23])
    nx1 = 10*x[0]-2.2*x[1]
    nx2 = 10*x[1]-1.5*x[2]
    nx3 = 2*x[0]-20*x[3]
    nx4 = -5*x[4]**2*x[2]**4*(10*x[1]-1.5*x[2])

    nx5 = 0.1*x[9]-3*x[5]+2*(x[3]+x[8]+x[13]+x[18]+x[23])
    nx6 = 10*x[5]-2.2*x[6]
    nx7 = 10*x[6]-1.5*x[7]
    nx8 = 2*x[5]-20*x[8]
    nx9 = -5*x[9]**2*x[7]**4*(10*x[6]-1.5*x[7])

    nx10 = 0.1*x[14]-3*x[10]+2*(x[3]+x[8]+x[13]+x[18]+x[23])
    nx11 = 10*x[10]-2.2*x[11]
    nx12 = 10*x[11]-1.5*x[12]
    nx13 = 2*x[10]-20*x[13]
    nx14 = -5*x[14]**2*x[12]**4*(10*x[11]-1.5*x[12])

    nx15 = 0.1*x[19]-3*x[15]+2*(x[3]+x[8]+x[13]+x[18]+x[23])
    nx16 = 10*x[15]-2.2*x[16]
    nx17 = 10*x[16]-1.5*x[17]
    nx18 = 2*x[15]-20*x[18]
    nx19 = -5*x[19]**2*x[17]**4*(10*x[16]-1.5*x[17])

    nx20 = 0.1*x[24]-3*x[20]+2*(x[3]+x[8]+x[13]+x[18]+x[23])
    nx21 = 10*x[20]-2.2*x[21]
    nx22 = 10*x[21]-1.5*x[22]
    nx23 = 2*x[20]-20*x[23]
    nx24 = -5*x[24]**2*x[22]**4*(10*x[21]-1.5*x[22])

    res = np.array([nx0, nx1, nx2, nx3, nx4,
                    nx5, nx6, nx7, nx8, nx9,
                    nx10, nx11, nx12, nx13, nx14,
                    nx15, nx16, nx17, nx18, nx19,
                    nx20, nx21, nx22, nx23, nx24])
    return res


def coupled_osc_30d(x, t):
    nx0 = 0.1*x[4]-3*x[0]+1.6666666666666667*(x[3]+x[8]+x[13]+x[18]+x[23]+x[28])
    nx1 = 10*x[0]-2.2*x[1]
    nx2 = 10*x[1]-1.5*x[2]
    nx3 = 2*x[0]-20*x[3]
    nx4 = -5*x[4]**2*x[2]**4*(10*x[1]-1.5*x[2])

    nx5 = 0.1*x[9]-3*x[5]+1.6666666666666667*(x[3]+x[8]+x[13]+x[18]+x[23]+x[28])
    nx6 = 10*x[5]-2.2*x[6]
    nx7 = 10*x[6]-1.5*x[7]
    nx8 = 2*x[5]-20*x[8]
    nx9 = -5*x[9]**2*x[7]**4*(10*x[6]-1.5*x[7])

    nx10 = 0.1*x[14]-3*x[10]+1.6666666666666667*(x[3]+x[8]+x[13]+x[18]+x[23]+x[28])
    nx11 = 10*x[10]-2.2*x[11]
    nx12 = 10*x[11]-1.5*x[12]
    nx13 = 2*x[10]-20*x[13]
    nx14 = -5*x[14]**2*x[12]**4*(10*x[11]-1.5*x[12])

    nx15 = 0.1*x[19]-3*x[15]+1.6666666666666667*(x[3]+x[8]+x[13]+x[18]+x[23]+x[28])
    nx16 = 10*x[15]-2.2*x[16]
    nx17 = 10*x[16]-1.5*x[17]
    nx18 = 2*x[15]-20*x[18]
    nx19 = -5*x[19]**2*x[17]**4*(10*x[16]-1.5*x[17])

    nx20 = 0.1*x[24]-3*x[20]+1.6666666666666667*(x[3]+x[8]+x[13]+x[18]+x[23]+x[28])
    nx21 = 10*x[20]-2.2*x[21]
    nx22 = 10*x[21]-1.5*x[22]
    nx23 = 2*x[20]-20*x[23]
    nx24 = -5*x[24]**2*x[22]**4*(10*x[21]-1.5*x[22])

    nx25 = 0.1*x[29]-3*x[25]+1.6666666666666667*(x[3]+x[8]+x[13]+x[18]+x[23]+x[28])
    nx26 = 10*x[25]-2.2*x[26]
    nx27 = 10*x[26]-1.5*x[27]
    nx28 = 2*x[25]-20*x[28]
    nx29 = -5*x[29]**2*x[27]**4*(10*x[26]-1.5*x[27])

    res = np.array([nx0, nx1, nx2, nx3, nx4,
                    nx5, nx6, nx7, nx8, nx9,
                    nx10, nx11, nx12, nx13, nx14,
                    nx15, nx16, nx17, nx18, nx19,
                    nx20, nx21, nx22, nx23, nx24,
                    nx25, nx26, nx27, nx28, nx29])
    return res


def _simulate_one_run(horizon, model, init_point):
    ts = np.linspace(0, horizon, horizon*500)

    if model == 'vanderpol':
        xs = odeint(_vanderpol_oscillator_deriv, init_point, ts)
    elif model == 'free_ball':
        xs = odeint(_free_ball_deriv, init_point, ts)
    elif model == 'predator_prev':
        xs = odeint(_predator_prey_deriv, init_point, ts)
    elif model == 'constant_moving':
        xs = odeint(_constant_moving_deriv, init_point, ts)
    elif model == 'brusselator':
        xs = odeint(_brusselator_deriv, init_point, ts)
    elif model == 'jet_engine':
        xs = odeint(_jet_engine_deriv, init_point, ts)
    elif model == 'buckling_column':
        xs = odeint(_buckling_column_deriv, init_point, ts)
    elif model == 'pbt':
        xs = odeint(_pbt_deriv, init_point, ts)
    elif model == '2d_controller':
        xs = odeint(_controller_2d_deriv, init_point, ts)
    elif model == '3d_controller':
        xs = odeint(_controller_3d_deriv, init_point, ts)
    elif model == 'lacoperon':
        xs = odeint(_lacoperon_deriv, init_point, ts)
    elif model == 'watt_steam':
        xs = odeint(_wattsteam_deriv, init_point, ts)
    elif model == 'coupled_vanderpol':
        xs = odeint(_coupled_vanderpol_deriv, init_point, ts)
    elif model == 'coupled_vanderpol_6d':
        xs = odeint(_coupled_vanderpol_6d_deriv, init_point, ts)
    elif model == 'coupled_vanderpol_8d':
        xs = odeint(_coupled_vanderpol_8d_deriv, init_point, ts)
    elif model == 'spring_pendulum':
        xs = odeint(_spring_pendulum_deriv, init_point, ts)
    elif model == 'lorentz_system':
        xs = odeint(_lorentz_system_deriv, init_point, ts)
    elif model == 'roessler_attractor':
        xs = odeint(_roessler_attractor_deriv, init_point, ts)
    elif model == 'biology_1':
        xs = odeint(_biology_1_deriv, init_point, ts)
    elif model == 'biology_2':
        xs = odeint(_biology_2_deriv, init_point, ts)
    elif model == 'laub_loomis':
        xs = odeint(_laub_loomis_deriv, init_point, ts)
    elif model == 'coupled_osc_5d':
        xs = odeint(coupled_osc_5d, init_point, ts)
    elif model == 'coupled_osc_10d':
        xs = odeint(coupled_osc_10d, init_point, ts)
    elif model == 'coupled_osc_15d':
        xs = odeint(coupled_osc_15d, init_point, ts)
    elif model == 'coupled_osc_20d':
        xs = odeint(coupled_osc_20d, init_point, ts)
    elif model == 'coupled_osc_25d':
        xs = odeint(coupled_osc_25d, init_point, ts)
    elif model == 'coupled_osc_30d':
        xs = odeint(coupled_osc_30d, init_point, ts)
    else:
        raise ValueError('Simulate eigen: invalid model name!')
    return xs


def _simulate(horizon, model, start_points):
    return list(_simulate_one_run(horizon, model, p) for p in start_points)


def _save_simu_traj(simu_traj, filename):
    c = 0
    with open(filename, 'w') as simu_op:
        for traj in simu_traj:
            c+=1
            print(c)
            for row in traj:
                for elem in row:
                    simu_op.write(str(elem) + ' ')
                simu_op.write('\n')


def run_simulate(time_horizon, model, bounds):
    from convex_set.hyperbox import HyperBox
    import random
    init_set = HyperBox(bounds)
    vertices = init_set.get_vertices()
    n = 50

    bounds = init_set.bounds.T

    simu_points = []
    for i in range(n):
        p = tuple(random.uniform(*b) for b in bounds)
        simu_points.append(p)

    if len(vertices) > 100:
        vertices = vertices[:100]
    simu_points.extend(list(vertices))

    simu_traj = _simulate(time_horizon, model, simu_points)
    return simu_traj


def main(horizon):
    ts = np.linspace(0, 2, 1500)

    import time
    # start_time = time.time()
    # for i in range(1000000):
    #     xs = odeint(brusselator_deriv, [2, 0.28], ts)
    # print(time.time() - start_time)
    # exit()
    xs = odeint(_biology_2_deriv, [1, 1, 1, 1, 1, 1, 1, 1, 1], ts)
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