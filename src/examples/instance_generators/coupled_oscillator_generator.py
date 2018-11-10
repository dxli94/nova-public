# x_{0+5*i} = x_{i}
# x_{1+5*i} = y_{i}
# x_{2+5*i} = z_{i}
# x_{3+5*i} = v_{i}
# x_{4+5*i} = u_{i}
# i = 0, 1, ..., N

var_template = 'x{}'


def get_x_index(i):
    return 5 * i


def get_y_index(i):
    return 5 * i + 1


def get_z_index(i):
    return 5 * i + 2


def get_v_index(i):
    return 5 * i + 3


def get_u_index(i):
    return 5 * i + 4


def make_x_dyn(i, N):
    u = var_template.format(get_u_index(i))
    x = var_template.format(get_x_index(i))
    sum_v = '+'.join([var_template.format(get_v_index(i)) for i in range(N)])

    ten_divided_by_N = 10.0 / N

    rv = '\"0.1*{}-3*{}+{}*({})\"'.format(u, x, ten_divided_by_N, sum_v)

    return rv


def make_y_dyn(i):
    x = var_template.format(get_x_index(i))
    y = var_template.format(get_y_index(i))

    rv = '\"10*{}-2.2*{}\"'.format(x, y)

    return rv


def make_z_dyn(i):
    y = var_template.format(get_y_index(i))
    z = var_template.format(get_z_index(i))

    rv = '\"10*{}-1.5*{}\"'.format(y, z)

    return rv


def make_v_dyn(i):
    x = var_template.format(get_x_index(i))
    v = var_template.format(get_v_index(i))

    rv = '\"2*{}-20*{}\"'.format(x, v)

    return rv


def make_u_dyn(i):
    y = var_template.format(get_y_index(i))
    z = var_template.format(get_z_index(i))
    u = var_template.format(get_u_index(i))

    rv = '\"-5*{}**2*{}**4*(10*{}-1.5*{})\"'.format(u, z, y, z)

    return rv


def make_lb_init(i):
    return [-0.003 + 0.002 * i,
            0.197 + 0.002 * i,
            0.997 + 0.002 * i,
            -0.003 + 0.002 * i,
            0.497 + 0.002 * i
            ]


def make_ub_init(i):
    return [-0.001 + 0.002 * i,
            0.199 + 0.002 * i,
            0.999 + 0.002 * i,
            -0.001 + 0.002 * i,
            0.499 + 0.002 * i
            ]


def make_dyn():
    rv = []
    for i in range(N):
        rv.extend([make_x_dyn(i, N), make_y_dyn(i), make_z_dyn(i), make_v_dyn(i), make_u_dyn(i)])

    rv = '[{}]'.format(','.join(rv))

    return rv


def make_init():
    lb = []
    ub = []

    for i in range(N):
        lb.extend(make_lb_init(i))
        ub.extend(make_ub_init(i))

    return [lb, ub]

N = 6
print(make_dyn())
print(make_init())
#
# print(make_x_dyn(0, 10))
# print(make_y_dyn(2))
# print(make_z_dyn(0))
# print(make_v_dyn(0))
# print(make_u_dyn(0))