# x_{0+2*i} = x_{i}
# x_{1+2*i} = y_{i}
# x_{1+2*N} = C
# i = 0, 1, ..., N

var_template = 'x{}'


def get_x_index(i):
    return 2 * i


def get_y_index(i):
    return 2 * i + 1


def get_C_index():
    return 2 * N + 1


def make_C_dyn(N):
    C = var_template.format(get_C_index())
    sum_y = '+'.join([var_template.format(get_y_index(i)) for i in range(N)])

    constant = 0.64 / N

    rv = '{} * ({}) - {} * {}'.format(constant, sum_y, N, C)

    return rv


def make_dyn():
    rv = []
    for i in range(N):
        rv.extend([make_x_dyn(i, N), make_y_dyn(i)])

    rv = '[{}]'.format(','.join(rv))

    return rv


def make_init():
    lb = []
    ub = []

    for i in range(N):
        lb.extend(make_lb_init(i))
        ub.extend(make_ub_init(i))

    return [lb, ub]

N = 3
print(make_dyn())
print(make_init())
#
# print(make_x_dyn(0, 10))
# print(make_y_dyn(2))
# print(make_z_dyn(0))
# print(make_v_dyn(0))
# print(make_u_dyn(0))