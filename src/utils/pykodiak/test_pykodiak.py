'''
Test / demo code for pykodiak interface.

Stanley Bak
Oct 2018
'''

import time

from sympy.parsing.sympy_parser import parse_expr

from pykodiak_interface import Kodiak


def test1():
    'main demo code'

    derivatives = ["x1", "(1-x0**2)*x1-x0"]

    # create sympy expressions from strings
    sympy_ders = [parse_expr(d) for d in derivatives]

    # add variables, in order, to kodiak
    Kodiak.add_variable('x0')
    Kodiak.add_variable('x1')

    # convert sympy expressions into kodiak expressions
    kodiak_ders = [Kodiak.sympy_to_kodiak(d) for d in sympy_ders]

    jac_mat = [[0., 1.], [-7.44, -0.96]]
    bias = 9.016
    bounds = [[1.25, 1.55], [2.28, 2.32]]

    start = time.time()

    iterations = 1000
    for _ in range(iterations):
        lb, ub, _, _ = Kodiak.minmax(kodiak_ders[1], jac_mat[1], bias, bounds)

    print("runtime for {} iterations: {} sec".format(iterations, round(time.time() - start, 3)))
    print("Kodiak computed enclosure: [{}, {}]".format(lb, ub))


# def test2():
#     derivatives = ["(1-x0**2)*x1-x0 + (x2 - x0)"]
#     sympy_ders = [parse_expr(d) for d in derivatives]
#
#     Kodiak.add_variable('x0')
#     Kodiak.add_variable('x1')
#     Kodiak.add_variable('x2')
#
#     kodiak_ders = [Kodiak.sympy_to_kodiak(d) for d in sympy_ders]
#
#     jac_mat = [-8.44, -0.96, 1.]
#     bias = 9.016
#     bounds = [[1.24999, 1.55001], [2.2799899999999997, 2.32001], [1.24999, 1.55001]]
#
#     lb, ub, ulb, lub = Kodiak.minmax(kodiak_ders[0], jac_mat, bias, bounds)
#     print("Kodiak computed enclosure: [{}, {}, {}, {}]".format(lb, ub, ulb, lub))


def main():
    test1()
    # test2()

if __name__ == "__main__":
    main()
