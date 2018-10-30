'''
Test / demo code for pykodiak interface.

Stanley Bak
Oct 2018
'''

import time

from sympy.parsing.sympy_parser import parse_expr

from pykodiak_interface import Kodiak

def main():
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
        lb, ub = Kodiak.minmax(kodiak_ders[1], jac_mat[1], bias, bounds)

    print("runtime for {} iterations: {} sec".format(iterations, round(time.time() - start, 3)))
    print("Kodiak computed enclosure: [{}, {}]".format(lb, ub))

if __name__ == "__main__":
    main()
