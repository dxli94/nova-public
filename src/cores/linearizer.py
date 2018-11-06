import numpy as np

from utils.pykodiak.pykodiak_interface import Kodiak
from utils.timerutil import Timers


def get_generator_matrix(dim):
    rv = np.zeros(shape=(2*dim, dim))

    for i, row in enumerate(rv):
        row[i // 2] = (-1)**i

    return rv


class Linearizer:
    def __init__(self, dim, nonlin_dyn, is_linear):
        self.dim = dim
        self.target_dyn = nonlin_dyn
        self.is_linear = is_linear
        self.is_scaled = False

        self.generator_matrix = get_generator_matrix(dim)

    def gen_abs_dynamics(self, abs_domain_bounds):
        """
        Core function of Linearizer. This linearizes a nonlinear dynamics
        with a linear dynamics plus an interval representing the linearization error.

        We apply Jacobian linearization at the center of the domain. The lienarization
        error is computed using Kodiak (https://github.com/nasa/Kodiak), which provides
        a rigorous guarantee on the error bounds.
        """
        # Timers.tic('total')

        # Somehow this is faster than taking mean/average directly
        abs_domain_centre = np.sum(abs_domain_bounds, axis=0) / 2

        abs_domain_lower_bounds = abs_domain_bounds[0]
        abs_domain_upper_bounds = abs_domain_bounds[1]

        # apply jacobian linearization
        matrix_A, b = self.jacobian_linearize(abs_domain_centre, self.target_dyn)

        u_bounds = []
        for i in range(self.dim):
            # if some dynamics is linear, we don't need to linearize it.
            if not self.target_dyn.is_scaled and self.is_linear[i]:
                u_min = u_max = 0
            else:
                coeff = matrix_A[i]
                bias = b[i]

                bounds = [[abs_domain_lower_bounds[i], abs_domain_upper_bounds[i]] for i in range(self.dim)]

                Timers.tic('minmax')
                kodiak_res = Kodiak.minmax(self.target_dyn.kodiak_ders[i], coeff, bias, bounds)
                u_min, u_max = -kodiak_res[0], kodiak_res[1]
                Timers.toc('minmax')

            u_bounds.extend([u_max, u_min])

        col_vec = np.array(u_bounds)

        poly_U = (self.generator_matrix, col_vec.reshape(len(col_vec), 1))

        return matrix_A, poly_U, b

    @staticmethod
    def jacobian_linearize(abs_center, non_linear_dyn):
        """
        Function for Jacobian linearization.
        """
        # f(x, y) = J(x - x0, y - y0) * [x - x0, y - y0].T + g(x0, y0)
        mat_a = np.array(non_linear_dyn.eval_jacobian(abs_center)).astype(np.float64)
        b = non_linear_dyn.eval(abs_center) - mat_a.dot(abs_center)

        return mat_a, b
