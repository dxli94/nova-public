import numpy as np
from scipy.optimize import linprog


class SupportFunctionProvider:
    """
    For now, assume the convex set is a polytope defined by Ax-b >= 0.
    """
    def __init__(self, poly):
        self.conv_set = poly

    def compute_support_function(self, direction):
        if self.conv_set.is_empty():
            raise RuntimeError("\n Compute Support Function called for an Empty Set.")
        elif self.conv_set.is_universe():
            raise RuntimeError("\n Cannot Compute Support Function of a Universe Polytope.\n")
        else:
            # Note that these bounds have to be set to (None, None) to allow (-inf, +inf),
            # otherwise (0, +inf) by default.
            # Besides, Scipy only deals with min(). Here we need max(). max(f(x)) = -min(-f(x))
            sf = linprog(c=-direction,
                         A_ub=self.conv_set.A, b_ub=self.conv_set.b,
                         bounds=(None, None))
            if sf.success:
                return sf.x, -sf.fun
            else:
                raise RuntimeError(sf.message)

    def compute_max_norm(self):
        raise NotImplementedError


if __name__ == '__main__':
    from Polyhedron import Polyhedron
    constr = [[1, 1, 2],
              [-1, 0, 0],
              [0, -1, 0]
              ]
    poly = Polyhedron(constr)
    direction = np.array([1, 0])
    sfp = SupportFunctionProvider(poly)
    assert float.__eq__(sfp.compute_support_function(direction)[1], 2)

    constr = [[-1, 0, 0],  # -x1 <= 0
              [1, 0, 2],   # x1 <= 2
              [0, -1, 0],  # -x2 <= 0
              [0, 1, 1]  # x2 <= 1
              ]
    poly = Polyhedron(constr)
    direction = np.array([2, 0])
    sfp = SupportFunctionProvider(poly)
    assert float.__eq__(sfp.compute_support_function(direction)[1], 4)
