import numpy as np
from ppl import Variable, Constraint_System, C_Polyhedron

normalised_factor = 1e9


def normalised(n):
    return n * normalised_factor


class HyperBox:
    def __init__(self, vertices):
        self.dim = len(vertices[0])
        if len(vertices) > 0:
            lower_bounds = [1e9] * self.dim
            upper_bounds = [-1e9] * self.dim
        else:
            raise RuntimeError('Emtpy vertex set.')

        for v in vertices:
            for idx in range(len(v)):
                if v[idx] < lower_bounds[idx]:
                    lower_bounds[idx] = v[idx]
                if v[idx] > upper_bounds[idx]:
                    upper_bounds[idx] = v[idx]

        self.bounds = np.array([[lower_bounds[idx], upper_bounds[idx]] for idx in range(len(upper_bounds))])

    def __str__(self):
        str_repr = ''
        for idx, elem in zip(range(len(self.bounds)), self.bounds):
            str_repr += 'dimension ' + str(idx) + ': ' + str(elem) + '\n'

        return str_repr

    def bloat(self, epsilon):
        self.bounds = np.array([[bd[0] - epsilon, bd[1] + epsilon] for bd in self.bounds])

    def to_ppl(self):
        cs = Constraint_System()
        variables = [Variable(idx) for idx in range(self.dim)]

        for idx, bd in zip(range(len(self.bounds)), self.bounds):
            cs.insert(normalised(variables[idx]) >= normalised(bd[0]))
            cs.insert(normalised(variables[idx]) <= normalised(bd[1]))

        return C_Polyhedron(cs)

    def to_constraints(self):
        """ Todo:
            To be generalised to high dimensions
        """
        coeff_matrix = np.array([[-1, 0],
                               [1, 0],
                               [0, -1],
                               [0, 1]])
        col_vec = np.array([-self.bounds[0][0],
                            self.bounds[0][1],
                            -self.bounds[1][0],
                            self.bounds[1][1]]
                           )
        return coeff_matrix, col_vec.reshape(len(col_vec), 1)
