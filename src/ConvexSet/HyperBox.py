import numpy as np
# from ppl import Variable, Constraint_System, C_Polyhedron
from itertools import combinations


def normalised(n):
    normalised_factor = 1e9
    return n * normalised_factor


def hyperbox_contain(sf_1, sf_2):
    for elem in zip(sf_1, sf_2):
        abs_domain_sf = elem[0][0]
        image_sf = elem[1]

        # print(abs_domain_sf, image_sf)
        # exit()

        if abs_domain_sf < image_sf:
            return False
    return True


def hyperbox_contain_by_bounds(bd1, bd2):
    lb1, ub1 = bd1
    lb2, ub2 = bd2

    return np.less(lb1, lb2).all() and np.less(ub2, ub1).all()

class HyperBox:
    def __init__(self, arg, opt=0):
        if opt == 0:
            # constructor with vertices
            self.dim = len(arg[0])
            if len(arg) > 0:
                lower_bounds = [1e9] * self.dim
                upper_bounds = [-1e9] * self.dim
            else:
                raise RuntimeError('Emtpy vertex set.')


            # print(vertices)
            for v in arg:
                for idx in range(len(v)):
                    if v[idx] < lower_bounds[idx]:
                        lower_bounds[idx] = v[idx]
                    if v[idx] > upper_bounds[idx]:
                        upper_bounds[idx] = v[idx]

            self.bounds = np.array([[lower_bounds[idx], upper_bounds[idx]] for idx in range(len(upper_bounds))])
            self.vertices = list(combinations(self.bounds, self.dim))

        if opt == 1:
            # constructor with bounds
            lb = arg[0]
            ub = arg[1]

            if lb.isinstance(np.ndarray):
                assert ub.shape[0] == lb.shape[0]
                self.dim = lb.shape[0]
            elif lb.isinstance(list):
                assert len(ub) == len(lb)
                self.dim = len(lb)
            else:
                raise ValueError("unsupported bound type, require list or numpy.ndarray, found {}.".format(type(lb)))
            self.bounds = np.array([lb, ub])
            self.vertices = list(combinations(self.bounds, self.dim))

    def __str__(self):
        str_repr = ''
        for idx, elem in zip(range(len(self.bounds)), self.bounds):
            str_repr += 'dimension ' + str(idx) + ': ' + str(elem) + '\n'

        return str_repr

    def bloat(self, epsilon):
        self.bounds = np.array([[bd[0] - epsilon, bd[1] + epsilon] for bd in self.bounds])

    # def to_ppl(self):
    #     cs = Constraint_System()
    #     variables = [Variable(idx) for idx in range(self.dim)]
    #
    #     for idx, bd in zip(range(len(self.bounds)), self.bounds):
    #         cs.insert(normalised(variables[idx]) >= normalised(bd[0]))
    #         cs.insert(normalised(variables[idx]) <= normalised(bd[1]))
    #
    #     return C_Polyhedron(cs)

    def to_constraints(self):
        """ Todo:
            To be generalised to high dimensions
        """
        coeff_matrix = np.array([[1, 0],
                               [-1, 0],
                               [0, 1],
                               [0, -1]])
        col_vec = np.array([self.bounds[0][1],
                            -self.bounds[0][0],
                            self.bounds[1][1],
                            -self.bounds[1][0]]
                           )
        return coeff_matrix, col_vec.reshape(len(col_vec), 1)


if __name__ == '__main__':
    bd1 = [[0, 1], [2, 3]]
    bd2 = [[1, 2], [1, 2]]

    lb1, ub1 = bd1
    lb2, ub2 = bd2

    print(np.less(lb1, lb2).all())
    print(np.less(ub2, ub1).all())
    print(hyperbox_contain_by_bounds(bd1, bd2))