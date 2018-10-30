import numpy as np
import itertools


def hyperbox_contain(sf_1, sf_2):
    for elem in zip(sf_1, sf_2):
        abs_domain_sf = elem[0][0]
        image_sf = elem[1]

        # print(abs_domain_sf, image_sf)
        # exit()

        if abs_domain_sf < image_sf:
            return False
    return True


def contains(bd1, bd2):
    lb1, ub1 = bd1
    lb2, ub2 = bd2

    return np.less(lb1, lb2).all() and np.less(ub2, ub1).all()


class HyperBox:
    def __init__(self, arg, opt=0):
        self.vertices = []

        if opt == 0:
            # constructor with vertices
            self.dim = len(arg[0])
            if len(arg) > 0:
                lb = [1e9] * self.dim
                ub = [-1e9] * self.dim
            else:
                raise RuntimeError('Emtpy vertex set.')

            # print(vertices)
            for v in arg:
                for idx in range(len(v)):
                    if v[idx] < lb[idx]:
                        lb[idx] = v[idx]
                    if v[idx] > ub[idx]:
                        ub[idx] = v[idx]

            self.bounds = np.array([lb, ub])
            self.lb = np.array(lb)
            self.ub = np.array(ub)
            self.update_vertices()

        if opt == 1:
            # constructor with bounds
            lb = arg[0]
            ub = arg[1]

            self.bounds = np.array([lb, ub])
            self.lb = np.array(lb)
            self.ub = np.array(ub)
            self.update_vertices()

    def __str__(self):
        str_repr = ''
        for idx, elem in zip(range(len(self.bounds)), self.bounds):
            name = 'lower bounds' if idx == 0 else 'upper bounds'
            str_repr += '{} : '.format(name) + str(elem) + '\n'

        return str_repr

    def bloat(self, epsilon):
        self.bounds = np.array([np.subtract(self.bounds[0], epsilon, dtype=float), np.add(self.bounds[1], epsilon)])
        self.update_vertices()

    def update_vertices(self):
        bounds = self.bounds.T
        self.vertices = list(itertools.product(*bounds))

    def get_vertices(self):
        return self.vertices

    @staticmethod
    def get_vertices_from_constr(coeff, col):
        """Input is assumed to be a hyperbox. Otherwise behaviour undefined."""
        dim = coeff.shape[1]
        bounds = np.zeros(shape=(dim, 2))

        for row_idx, row in enumerate(coeff):
            for col_idx, cell in enumerate(row):
                # print(col_idx, cell)

                if cell < 0:
                    bounds[col_idx][1] = col[row_idx] / cell
                    # lower_bounds[col_idx] = col[row_idx] / cell
                elif cell > 0:
                    bounds[col_idx][0] = col[row_idx] / cell
                    # upper_bounds[col_idx] = col[row_idx] / cell

        rv = list(itertools.product(*bounds))
        return rv


if __name__ == '__main__':
    bd1 = [[0, 1], [2, 3]]
    bd2 = [[1, 2], [1, 2]]

    vertices = [[2, 2], [2, -3], [-2, -3], [-2, 2]]
    hb = HyperBox(vertices)

    print(hb.bounds)

    hb.bloat(0.01)

    print(hb.bounds)

    # lb1, ub1 = bd1
    # lb2, ub2 = bd2
    #
    # print(np.less(lb1, lb2).all())
    # print(np.less(ub2, ub1).all())
    # print(hyperbox_contain_by_bounds(bd1, bd2))