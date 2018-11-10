import numpy as np
import itertools


def hyperbox_contain(sf_1, sf_2):
    for elem in zip(sf_1, sf_2):
        abs_domain_sf = elem[0][0]
        image_sf = elem[1]

        if abs_domain_sf < image_sf:
            return False
    return True


def contains(bd1, bd2):
    lb1, ub1 = bd1
    lb2, ub2 = bd2

    return np.less(lb1, lb2).all() and np.less(ub2, ub1).all()


class HyperBox:
    def __init__(self, arg, opt=0):
        """
        Create a hyperbox instance.

        Opt=0: create a hyperbox from vertices. This is useful in low-dimensional scenarios.
        Opt=1: create a hyperbox from bounds.
        """
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

        if opt == 1:
            # constructor with bounds
            lb = arg[0]
            ub = arg[1]

            self.bounds = np.array([lb, ub])
            self.lb = np.array(lb)
            self.ub = np.array(ub)

    def __str__(self):
        str_repr = ''
        for idx, elem in zip(range(len(self.bounds)), self.bounds):
            name = 'lower bounds' if idx == 0 else 'upper bounds'
            str_repr += '{} : '.format(name) + str(elem) + '\n'

        return str_repr

    def bloat(self, epsilon):
        """
        Push bounds on each dimension "outwards" for epsilon distance.
        """
        self.bounds = np.array([np.subtract(self.bounds[0], epsilon, dtype=float), np.add(self.bounds[1], epsilon)])

    def update_vertices(self):
        """
        Vertices of a hyperbox is the cartesian product of bounds on each dimension.
        """
        bounds = self.bounds.T
        self.vertices = list(itertools.product(*bounds))

    def get_vertices(self):
        """
        Lazy evaluation: update vertices only when vertices is required.
        """
        self.update_vertices()
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

    @staticmethod
    def get_bounds_from_constr(coeff, col):
        """
        Be cautious. For efficiency reason,
        this makes assumption on the ordering of constraints.

        Requires a refactoring later on.
        """
        sum_coeff = np.sum(coeff, axis=1)
        col = np.reshape(col, (1, -1))[0]

        pos_clip = np.clip(sum_coeff, 0, np.inf)
        pos_index = pos_clip.nonzero()
        neg_clip = np.clip(sum_coeff, -np.inf, 0)
        neg_index = neg_clip.nonzero()

        lb_with_zero = np.multiply(neg_clip, col)
        ub_with_zero = np.multiply(pos_clip, col)

        lb = lb_with_zero[neg_index]
        ub = ub_with_zero[pos_index]

        return lb, ub

if __name__ == '__main__':
    coeff = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    col = [[2], [-1], [3], [-2]]

    print(HyperBox.get_bounds_from_constr(coeff, col))