import numpy as np


class Verifier:
    def __init__(self, prop):
        # a tuple (A, b), such that Ax <= b
        self.p = prop
        self.col_idx = None
        self.sf_mat = None

    def set_support_func_mat(self, directions, mat):
        self.sf_mat = np.array(mat)
        p_coeff, p_col = self.p

        for idx, l in enumerate(directions):
            if (-p_coeff == l).all():
                self.col_idx = idx

    def verify_prop(self):
        assert self.p
        assert self.sf_mat is not None

        sf_col = self.sf_mat[:, self.col_idx]

        if (-sf_col <= self.p[1]).any():
            return False
        else:
            return True

if __name__ == '__main__':
    p = np.array([0, -1]), np.array([-1.9])
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    sf_mat = np.array([[2, 2, 1.89999, 2]])

    v = Verifier(p)
    v.set_support_func_mat(directions, sf_mat)
    print(v.verify_prop())