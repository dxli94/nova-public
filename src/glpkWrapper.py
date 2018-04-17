from cvxopt import glpk
import cvxopt as cvx
import numpy as np


class glpkWrapper:
    def __init__(self, sys_dim):
        self.A = cvx.spmatrix([], [], [], (0, sys_dim), 'd')
        self.b = cvx.matrix(0.0, (0, 1))
        glpk.options = dict(msg_lev='GLP_MSG_OFF')

    def lp(self, c, G, h):
        status, x, _, _ = glpk.lp(c, G, h, self.A, self.b)
        return np.array(-c).reshape(1, len(c)).dot(np.array(x))[0][0]

if __name__ == '__main__':
    directions = np.array([[1, 0], [2, 0],
                           [-1, 0], [-2, 0],
                           [1, 1]
                           ])
    G = cvx.matrix(np.array([[1, 1], [-1, 0], [0, -1]]), tc='d')
    col_vec = cvx.matrix(np.array([[2], [0], [0]]), tc='d')
    c = cvx.matrix(directions[4], tc='d')

    lp = glpkWrapper(2)
    # poly = Polyhedron(A, col_vec)

    print(lp.lp(c=c, G=G, h=col_vec))