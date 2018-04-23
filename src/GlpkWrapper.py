from cvxopt import glpk, mul, spmatrix, matrix


class GlpkWrapper:
    def __init__(self, sys_dim):
        self.A = spmatrix([], [], [], (0, sys_dim), 'd')
        self.b = matrix(0.0, (0, 1))
        glpk.options = dict(msg_lev='GLP_MSG_OFF')

    def lp(self, c, G, h):
        status, x, _, _ = glpk.lp(c, G, h, self.A, self.b)
        return sum(mul(-c, x))