import sympy
from functools import reduce

import numpy as np


class AffineDynamics:
    def __init__(self, dim, x0_matrix=None, x0_col=None, a_matrix=None, b_matrix=None,
                 u_coeff=None, u_col=None):
        # x' = Ax(t) + Bu(t)
        self.dim = dim
        self.a_matrix = a_matrix

        if b_matrix is not None:
            self.matrix_B = b_matrix
        else:
            self.matrix_B = np.identity(dim)
        self.u_coeff = u_coeff  # u(t) coeff matrix
        self.u_col = u_col  # u(t) col vec

        self.x0_coeff = x0_matrix
        self.x0_col = x0_col

    def get_dim(self):
        return self.dim

    def get_dyn_coeff_matrix_A(self):
        return self.a_matrix

    def get_dyn_matrix_B(self):
        return self.matrix_B

    def get_dyn_coeff_matrix_U(self):
        return self.u_coeff

    def get_dyn_col_vec_U(self):
        return self.u_col

    def get_dyn_init_X0(self):
        return self.x0_coeff, self.x0_col


class GeneralDynamics:
    def __init__(self, id_to_vars, *args):
        self.vars_dict = id_to_vars
        self.sp_dynamics = [sympy.sympify(arg) for arg in args]
        num_free_vars = len(reduce((lambda x, y: x.union(y)), [d.free_symbols for d in self.sp_dynamics]))
        assert len(self.vars_dict) >= num_free_vars, \
            "inconsistent number of variables declared ({}) with used in dynamics ({})" \
                .format(len(self.vars_dict), num_free_vars)

        self.state_vars = tuple(sympy.symbols(str(self.vars_dict[key])) for key in self.vars_dict)
        self.lamdafied_dynamics = sympy.lambdify(self.state_vars, self.sp_dynamics)
        self.jacobian_mat = sympy.lambdify(self.state_vars, sympy.Matrix(self.sp_dynamics).jacobian(self.state_vars))

        self.str_rep = args

    def eval(self, vals):
        return self.lamdafied_dynamics(*vals)

    def eval_jacobian(self, vals):
        return self.jacobian_mat(*vals)

    def __str__(self):
        return str(self.sp_dynamics)

    # def update_dynamics(self, *args):
    #     # todo does not work yet
    #     self.dynamics = [sympy.sympify(arg) for arg in args]
    #     self.lamdafied_dynamics = sympy.lambdify(self.state_vars, self.dynamics)
    #     self.jacobian_mat = sympy.lambdify(self.state_vars, sympy.Matrix(self.dynamics).jacobian(self.state_vars))


if __name__ == '__main__':
    # id_to_vars = {0: 'x0',
    #               1: 'x1',
    #               2: 'x2'}
    # gd = GeneralDynamics(id_to_vars, 'x0/x1', 'x1+x2')
    #
    # for i in range(0, 500000):
    #     if i % 10000 == 0:
    #         print(i)
    #     gd.eval_jacobian([1, 2, 3])
    #
    # pass

    from multiprocessing import Pool
    import dill

    dill.settings['recurse'] = True
    # from pathos.multiprocessing import Pool

    x = sympy.symbols('x')
    expr = sympy.sympify('x*x')

    import time

    def basinhopping_wrapper(expr, x, vals):
        jacobian_lambda = sympy.lambdify(x, sympy.Matrix([expr]).jacobian([x]))
        return jacobian_lambda(*vals)


    # for i in range(10000):
    jacobian_lambda = sympy.lambdify(x, sympy.Matrix([expr]).jacobian([x]))

    pool = Pool()

    start = time.time()
    # for i in range(1000):
    #     res1 = pool.apply_async(basinhopping_wrapper, [expr, x, [1]])
    #     res2 = pool.apply_async(basinhopping_wrapper, [expr, x, [2]])
    #
    #     a = [res1.get(), res2.get()]

    for i in range(1000):
        basinhopping_wrapper(expr, x, [1])
        basinhopping_wrapper(expr, x, [2])

    print(time.time()-start)