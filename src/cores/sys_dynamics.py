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

        self.scale_func = None, None
        self.scale_func_jac = None
        self.is_scaled = False

    def eval(self, vals):
        """
        Evaluate the dynamics given a specific point (vals).

        If the dynamic is unscaled, return the result from the lamdafication.
        Otherwise, multiply the lamdafication result by the distance.

        eval(dyn, vals) = eval(dist_func, vals) * eval(orig_func, vals)
        """

        orig_eval = self.lamdafied_dynamics(*vals)
        if self.is_scaled:
            dist_val = self.scale_func[0].dot(vals) + self.scale_func[1]
            return np.multiply(dist_val, orig_eval)
        else:
            return orig_eval

    def eval_jacobian(self, vals):
        """
        Evaluate the Jacobian matrix given a specific point (vals).

        If the dynamic is unscaled, return the result from the lamdafication.
        Otherwise, use the product rule of the partial derivative to compute
        the Jacobian of the scaled dynamics. The rule is as follows.

        h(x) = f(x)g(x)
        h'(x) = f'(x)g(x) + f(x)g'(x)

        Here, h(x) is the (scaled) dynamics, f(x) is the distance function, g(x) is the original dynamics.
        """

        if self.is_scaled:  # the dynamics is scaled
            f = self.scale_func[0].dot(vals) + self.scale_func[1]
            f_deriv = self.scale_func_jac
            g = self.lamdafied_dynamics(*vals)
            g_deriv = self.jacobian_mat(*vals)

            # product rule of computing partial derivatives
            rv = np.einsum('i,ij->ij', g, f_deriv) + np.einsum('i,ij->ij', f, g_deriv)
            return rv
        else:  # the dynamics is unscaled
            return self.jacobian_mat(*vals)

    def __str__(self):
        return str(self.sp_dynamics)

    def apply_dynamic_scaling(self, a, b):
        """
        Apply dynamic-scaling.

        The scaling function is in the form ax + b, which represents the distance to a hyper-plane.
        We multiply the distance function to each dynamic to acquire the so-called "scaled dynamics".

        The Jacobian of the scaled dynamics is [a, a, ..., a], (n rows, n = dimension).
        """
        elem1 = np.array([a for _ in range(len(a))])
        elem2 = np.repeat(b, len(a))
        self.scale_func = elem1, elem2
        self.scale_func_jac = elem1
        self.is_scaled = True

    def reset_dynamic(self):
        self.is_scaled = False