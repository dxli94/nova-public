import sympy
from functools import reduce

import numpy as np
from copy import deepcopy
from sympy.parsing.sympy_parser import parse_expr

from utils.pykodiak.pykodiak_interface import Kodiak
from utils.timerutil import Timers


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
        self.dyn_str = list(args)
        self.dyn_sp = [sympy.sympify(arg) for arg in args]
        num_free_vars = len(reduce((lambda x, y: x.union(y)), [d.free_symbols for d in self.dyn_sp]))
        assert len(self.vars_dict) >= num_free_vars, \
            "inconsistent number of variables declared ({}) with used in dynamics ({})" \
                .format(len(self.vars_dict), num_free_vars)

        self.state_vars = tuple(sympy.symbols(str(self.vars_dict[key])) for key in self.vars_dict)
        self.lamdafied_dynamics = sympy.lambdify(self.state_vars, self.dyn_sp)
        self.jacobian_mat = sympy.lambdify(self.state_vars, sympy.Matrix(self.dyn_sp).jacobian(self.state_vars))

        self.str_rep = args

        self.scale_func = None, None
        self.scale_func_jac = None
        self.is_scaled = False

        # attrs for Kodiak
        self.sympy_ders = [parse_expr(d) for d in args]
        self._sympy_ders = deepcopy(self.sympy_ders)
        self._sympy_ders_scaled_template = self.make_sympy_ders_scaled_template(args)

        self.kodiak_ders = [Kodiak.sympy_to_kodiak(d) for d in self.sympy_ders]
        self._kodiak_ders_copy = deepcopy(self.kodiak_ders)

    def __str__(self):
        return str(self.dyn_sp)

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

    def make_sympy_ders_scaled_template(self, orig_dynamics):
        """
        Make a template for the scaled function.

        Sympy.parse_expr() is very expensive. I avoid calling it upon each newly scaled dynamics
        by creating such a template with parameter symbols a0, a1, ... an-1, b.
        Then each time, we just need to replace these parameters with proper values, thus avoiding
        constructing the parsing tree.
        """
        params_list = []

        linear_term = ''
        for idx in range(len(orig_dynamics)):
            param = 'a{}'.format(idx)
            linear_term += '{}*x{}+'.format(param, idx)
            params_list.append(param)

        param = 'b'
        params_list.append(param)
        scaling_func_str = '{}{}'.format(linear_term, param)

        scaled_dynamics_str = []
        for dyn in self.dyn_str:
            scaled_dynamics_str.append('({})*({})'.format(scaling_func_str, dyn))

        return [sympy.lambdify(params_list, parse_expr(d)) for d in scaled_dynamics_str]

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

        Timers.tic('subs(subs_dict)')
        # todo can we further optimize??
        self.sympy_ders = [d(*a, b) for d in self._sympy_ders_scaled_template]
        Timers.toc('subs(subs_dict)')

        Timers.tic('Kodiak.sympy_to_kodiak(d)')
        self.kodiak_ders = [Kodiak.sympy_to_kodiak(d) for d in self.sympy_ders]
        Timers.toc('Kodiak.sympy_to_kodiak(d)')

    def reset_dynamic(self):
        """
        Reset to the original dynamics.
        """
        self.is_scaled = False

        self.sympy_ders = deepcopy(self._sympy_ders)
        self.kodiak_ders = deepcopy(self._kodiak_ders_copy)