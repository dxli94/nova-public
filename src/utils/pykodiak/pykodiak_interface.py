'''
Python interface file for pykodiak.so

Staley Bak
Oct 2018
'''

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

from sympy import Mul, Expr, Add, Pow, Symbol, Number, sin

def get_script_path(filename):
    '''get the path this script, pass in __file__ for the filename'''
    return os.path.dirname(os.path.realpath(filename))

class Kodiak:
    'static class for interfacing with kodiak'

    initialized = False

    @classmethod
    def init(cls):
        'initialize the static members'

        if not cls.initialized:
            cls.initialized = True

            lib_path = os.path.join(get_script_path(__file__), 'pykodiak.so')
            cls.lib = ctypes.CDLL(lib_path)

            _init = cls.lib.init
            _init.restype = None
            _init.argtypes = []

            cls._use_bernstein = cls.lib.useBernstein
            cls._use_bernstein.restype = None
            cls._use_bernstein.argtypes = [ctypes.c_int]

            cls._set_precision = cls.lib.setPrecision
            cls._set_precision.restype = None
            cls._set_precision.argtypes = [ctypes.c_int]

            cls._add_variable = cls.lib.addVariable
            cls._add_variable.restype = None
            cls._add_variable.argtypes = [ctypes.c_char_p]

            cls._lookup_variable = cls.lib.lookupVariable
            cls._lookup_variable.restype = ctypes.c_int
            cls._lookup_variable.argtypes = [ctypes.c_char_p]

            cls._print_expression = cls.lib.printExpression
            cls._print_expression.restype = None
            cls._print_expression.argtypes = [ctypes.c_int]

            cls._make_double = cls.lib.makeDouble
            cls._make_double.restype = ctypes.c_int
            cls._make_double.argtypes = [ctypes.c_double]

            cls._make_mult = cls.lib.makeMult
            cls._make_mult.restype = ctypes.c_int
            cls._make_mult.argtypes = [ctypes.c_int, ctypes.c_int]

            cls._make_add = cls.lib.makeAdd
            cls._make_add.restype = ctypes.c_int
            cls._make_add.argtypes = [ctypes.c_int, ctypes.c_int]

            cls._make_sq = cls.lib.makeSq
            cls._make_sq.restype = ctypes.c_int
            cls._make_sq.argtypes = [ctypes.c_int]

            cls._make_sqrt = cls.lib.makeSqrt
            cls._make_sqrt.restype = ctypes.c_int
            cls._make_sqrt.argtypes = [ctypes.c_int]

            cls._make_intpow = cls.lib.makeIntPow
            cls._make_intpow.restype = ctypes.c_int
            cls._make_intpow.argtypes = [ctypes.c_int, ctypes.c_int]

            cls._make_sin = cls.lib.makeSin
            cls._make_sin.restype = ctypes.c_int
            cls._make_sin.argtypes = [ctypes.c_int]

            cls._make_sin = cls.lib.makeSin
            cls._make_sin.restype = ctypes.c_int
            cls._make_sin.argtypes = [ctypes.c_int]

            cls._minmax = cls.lib.minmax
            cls._minmax.restype = None
            cls._minmax.argtypes = [ctypes.c_int, # nonlinear expression
                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int,
                                    ctypes.c_double, # bias
                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int,
                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            #### initialize with default values ####
            _init()
            cls._use_bernstein(1)
            cls._set_precision(-9) # 10^-9 precision

    @classmethod
    def minmax(cls, nonlinear_exp_index, linear_approx, linear_bias, bounds):
        ''''return the lower and upper bound of the passed-in nonlinear function minus the linear approximation,
        within the passed-in bounds'''

        cls.init()

        if not isinstance(linear_approx, np.ndarray):
            linear_approx = np.array(linear_approx, dtype=float)

        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds, dtype=float)

        rv = np.array([0, 0], dtype=float)

        cls._minmax(nonlinear_exp_index, linear_approx, len(linear_approx), linear_bias,
                    bounds, bounds.shape[0], bounds.shape[1], rv, rv.shape[0])

        return rv[0], rv[1]

    @classmethod
    def use_bernstein(cls, use_bernstein):
        'should we use bernstein polynomials for optimization (false = interval arithmetic), default: True'

        cls.init()
        cls._use_bernstein(1 if use_bernstein else 0)

    @classmethod
    def set_precision(cls, precision):
        'set the prevision for the answer, more negative = more accurage, default: -9 (10^-9)'

        cls.init()
        cls._set_precision(precision)

    @classmethod
    def add_variable(cls, name):
        'add an optimization variable (should be called in order)'

        cls.init()
        cls._add_variable(name.encode('utf-8'))

    @classmethod
    def lookup_variable(cls, name):
        'lookup the expression index for a variable previously-added with add_variable()'

        cls.init()
        return cls._lookup_variable(name.encode('utf-8'))

    @classmethod
    def print_expression(cls, i):
        'print the expression with the given index to stdout'

        cls.init()
        cls._print_expression(i)

    @classmethod
    def make_double(cls, d):
        'make and return an expression index for a double number'

        cls.init()
        return cls._make_double(d)

    @classmethod
    def make_mult(cls, a, b):
        'make and return an expression index for a multiplication of the two passed-in expression indices'

        cls.init()
        return cls._make_mult(a, b)

    @classmethod
    def make_add(cls, a, b):
        'make and return an expression index for an addition of the two passed-in expression indices'

        cls.init()
        return cls._make_add(a, b)

    @classmethod
    def make_sq(cls, a):
        'make and return an expression index for the square of the the passed-in expression index'

        cls.init()
        return cls._make_sq(a)

    @classmethod
    def make_sqrt(cls, a):
        'make and return an expression index for the square of the the passed-in expression index'

        cls.init()
        return cls._make_sqrt(a)

    @classmethod
    def make_intpow(cls, a, intb):
        '''make and return an expression index for passed-in expression index raised to the integer intb
        note: intb is an integer, not an expression index
        '''

        cls.init()
        return cls._make_intpow(a, intb)

    @classmethod
    def make_sin(cls, a):
        'make and return an expression index for the sine of the the passed-in expression index'

        cls.init()
        return cls._make_sin(a)

    @classmethod
    def sympy_to_kodiak(cls, sympy_exp):
        '''convert a sympy expression to Kodiak expression

        this function actually returns an int, which is the expression index in the c++ 'reals' vector that
        represents the (newly-constructed) expression
        '''

        rv = None
        e = sympy_exp

        if not isinstance(e, Expr):
            raise RuntimeError("Expected sympy Expr: " + repr(e))

        if isinstance(e, Symbol):
            rv = cls.lookup_variable(e.name)

            if rv is None:
                raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

        elif isinstance(e, Number):
            rv = cls.make_double(float(e))
        elif isinstance(e, Mul):
            rv = cls.sympy_to_kodiak(e.args[0])

            for arg in e.args[1:]:
                val = cls.sympy_to_kodiak(arg)
                rv = cls.make_mult(rv, val)
        elif isinstance(e, Add):
            rv = cls.sympy_to_kodiak(e.args[0])

            for arg in e.args[1:]:
                val = cls.sympy_to_kodiak(arg)
                rv = cls.make_add(rv, val)
        elif isinstance(e, Pow):
            term = cls.sympy_to_kodiak(e.args[0])
            exponent = e.args[1]

            if isinstance(exponent, Number) and float(exponent) == 2:
                # squared
                rv = cls.make_sq(term)
            elif isinstance(exponent, Number) and float(exponent) == 0.5:
                # sqrt
                rv = cls.make_sqrt(term)
            elif isinstance(exponent, Number) and float(exponent) == int(exponent):
                # intpow
                rv = cls.make_intpow(term, exponent)

            # non-integer powers are not supported in Kodiak
        elif isinstance(e, sin):
            a = cls.sympy_to_kodiak(e.args[0])
            rv = cls.make_sin(a)

        # for all expression supported by Kodiak, see Real.hpp

        if rv is None:
            raise RuntimeError("conversion of '{}' to a Kodiak expression unimplemented (type {})".format(str(e), repr(type(e))))

        return rv
