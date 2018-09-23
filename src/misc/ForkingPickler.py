import dill
import copyreg
import multiprocessing
import cloudpickle
import pickle
import sympy


class ForkingPickler(pickle.Pickler):
    '''Pickler subclass used by multiprocessing.'''
    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    load = cloudpickle.load
    loads = cloudpickle.loads
    dump = cloudpickle.dump
    dumps = cloudpickle.dumps

    def __init__(self, *args):
        super().__init__(*args)
        self.dispatch_table = self._copyreg_dispatch_table.copy()
        self.dispatch_table.update(self._extra_reducers)

    @classmethod
    def register(cls, type, reduce):
        '''Register a reduce function for a type.'''
        cls._extra_reducers[type] = reduce

ctx = multiprocessing.get_context()
ctx.reducer = ForkingPickler
multiprocessing.connection._ForkingPickler = ForkingPickler

pool = ctx.Pool()
# res1 = pool.apply_async(jacobian_lambda, [1])
# res1.get()

x = sympy.symbols('x')
expr = sympy.sympify('x*x')
jacobian_lambda = sympy.lambdify(x, sympy.Matrix([expr]).jacobian([x]))

# pool = Pool()
res1 = pool.apply_async(jacobian_lambda, [1])
res2 = pool.apply_async(jacobian_lambda, [2])
print([res1.get(), res2.get()])