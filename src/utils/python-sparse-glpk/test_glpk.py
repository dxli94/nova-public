'''
Stanley Bak
Unit tests for python_glpk_sparse
May 2018
'''

import cvxopt
import numpy as np

from scipy.sparse import csr_matrix, csc_matrix

from python_sparse_glpk import LpInstance

def test_cpp():
    'runs the c++ test() function for the glpk interface'

    assert LpInstance.test() == 0

def compare_opt(a_ub, b_ub, direction):
    'compare cvx opt versus our glpk interface (both csc and csr)'

    # make sure we're using floats not ints
    a_ub = [[float(x) for x in row] for row in a_ub]
    b_ub = [float(x) for x in b_ub]
    c = [float(x) for x in direction]

    num_vars = len(a_ub[0])

    # solve it with cvxopt
    options = {'show_progress': False}
    sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(a_ub).T, cvxopt.matrix(b_ub), options=options)

    #if sol['status'] == 'primal infeasible':
    #    res_cvxopt = None

    if sol['status'] != 'optimal':
        raise RuntimeError("cvxopt LP failed: {}".format(sol['status']))

    res_cvxopt = [float(n) for n in sol['x']]

    #print "cvxopt value = {}, result = {}".format(np.dot(res_cvxopt, c), repr(res_cvxopt))

    # solve it with the glpk <-> c++ interface
    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    res_glpk = lp.minimize(direction)

    #print "glpk interface value = {}, result = {}".format(np.dot(res_glpk, c), repr(res_glpk))

    assert num_vars == len(res_cvxopt)
    assert abs(np.dot(res_glpk, c) - np.dot(res_cvxopt, c)) < 1e-6

    # try again with csc constraints
    lp = LpInstance()

    a_csc = csc_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csc(a_csc)

    res_glpk = lp.minimize(direction)

    assert num_vars == len(res_cvxopt)
    assert abs(np.dot(res_glpk, c) - np.dot(res_cvxopt, c)) < 1e-6

def test_simple():
    '''test consistency with cvxopt on a simple problem'''

    # max 0.6x + 0.5y st.
    # x + 2y <= 1
    # 3x + y <= 2

    a_ub = [[1, 2], [3, 1]]
    b_ub = [1, 2]
    c = [-0.6, -0.5]

    compare_opt(a_ub, b_ub, c)

def test_underconstrained():
    'test an underconstrained case (fails for cvxopt)'

    a_ub = [[1.0, 0.0], [-1.0, 0.0]]
    b_ub = [1.0, 1.0]
    direction = [1.0, 0.0]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    res_glpk = lp.minimize(direction)

    assert abs(res_glpk[0] - (-1) < 1e-6)

def test_tricky():
    '''test consistency with cvxopt on a tricky problem (scipy.linprog fails)'''

    a_ub = [[-1.0, 0.0, 0.0, -2.1954134149515525e-08, 1.0000000097476742, 0.0],
            [1.0, -0.0, -0.0, 2.1954134149515525e-08, -1.0000000097476742, -0.0],
            [0.0, -1.0, 0.0, -1.000000006962809, 2.5063524589086228e-08, 0.0],
            [-0.0, 1.0, -0.0, 1.000000006962809, -2.5063524589086228e-08, -0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 1.0000000000000009],
            [-0.0, -0.0, 1.0, -0.0, -0.0, -1.0000000000000009],
            [0., 0., 0., 1.0, 0.0, 0.0],
            [0., 0., 0., -1.0, 0.0, 0.0],
            [0., 0., 0., 0.0, 1.0, 0.0],
            [0., 0., 0., 0.0, -1.0, 0.0],
            [0., 0., 0., 0.0, 0.0, 1.0],
            [0., 0., 0., 0.0, 0.0, -1.0]]

    b_ub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, -0.0]

    num_vars = len(a_ub[0])
    # c = [1.0 if i % 2 == 0 else 0.0 for i in xrange(num_vars)]
    c = [1.0 if i % 2 == 0 else 0.0 for i in range(num_vars)]


    compare_opt(a_ub, b_ub, c)
