import numpy as np
from ConvexSet.polyhedron import Polyhedron

coeff_matrix = np.array([[1, 1], [-1, 0], [0, -1]])
col_vec = np.array([[2], [0], [0]])
poly = Polyhedron(coeff_matrix, col_vec)


def test_add_constraint():
    poly.add_constraint(np.array([1, -1]), np.array([0]))
    test_2 = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]
    np.testing.assert_equal(set(poly.vertices), set(test_2))


def test_get_inequalities():
    np.testing.assert_equal(poly.get_inequalities().all(), np.array([[1, 1, 2], [-1, 0, 0], [0, -1, 0]]).all())


def test_compute_max_norm():
    np.testing.assert_almost_equal(poly.compute_max_norm(), 2)


def test_compute_support_function():
    directions = np.array([[1, 0], [2, 0],
                           [-1, 0], [-2, 0],
                           [1, 1]
                           ])
    sf = [poly.compute_support_function(l) for l in directions]
    correct_sf = [2, 4, 0, 0, 2]
    np.testing.assert_almost_equal(sf, correct_sf)
