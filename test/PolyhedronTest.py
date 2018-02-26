import unittest

import numpy as np
from src.ConvexSet.Polyhedron import Polyhedron


class TestPolyhedronMethods(unittest.TestCase):
    coeff_matrix = np.array([[1, 1], [-1, 0], [0, -1]])
    col_vec = np.array([[2], [0], [0]])
    poly = Polyhedron(coeff_matrix, col_vec)

    def test_add_constraint(self):
        self.poly.add_constraint(np.array([1, -1]), np.array([0]))
        test_2 = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]
        self.assertEqual(set(self.poly.vertices), set(test_2))

    def test_get_inequalities(self):
        self.assertEqual(self.poly.get_inequalities().all(), np.array([[1, 1, 2], [-1, 0, 0], [0, -1, 0]]).all())

    def test_compute_max_norm(self):
        self.assertEqual(self.poly.compute_max_norm(), 2)

    def test_compute_support_function(self):
        directions = np.array([[1, 0], [2, 0],
                      [-1, 0], [-2, 0],
                      [1, 1]
                      ])
        sf = [self.poly.compute_support_function(l) for l in directions]
        correct_sf = [2, 4, 0, 0, 2]
        self.assertEqual(sf, correct_sf)


if __name__ == '__main__':
    unittest.main()
