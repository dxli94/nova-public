import unittest

import numpy as np
from src.ConvexSet.TransPoly import TransPoly


class TestTransPolyMethods(unittest.TestCase):
    def test_no_rotation(self):
        dynamics_matrix_B = np.identity(2)

        # U is a square with inf_norm = 2
        dynamics_coeff_matrix_U = np.array([[-1, 0],  # u1 >= 0
                                   [1, 0],  # u1 <= 1
                                   [0, -1],  # u2 >= 0
                                   [0, 1]])  # u2 <= 1
        dynamics_col_vec_U = np.array([[0], [1], [0], [1]])
        transPoly = TransPoly(dynamics_matrix_B, dynamics_coeff_matrix_U, dynamics_col_vec_U)

        directions = [[1, 0], [2, 0],
                      [-1, 0], [-2, 0],
                      [1, 1]
                      ]

        # test compute support functions
        sf = [transPoly.compute_support_function(l) for l in directions]
        correct_sf = [1, 2, 0, 0, 2]
        np.testing.assert_almost_equal(sf, correct_sf)

        max_norm = transPoly.compute_max_norm()
        correct_max_norm = 1  # (1,1) on l: (1,0)
        np.testing.assert_almost_equal(max_norm, correct_max_norm)

    def test_shear(self):
        dynamics_matrix_B = np.array([[1, 1], [0, 1]])

        # U is a square with inf_norm = 2
        dynamics_coeff_matrix_U = np.array([[-1, 0],  # u1 >= 0
                                   [1, 0],  # u1 <= 1
                                   [0, -1],  # u2 >= 0
                                   [0, 1]])  # u2 <= 1
        dynamics_col_vec_U = np.array([[0], [1], [0], [1]])
        transPoly = TransPoly(dynamics_matrix_B, dynamics_coeff_matrix_U, dynamics_col_vec_U)

        # BU a diamond defined by: (0,0), (1,1), (2,1), (1,0)

        directions = np.array([[1, 0],
                      [2, 0],
                      [-1, 0],
                      [-2, 0],
                      [1, 1]
                      ])
        sf = [transPoly.compute_support_function(l) for l in directions]
        correct_sf = [2, 4, 0, 0, 3]
        np.testing.assert_almost_equal(sf, correct_sf)

        max_norm = transPoly.compute_max_norm()
        correct_max_norm = 2  # (2,1) on l: (1,0)
        np.testing.assert_almost_equal(max_norm, correct_max_norm)

    def test_shift(self):

        dynamics_matrix_B = np.array([[1, 0], [0, 1]])

        # U is a square with inf_norm = 2
        dynamics_coeff_matrix_U = np.array([[-1, 0],  # u1 >= 1
                                   [1, 0],  # u1 <= 2
                                   [0, -1],  # u2 >= 1
                                   [0, 1]])  # u2 <= 2
        dynamics_col_vec_U = np.array([[-1], [2], [-1], [2]])
        transPoly = TransPoly(dynamics_matrix_B, dynamics_coeff_matrix_U, dynamics_col_vec_U)

        # BU a diamond defined by: (0,0), (1,1), (2,1), (1,0)

        directions = np.array([[1, 0],
                      [2, 0],
                      [-1, 0],
                      [-2, 0],
                      [1, 1]
                      ])
        sf = [transPoly.compute_support_function(l) for l in directions]
        correct_sf = [2, 4, -1, -2, 4]
        np.testing.assert_almost_equal(sf, correct_sf)

        max_norm = transPoly.compute_max_norm()
        correct_max_norm = 2  # (1,2) on l: (0,1)
        np.testing.assert_almost_equal(max_norm, correct_max_norm)

if __name__ == '__main__':
    unittest.main()
