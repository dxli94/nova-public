import numpy as np

from SysDynamics import SysDynamics


class DataReader:
    def __init__(self, path2instance):
        self.path = path2instance
        self.dynamics_matrix_A = None
        self.dynamics_matrix_B = None
        self.dynamics_coeff_matrix_U = None
        self.dynamics_col_vec_U = None
        self.init_coeff_matrix_X0 = None
        self.init_col_vec_X0 = None

    def read_data(self):
        with open(self.path, 'r') as ins_file:
            isShape = True  # 0 for shape line; 1 for data line
            index = 0
            for line in ins_file:
                if line.startswith('#') or not line.strip('\n'):
                    continue
                else:
                    if isShape:
                        shape = tuple([int(elem) for elem in line.strip('\n').split()])
                        isShape = not isShape
                    else:
                        data = [float(elem) for elem in line.strip('\n').split()]
                        arr = np.array(data).reshape(shape)

                        if index == 0:
                            self.dynamics_matrix_A = arr
                        elif index == 1:
                            self.dynamics_matrix_B = arr
                        elif index == 2:
                            self.dynamics_coeff_matrix_U = arr
                        elif index == 3:
                            self.dynamics_col_vec_U = arr
                        elif index == 4:
                            self.init_coeff_matrix_X0 = arr
                        elif index == 5:
                            self.init_col_vec_X0 = arr

                        index += 1
                        isShape = not isShape

            return SysDynamics(init_coeff_matrix_X0=self.init_coeff_matrix_X0,
                               init_col_vec_X0=self.init_col_vec_X0,
                               dynamics_matrix_A=self.dynamics_matrix_A,
                               dynamics_matrix_B=self.dynamics_matrix_B,
                               dynamics_coeff_matrix_U=self.dynamics_coeff_matrix_U,
                               dynamics_col_vec_U=self.dynamics_col_vec_U)