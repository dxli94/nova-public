import numpy as np

from SysDynamics import SysDynamics


def read_next_block(ins_file):
    line = ins_file.__next__()
    data = []

    while True:
        if line.startswith('#'):
            line = ins_file.__next__()
        elif not len(line.strip('\n').split()):
            break
        else:
            data.extend([float(elem) for elem in line.strip('\n').split()])
            try:
                line = ins_file.__next__()
            except StopIteration:
                break
    return data


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
            for index in range(11):
                data = read_next_block(ins_file)
                if index == 0:
                    direction_type = int(data[0])
                elif index == 1:
                    horizon = float(data[0])
                elif index == 2:
                    sampling_time = float(data[0])
                elif index == 3:
                    opvars = tuple([int(entry) for entry in data])
                elif index == 4:
                    dim = int(data[0])
                elif index == 5:
                    arr = np.array(data).reshape((dim, dim))
                    self.dynamics_matrix_A = arr
                elif index == 6:
                    arr = np.array(data).reshape((dim, dim))
                    self.dynamics_matrix_B = arr
                elif index == 7:
                    arr = np.array(data).reshape(len(data)//dim, dim)
                    self.dynamics_coeff_matrix_U = arr
                elif index == 8:
                    arr = np.array(data).reshape(len(data), 1)
                    self.dynamics_col_vec_U = arr
                elif index == 9:
                    arr = np.array(data).reshape(len(data)//dim, dim)
                    self.init_coeff_matrix_X0 = arr
                elif index == 10:
                    arr = np.array(data).reshape(len(data), 1)
                    self.init_col_vec_X0 = arr

            return direction_type, horizon, sampling_time, opvars, \
                   SysDynamics(dim=dim,
                               init_coeff_matrix_X0=self.init_coeff_matrix_X0,
                               init_col_vec_X0=self.init_col_vec_X0,
                               dynamics_matrix_A=self.dynamics_matrix_A,
                               dynamics_matrix_B=self.dynamics_matrix_B,
                               dynamics_coeff_matrix_U=self.dynamics_coeff_matrix_U,
                               dynamics_col_vec_U=self.dynamics_col_vec_U)


if __name__ == '__main__':
    data_reader = DataReader('../new_instances/sample_box.txt')
    data_reader.read_data()
