class SysDynamics:
    def __init__(self, matrix_a, matrix_init_coeff, matrix_init_col, matrix_b=None, matrix_input=None):
        # x' = Ax(t) + Bu(t)
        self.matrix_a = matrix_a  # A (2*2) . x(t) (2*1) = Ax(t) (2*1)
        self.matrix_b = matrix_b  # B (2*m) . u(t) (m*1) = Bu(t) (2*1)
        self.matrix_input = matrix_input  # u(t)

        self.matrix_init_coeff = matrix_init_coeff
        self.matrix_init_col = matrix_init_col

    def get_dyn_matrix_a(self):
        return self.matrix_a

    def get_dyn_matrix_b(self):
        return self.matrix_b

    def get_dyn_input_matrix(self):
        return self.matrix_input

    def get_dyn_init(self):
        return self.matrix_init_coeff, self.matrix_init_col
