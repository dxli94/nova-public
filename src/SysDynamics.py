class SysDynamics:
    def __init__(self, dim, init_coeff_matrix_X0, init_col_vec_X0, dynamics_matrix_A=None, dynamics_matrix_B=None,
                 dynamics_coeff_matrix_U=None, dynamics_col_vec_U=None):
        # x' = Ax(t) + Bu(t)
        self.dim = dim
        self.matrix_A = dynamics_matrix_A  # A (2*2) . x(t) (2*1) = Ax(t) (2*1)
        self.matrix_B = dynamics_matrix_B  # B (2*m) . u(t) (m*1) = Bu(t) (2*1)
        self.coeff_matrix_U = dynamics_coeff_matrix_U  # u(t) coeff matrix
        self.col_vec_U = dynamics_col_vec_U  # u(t) col vec

        self.init_coeff_matrix = init_coeff_matrix_X0
        self.init_col_vec = init_col_vec_X0

    def get_dim(self):
        return self.dim

    def get_dyn_coeff_matrix_A(self):
        return self.matrix_A

    def get_dyn_matrix_B(self):
        return self.matrix_B

    def get_dyn_coeff_matrix_U(self):
        return self.coeff_matrix_U

    def get_dyn_col_vec_U(self):
        return self.col_vec_U

    def get_dyn_init_X0(self):
        return self.init_coeff_matrix, self.init_col_vec
