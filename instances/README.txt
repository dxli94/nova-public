1. Blank lines and those starting with hash symbol will be ignored.

2. The system under analysis is of the form:
        x'(t) = Ax(t) + Bu(t), u(t) \in U, x(0) \in X0
where U and X0 are compact convex sets denoting the sets of inputs U
and initial states X0, respectively.

3. Instance files follow the format below:
   - each non-zero matrix A, B, U_coeff, U_col and X0 are encoded by a matrix spreading on multiple lines:
   each line represents a row of the matrix.
   - A, B, U, X0 come in order. U is specified by:
        U_coeff * x(t) <= U_col,
     similarly, X0 is specified by:
        X0_coeff * x(t) <= X0_col