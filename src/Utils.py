import numpy as np
from scipy.linalg import expm


def mat_exp(A, t):
    return expm(np.multiply(A, t))
