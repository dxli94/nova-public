import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu = 1


def van_der_pol_oscillator_deriv(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 - 1.0) * x[1] - x[0]
    # nx1 = -x[0] - x[1]
    res = np.array([nx0, nx1])
    return res


ts = np.linspace(0, 3, 80)

xs = odeint(van_der_pol_oscillator_deriv, [1.25, -2.3], ts)
print('\n'.join(str(x) for x in list(enumerate(xs))))
plt.plot(xs[:, 0], xs[:, 1])
plt.gca().set_aspect('equal')
# plt.savefig('vanderpol_oscillator.png')
plt.show()