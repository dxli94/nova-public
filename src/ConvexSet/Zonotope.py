import numpy as np


class Zonotope:
    def __init__(self, c, G):
        self.c = c
        self.G = G
        self.order = 1

    def add(self, zono):
        self.c = np.add(self.c, zono.c)
        self.G.extend(zono.G)
        self.order += zono.order

    def linear_map(self, M):
        # MZ = (Mc, <Mg1, ... Mgk>)
        self.c = np.dot(M, self.c)
        new_G = []
        for g in self.G:
            new_G.append(np.dot(M, g))
        self.G = np.apply_along_axis(np.dot, 0, M)

    def __str__(self):
        return "centre: {}\ngenerators: {}".format(self.c, self.G)


if __name__ == '__main__':
    c1 = [0, 0]
    G1 = [[1, 0], [0, 1]]

    c2 = [1, 1]
    G2 = [[2, 0], [0, 2]]

    M = [[1, 0], [0, 1]]

    zono_1 = Zonotope(c1, G1)
    zono_2 = Zonotope(c2, G2)

    print('zono_1: {}\n'.format(zono_1))
    print('zono_2: {}\n'.format(zono_2))

    zono_1.add(zono_2)
    print('zono_1 add zono_2: {}\n'.format(zono_1))

    zono_2.linear_map(M)
    print('zono_2 after linear map {} is {}\n'.format(M, zono_2))