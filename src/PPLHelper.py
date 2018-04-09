from ppl import Variable, Constraint_System, C_Polyhedron
import numpy as np

normalised_factor = 1000


def normalised(n):
    return n * normalised_factor


def create_ppl_polyhedra_from_support_functions(sf_vec, directions, dim):
    cs = Constraint_System()
    variables = np.array([Variable(idx) for idx in range(dim)])

    for idx, sf_val in zip(range(len(sf_vec)), sf_vec):
        cs.insert(normalised(np.dot(directions[idx], variables)) <= normalised(sf_val))
    return C_Polyhedron(cs)


def contains(poly_1, poly_2):
    # print("poly_1: " + str(poly_1.constraints()) + " contains:")
    # print("poly_2: " + str(poly_2.constraints()) + " ?")
    # print(poly_1.contains(poly_2))

    return poly_1.contains(poly_2)


if __name__ == '__main__':
    create_ppl_polyhedra_from_support_functions([1, 1], np.array([[-1, 0], [1, 0]]), 2)
