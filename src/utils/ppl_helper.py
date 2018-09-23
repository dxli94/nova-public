from ppl import Variable, Constraint_System, C_Polyhedron
import numpy as np

# in ppl, most things are integer!
normalised_factor = 1e4


def normalised(n):
    return n * normalised_factor


def get_vertices(coeff_matrix, col_vec, dim):
    cs = Constraint_System()
    variables = np.array([Variable(idx) for idx in range(dim)])

    for idx, sf_val in zip(range(len(col_vec)), col_vec):
        cs.insert(normalised(np.dot(coeff_matrix[idx], variables)) <= normalised(sf_val))

    poly = C_Polyhedron(cs)

    generators = poly.minimized_generators()
    vertices = []
    for g in generators:
        v = np.divide([float(g.coefficient(v)) for v in variables], float(g.divisor()))
        vertices.append(v.tolist())

    return vertices


def create_polytope(coeff_matrix, col_vec, dim):
    cs = Constraint_System()
    variables = np.array([Variable(idx) for idx in range(dim)])

    for idx, sf_val in zip(range(len(col_vec)), col_vec):
        cs.insert(normalised(np.dot(coeff_matrix[idx], variables)) <= normalised(sf_val))

    poly = C_Polyhedron(cs)

    return poly


def get_2dVert_from_poly(ppl_poly, dim):
    generators = ppl_poly.minimized_generators()
    vertices = []
    variables = np.array([Variable(idx) for idx in range(dim)])
    for g in generators:
        v = np.divide([float(g.coefficient(v)) for v in variables], float(g.divisor()))
        vertices.append(v.tolist())

    return vertices


def contains(poly_1, poly_2):
    print("poly_1: " + str(poly_1.constraints()) + " contains:")
    print("poly_2: " + str(poly_2.constraints()) + " ?")
    print(poly_1.contains(poly_2))

    return poly_1.contains(poly_2)

if __name__ == '__main__':
    coeff_matrix = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    col_vec = np.array([1, 1, 1, 1])

    # p = create_ppl_polyhedra_from_constraints(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), [1, 1, 1, 1], 2)
    # genarators = p.generators()

    print(get_vertices(coeff_matrix, col_vec))