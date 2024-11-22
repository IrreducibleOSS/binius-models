from .tensor_alg import (
    LargeFieldElem,
    SmallFieldElem,
    TensorAlgElem,
    degree_parameters,
    generate_relative_basis,
    large_field_recombination,
    small_field_expansion,
)


# test if the expansion and recombination functions are inverses of each other.
# the former expresses an element of L as a linear combination of the basis elements of K,
# and the latter does the reverse.
def test_expansion_recombination():
    elem = LargeFieldElem.random()
    expansion = small_field_expansion(elem)
    recombination = large_field_recombination(expansion)
    assert recombination == elem, "Recombination should be the inverse of expansion."
    k_elem = SmallFieldElem.random()
    expansion = small_field_expansion(LargeFieldElem(k_elem.value))
    assert expansion[0] == k_elem, "The first element of the expansion should be the original element of K."
    assert expansion[1:] == [
        SmallFieldElem.zero() for _ in range(len(expansion) - 1)
    ], "The rest of the expansion should be zero."


# basic tests of the TensorAlgElem class.
def test_TensorAlgElem():
    (_, _, relative_degree, _) = degree_parameters()
    matrix = [[SmallFieldElem.random() for _ in range(relative_degree)] for _ in range(relative_degree)]
    a = TensorAlgElem(matrix)
    ell = LargeFieldElem.random()
    ell_inv = ell.inverse()
    # test that mul_by_phi_0 works correctly
    b = a.mul_by_phi_0(ell).mul_by_phi_0(ell_inv)
    assert a == b
    # test compatibility of mul_by_phi_0 and mul_by_phi_1 when applied to the same element of k.
    k = SmallFieldElem.random()
    k_inv = k.inverse()
    assert a == a.mul_by_phi_0(k).mul_by_phi_1(k_inv)
    assert a.mul_by_K(k) == a.mul_by_phi_0(k)


# test row/column representation
def test_row_column_representation():
    (_, _, relative_degree, _) = degree_parameters()
    vec_k = [SmallFieldElem.random().upcast(LargeFieldElem) for _ in range(relative_degree)]

    # check that from_row_representation and from_column_representation are the same on a vector with elements in K.
    assert TensorAlgElem.from_column_representation(vec_k) == TensorAlgElem.from_row_representation(vec_k).swap()

    # more generally, given an L-vector, from column representation -> contraction
    # yields the correct basis combination.
    vec_l = [LargeFieldElem.random() for _ in range(relative_degree)]
    # operations are involutive
    assert TensorAlgElem.from_column_representation(vec_l).column_representation() == vec_l
    assert TensorAlgElem.from_row_representation(vec_l).row_representation() == vec_l

    # check a naive column representation function that doesn't use the swap method.
    a = TensorAlgElem.random()
    naive_column_representation = [
        large_field_recombination([a.matrix[i][j] for i in range(relative_degree)]) for j in range(relative_degree)
    ]
    assert a.column_representation() == naive_column_representation, "the column representation function has a bug"

    # check that the column representation of 1\otimes beta_i is [0,..,1,0,..0]
    a = TensorAlgElem.phi_0(LargeFieldElem.one())
    basis = generate_relative_basis()
    for i in range(relative_degree):
        b = a.mul_by_phi_1(basis[i])
        columns_b = b.column_representation()
        assert columns_b == [LargeFieldElem.zero() if j != i else LargeFieldElem.one() for j in range(relative_degree)]


# test phi_0(l)*phi_1(l') = phi_1(l')*phi_0(l)
def test_phi_i():
    ell = LargeFieldElem.random()
    a = TensorAlgElem.phi_0(ell)
    ell_prime = LargeFieldElem.random()
    b = TensorAlgElem.phi_1(ell_prime)
    assert a.mul_by_phi_1(ell_prime) == b.mul_by_phi_0(ell)


def test_one():
    a = TensorAlgElem.one()
    b = TensorAlgElem.phi_0(LargeFieldElem.one())
    c = TensorAlgElem.phi_1(LargeFieldElem.one())
    assert a == b, "phi_0(1) should be the same as one."
    assert a == c, "phi_1(1) should be the same as one."
    d = TensorAlgElem.zero()
    assert a + b == d, "one - phi_0(1) should be zero."
