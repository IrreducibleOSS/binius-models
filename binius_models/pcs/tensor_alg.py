# (C) 2024 Irreducible Inc.

# Given a finite field extension of binary fields L/K (assumed to be "binary tower fields")
# construct a class to represent computations in A:=L⊗_K L, the *tensor algebra*.
# Many of our computations assume an implicit basis; in the structure of "binary tower fields"
# there is often something like a God-given choice.

from typing import Tuple

# Right now, the code is set up for K = 𝔽_{2⁸} and L = 𝔽_{2¹²⁸}
# To maintain a semblance of modularity, we use SmallFieldElem to refer
# to elements of K and LargeFieldElem to refer to elements of L.
from ..ips.utils import Elem8b as SmallFieldElem
from ..ips.utils import Elem128b as LargeFieldElem


# Information about L / K.
def degree_parameters() -> Tuple[int, int, int, int]:
    large_degree = LargeFieldElem.field.degree
    small_degree = SmallFieldElem.field.degree
    relative_degree = large_degree // small_degree  # We will often write n = 2^κ for the relative degree
    kappa = relative_degree.bit_length() - 1
    return (large_degree, small_degree, relative_degree, kappa)


# An element of L ⊗_K L.
class TensorAlgElem:
    # We explicitly encode L/K and the attendant numerical information.
    large_field = LargeFieldElem.field
    small_field = SmallFieldElem.field
    (large_degree, small_degree, relative_degree, kappa) = degree_parameters()

    def __init__(self, matrix: list[list[SmallFieldElem]]):
        # the representation of the element of L ⊗_K L as a matrix with coefficients in K,
        # with respect to the basis βᵢ ⊗ βⱼ. However, we won't explicitly
        # need the basis (βᵢ).
        assert len(matrix) == self.relative_degree
        assert all(len(matrix[i]) == self.relative_degree for i in range(self.relative_degree))
        self.matrix = matrix

    def __add__(self, other: "TensorAlgElem"):
        return TensorAlgElem(
            [
                [self.matrix[i][j] + other.matrix[i][j] for j in range(len(self.matrix[0]))]
                for i in range(len(self.matrix))
            ]
        )

    def __neg__(self):
        # -a = a because char(K) = 2.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorAlgElem):
            return NotImplemented
        return self.matrix == other.matrix

    @staticmethod
    def random() -> "TensorAlgElem":
        return TensorAlgElem(
            [
                [SmallFieldElem.random() for _ in range(TensorAlgElem.relative_degree)]
                for _ in range(TensorAlgElem.relative_degree)
            ]
        )

    @staticmethod
    def zero() -> "TensorAlgElem":
        return TensorAlgElem(
            [
                [SmallFieldElem.zero() for _ in range(TensorAlgElem.relative_degree)]
                for _ in range(TensorAlgElem.relative_degree)
            ]
        )

    @staticmethod
    def one() -> "TensorAlgElem":
        return TensorAlgElem(
            [
                [
                    SmallFieldElem.one() if i == j and j == 0 else SmallFieldElem.zero()
                    for j in range(TensorAlgElem.relative_degree)
                ]
                for i in range(TensorAlgElem.relative_degree)
            ]
        )

    # swap is the function defined on simple tensors sends x ⊗ y to y ⊗ x and extended linearly.
    # swap only makes sense because we are working with L ⊗_K L. (In other words, it wouldn't
    # make sense with L ⊗_K L' for some other extension L' / K.)
    def swap(self) -> "TensorAlgElem":
        return TensorAlgElem(
            [[self.matrix[j][i] for j in range(self.relative_degree)] for i in range(self.relative_degree)]
        )

    def row_representation(self) -> list[LargeFieldElem]:
        return [large_field_recombination(self.matrix[i]) for i in range(self.relative_degree)]

    def column_representation(self) -> list[LargeFieldElem]:
        return self.swap().row_representation()

    @staticmethod
    def from_row_representation(row: list[LargeFieldElem]) -> "TensorAlgElem":
        assert len(row) == TensorAlgElem.relative_degree, "input list has the wrong length"
        return TensorAlgElem([small_field_expansion(row[i]) for i in range(len(row))])

    @staticmethod
    def from_column_representation(column: list[LargeFieldElem]) -> "TensorAlgElem":
        return TensorAlgElem.from_row_representation(column).swap()

    @staticmethod
    def phi_0(x: LargeFieldElem) -> "TensorAlgElem":
        expansion = small_field_expansion(x)
        matrix = [
            [expansion[i]] + [SmallFieldElem.zero()] * (TensorAlgElem.relative_degree - 1)
            for i in range(TensorAlgElem.relative_degree)
        ]
        return TensorAlgElem(matrix)

    @staticmethod
    def phi_1(y: LargeFieldElem) -> "TensorAlgElem":
        return TensorAlgElem.phi_0(y).swap()

    def mul_by_K(self, k: SmallFieldElem) -> "TensorAlgElem":
        return TensorAlgElem(
            [[self.matrix[i][j] * k for j in range(self.relative_degree)] for i in range(self.relative_degree)]
        )

    # recall that φ₀ : L → A is the map sending x ↦ x ⊗ 1
    # if a ∈ A and x ∈ L, this computes φ₀(x) ⋅ a ∈ A.
    def mul_by_phi_0(self, x: LargeFieldElem) -> "TensorAlgElem":
        old_columns = self.column_representation()
        new_columns = [old_columns[i] * x for i in range(self.relative_degree)]
        return self.from_column_representation(new_columns)

    # recall that φ₁ : L → A is the map sending y ↦ 1 ⊗ y
    # if a ∈ A and y ∈ L, this computes φ₁(y) ⋅ a ∈ A
    def mul_by_phi_1(self, y: LargeFieldElem) -> "TensorAlgElem":
        old_rows = self.row_representation()
        new_rows = [old_rows[i] * y for i in range(self.relative_degree)]
        return self.from_row_representation(new_rows)


# Expand a LargeFieldElem into a vector of SmallFieldElem.
# More precisely, in our internal representation, there is an implicit basis:
# β₀, β₁, … , β_{κ − 1} for L / K. This function takes
# an element of L and returns its coordinates in this basis.
def small_field_expansion(elem: LargeFieldElem) -> list[SmallFieldElem]:
    val = elem.value
    (large_degree, small_degree, relative_degree, kappa) = degree_parameters()
    bit_mask = (1 << small_degree) - 1
    return [SmallFieldElem(val >> small_degree * i & bit_mask) for i in range(relative_degree)]


# Recombine a vector of SmallFieldElem into a LargeFieldElem.
# As in `small_field_expansion`, we assume that the input is given with respect
# a god-given basis: β₀, β₁, … , β_{κ − 1} for L / K.
def large_field_recombination(expansion: list[SmallFieldElem]) -> LargeFieldElem:
    (_, small_degree, relative_degree, _) = degree_parameters()
    assert len(expansion) == relative_degree, "wrong length"
    val = 0
    for i in range(relative_degree):
        val |= expansion[i].value << small_degree * i
    return LargeFieldElem(val)


# Generate the canonical basis for L / K.
def generate_relative_basis() -> list[LargeFieldElem]:
    (_, small_degree, relative_degree, _) = degree_parameters()
    return [LargeFieldElem(1 << small_degree * i) for i in range(relative_degree)]
