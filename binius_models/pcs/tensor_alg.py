# (C) 2024 Irreducible Inc.

# Given a finite field extension of binary fields L/K (assumed to be "binary tower fields")
# construct a class to represent computations in A:=L⊗_K L, the *tensor algebra*.
# Many of our computations assume an implicit basis; in the structure of "binary tower fields"
# there is often something like a God-given choice.

from typing import Generic, TypeVar

from ..finite_fields.finite_field import FiniteFieldElem

K = TypeVar("K", bound=FiniteFieldElem)
L = TypeVar("L", bound=FiniteFieldElem)


class TensorAlgElem(Generic[K, L]):
    def __init__(self, small: type[K], large: type[L], columns: list[L]):
        # the representation of the element of L ⊗_K L as a matrix with coefficients in K,
        # with respect to the basis βᵢ ⊗ βⱼ. However, we won't explicitly
        # need the basis (βᵢ).
        self.small = small
        self.large = large

        self.kappa = large.field.degree.bit_length() - small.field.degree.bit_length()  # both powers of 2?

        assert len(columns) == 1 << self.kappa
        self.columns = columns

    def __add__(self, other: "TensorAlgElem[K, L]") -> "TensorAlgElem[K, L]":
        return TensorAlgElem(
            self.small, self.large, [self.columns[i] + other.columns[i] for i in range(1 << self.kappa)]
        )

    def transpose(self) -> "TensorAlgElem[K, L]":
        columns = self.columns
        mask = (1 << self.small.field.degree) - 1
        transposed = [
            sum(
                (
                    self.large((columns[u].value >> v * self.small.field.degree & mask) << u * self.small.field.degree)
                    for u in range(1 << self.kappa)
                ),
                self.large.zero(),
            )
            for v in range(1 << self.kappa)
        ]
        return TensorAlgElem(self.small, self.large, transposed)

    def scale_vertical(self, scalar: L) -> "TensorAlgElem[K, L]":
        return TensorAlgElem(self.small, self.large, [scalar * self.columns[i] for i in range(1 << self.kappa)])

    def scale_horizontal(self, scalar: L) -> "TensorAlgElem[K, L]":
        return self.transpose().scale_vertical(scalar).transpose()
