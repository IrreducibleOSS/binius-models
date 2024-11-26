from typing import Generic, TypeVar

from ..finite_fields.tower import BinaryTowerFieldElem
from ..hashes.vision.utils import matrix_inverse
from .additive_ntt import AdditiveNTT

F = TypeVar("F", bound=BinaryTowerFieldElem)


# We are given a field, a positive integer `ell`, and a positive integer `d`.
# This class assumes that the evaluations of a degree less than d* 2^{ell} polynomial
# are given at the "first  d* 2^{ell} points" of the target domain.
# The main method, interpolate, is to output the coefficients in novel polynomial basis.
class InterpolateNonTwoPrimary(Generic[F]):
    def __init__(self, field: type[F], ell: int, d: int) -> None:
        self.ell = ell
        self.d = d
        # self.log_d is the number of bits we need to represent the numbers 0, .., d-1.
        self.log_d = (d - 1).bit_length()
        log_h = self.log_d + self.ell
        if field.field.degree < log_h:
            raise ValueError("field degree must be at least log(d) + ell")
        self.field = field
        self.additive_ntt = AdditiveNTT(field=field, max_log_h=log_h, log_rate=0)
        self._precompute_inverse_vandermonde()

    def _precompute_inverse_vandermonde(self) -> None:
        # `vandermonde` will be the d × d matrix whose (i, j)ᵗʰ entry is X^(ℓ)ⱼ(zᵢ);
        # i.e., it will be the evaluation of the ℓᵗʰ-order (!) novel basis polynomials on d points of S^(ℓ).
        # here, we write zᵢ for the iᵗʰ element of S^(ℓ), where we enumerate that thing's points in the usual way.
        # that is, we take the "standard" basis for it, Ŵ_ℓ(β_ℓ), Ŵ_ℓ(β_{ℓ + 1}), ..., and lexicographically combine.

        # why do we want that matrix? it's essentially a DFT matrix, on an appropriate higher-order domain.
        # this will be the thing we want to apply the inverse of stride-wise, after doing ℓ iNTT rounds.
        vandermonde = [[self.field.zero() for _ in range(self.d)] for _ in range(self.d)]

        constants = self.additive_ntt.constants
        for j in range(self.log_d):
            for i in range(j, self.log_d):
                # first handle the case of power-of-two-valued i and j; these things are our precomputed constants!
                vandermonde[1 << i][1 << j] = self.field.one() if i == j else constants[j + self.ell][i - j - 1]
                # now, i.e. below, we extend by addition / linearity to the case of non-power-of-two-valued i.
                for k in range(1, min(1 << i, self.d - (1 << i))):  # if 1 << j | k >= self.d: break
                    vandermonde[1 << i | k][1 << j] = vandermonde[k][1 << j] + vandermonde[1 << i][1 << j]

        for i in range(self.d):
            vandermonde[i][0] = self.field.one()
            for j in range(self.log_d):
                # now, for each i, we are going to fill in the values for non-power-of-two-valued j by multiplication.
                # the point here is the multiplicative substructure of the (evaluations of the) novel basis polys.
                for k in range(1, min(1 << j, self.d - (1 << j))):
                    vandermonde[i][1 << j | k] = vandermonde[i][k] * vandermonde[i][1 << j]

        self.inverse_vandermonde = matrix_inverse(vandermonde)  # invert once and for all and memoize

    def interpolate(self, evaluations: list[F]) -> list[F]:
        if len(evaluations) != self.d << self.ell:
            raise ValueError("Input evaluations must be of length d * 2^ell")
        stash = sum(
            (
                self.additive_ntt._inverse_transform(evaluations[coset << self.ell : coset + 1 << self.ell], coset)
                for coset in range(self.d)
            ),
            [],
        )
        result = [self.field.zero() for _ in range(self.d << self.ell)]
        for s in range(1 << self.ell):
            for i in range(self.d):
                for j in range(self.d):
                    result[i << self.ell | s] += self.inverse_vandermonde[i][j] * stash[j << self.ell | s]
        return result
