# (C) 2024 Irreducible Inc.

from typing import Generic, TypeVar

from binius_models.finite_fields.tower import BinaryTowerFieldElem
from binius_models.ntt.additive_ntt import AdditiveNTT

F = TypeVar("F", bound=BinaryTowerFieldElem)


class VisionMDSTransformation(Generic[F]):
    def __init__(self, field: type[F], log_h: int = 3) -> None:
        if field.field.degree < log_h + 2:
            raise ValueError("field degree must be at least 3 + log_h")
        if log_h < 3:
            raise ValueError("log_h must be at least 3")

        self.additive_ntt = AdditiveNTT(field=field, max_log_h=log_h + 2, log_rate=1)
        self.field = field
        self.log_h = log_h

    def transform(self, input: list[F]) -> list[F]:
        chunk_size = 1 << self.log_h
        assert len(input) == 3 * chunk_size, "length of input must be 3 * 2^log_h"
        constants = self.additive_ntt.constants  # just to save characters

        stash: list[F] = sum(  # start with size-8 inverse NTTs at the cosets 0, 1, and 2
            (
                self.additive_ntt._inverse_transform(input[coset * chunk_size : (coset + 1) * chunk_size], coset)
                for coset in range(3)
            ),
            [],
        )
        result = stash.copy()
        for k in range(chunk_size):
            # the whole purpose of this loop is essentially to do naïvely in the base case of size-3 input
            # what we want to do smartly on size-24 input: namely to take the values of a polynomial of degree < 3
            # at ω̂₀, ω̂₁, ω̂₂, where hat means w.r.t. the "standard" (see Rem. 31 of FRI-Binius) basis of S⁽³⁾,
            # and to return the values of the same polynomial at the points ω̂₃, ω̂₄, ω̂₅ of of S⁽³⁾.
            # we do this essentially by multiplying the 3 points by the inverse of a "Vandermonde" at ω̂₀, ω̂₁, ω̂₂,
            # and then by a further "Vandermonde" at ω̂₀, ω̂₁, ω̂₂. here, "Vandermonde" in quotes means an analogue of
            # that sort of matrix, but where each row gives the evaluations of the (3ʳᵈ-order) novel basis polynomials
            # at the relevant point, as opposed to of the standard monomial basis polynomials, i.e. the monomials.
            # by tricky sub-expression re-use, we are able to get the number of total mults per iteration to 3.
            # for the general interpolate algorithm, see `binius_crates::ntt::odd_interpolate` in the Binius code.

            stash[chunk_size | k] += stash[k]
            temp_0 = constants[self.log_h][1] * stash[chunk_size | k]
            stash[chunk_size << 1 | k] += temp_0 + stash[k]

            temp_1 = constants[self.log_h][2] * stash[chunk_size | k]
            temp_2 = constants[self.log_h + 1][1] * stash[chunk_size << 1 | k]
            result[k] += temp_0 + stash[chunk_size | k] + stash[chunk_size << 1 | k]
            result[chunk_size | k] = stash[k] + temp_1 + temp_2
            result[chunk_size << 1 | k] = result[chunk_size | k] + stash[chunk_size | k]
        return sum(  # standard forward size-8 NTTs at the cosets 3, 4, and 5
            (
                self.additive_ntt._transform(result[coset * chunk_size : (coset + 1) * chunk_size], coset + 3)
                for coset in range(3)
            ),
            [],
        )
