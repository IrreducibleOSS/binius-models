# (C) 2024 Irreducible Inc.

# An Implementation of Ring-Switching protocol.
# On a high level, this allows us to instantiate a "small field" PCS given a large-field PCS with no extra overhead.
# We provide two "output" methods: one for instantiating a `LargeFieldClaim`, and another for passing the output to
# a FRI-Binius instance. In particular, the latter instantiates an end-to-end protocol.

# After one shot, one is reduced to a multilinear evaluation claim over
# A := L ⊗_K L. We then run a sort of induced "row-batched" sumcheck.
# The source for this material is Sections 4 and 5 of [DP24], which has as antecedent [D24].

# A word on notation. In [DP24], ŝ is the A-valued evaluation
# of the multilinear and s₀ is the row-batched version of this claim. We follow this notation.

# [DP24]: https://eprint.iacr.org/2024/504.pdf
# [D24]: https://hackmd.io/@benediamond/BJgKxUau0
from ..ips.sumcheck import Sumcheck

# We are temporarily importing Polynomial128, because the sumcheck code
# uses it. Eventually it will be replaced by Polynomial
from ..ips.utils import Polynomial128
from .fri_binius import FRIBinius

# This code again fixes L / K to be 𝔽_{2¹²⁸} / 𝔽_{2⁸} via the below imports.
# In particular, to work with different fields, one must also change the code in
# the `tensor_alg.py`, `sumcheck.py` and `fri_binius.py` files.
from .tensor_alg import (
    LargeFieldElem,
    SmallFieldElem,
    TensorAlgElem,
    degree_parameters,
    large_field_recombination,
    small_field_expansion,
)


class RingSwitching:
    (large_degree, small_degree, relative_degree, kappa) = degree_parameters()

    def __init__(self, var: int, log_rate: int) -> None:
        self.var = var
        self.fri_binius = FRIBinius(LargeFieldElem, var - self.kappa, log_rate)
        self.challenges: list[LargeFieldElem] = []  # we ONLY need this in `finalize`; see below

    def _pack_polynomial(self, multilinear: list[SmallFieldElem]) -> list[LargeFieldElem]:
        return [
            large_field_recombination([multilinear[v + (w << self.kappa)] for v in range(1 << self.kappa)])
            for w in range(1 << self.fri_binius.var)
        ]

    def commit(self, multilinear: list[SmallFieldElem]) -> None:
        self.multilinear = multilinear  # only will need to use this once below; to compute ŝ
        packed_multilinear = self._pack_polynomial(multilinear)
        self.fri_binius.commit(packed_multilinear)

    def initialize_proof(self, r: list[LargeFieldElem], r_double_prime: list[LargeFieldElem]) -> TensorAlgElem:
        # prepare sumcheck
        tensor_expansion_second_chunk_r = tensor_expansion(r[self.kappa :])
        tensor_expanded_r_double_prime = tensor_expansion(r_double_prime)
        a = [
            small_large_dot_product(
                small_field_expansion(tensor_expansion_second_chunk_r[i]), tensor_expanded_r_double_prime
            )
            for i in range(1 << self.fri_binius.var)
        ]
        composition_poly = Polynomial128(2, {tuple([1, 1]): LargeFieldElem.one()})
        self.sumcheck = Sumcheck([self.fri_binius.multilinear, a], composition_poly, False)
        # compute ŝ := φ₁(t')(φ₀(r_κ), … , φ₀(r_{ℓ − 1})), which is an element of A := L ⊗_K L.
        tensor_expansion_second_chunk_r = tensor_expansion(r[self.kappa :])
        column_representation_s_hat = [
            small_large_dot_product(
                [self.multilinear[w] for w in range(v, 1 << self.var, 1 << self.kappa)],
                tensor_expansion_second_chunk_r,
            )
            for v in range(1 << self.kappa)
        ]
        return TensorAlgElem.from_column_representation(column_representation_s_hat)  # == ŝ

    def advance_state(self) -> list[LargeFieldElem]:
        return self.sumcheck.compute_round_polynomial()

    def receive_challenge(self, r: LargeFieldElem) -> None:
        self.challenges.append(r)
        self.sumcheck.receive_challenge(r)

    def finalize(self) -> LargeFieldElem:
        # prepares the underlying FRI-Binius object to roll.
        self.fri_binius.initialize_proof(self.challenges)  # self.sumcheck.challenges == r'
        # note: the above is wasteful; since internally it will tensor-expand `self.challenges`.
        # we more-or-less already did that, during our sumcheck---at least, before switchover, we started to.
        # in any case, the point of this model is illustrative---we have an artificial separation between the
        # ring-switching class and the underlying L-PCS (which, here, we elect to instantiate using a FRI-Binius).
        # of course in practice, we can, and will, integrate these---i.e., do Section 6, as opposed to Section 4.
        # so this wasted work won't be present in the real thing.
        return self.sumcheck.multilinears[0][0]  # should equal s' === t'(r')!!!


# Given r := (r₀, … , r_{ℓ − 1}), compute the "tensor expansion":
# ⨂_{i = 0}^{ℓ − 1} (1 – rᵢ, rᵢ). This will be a list of length 2^ℓ.
def tensor_expansion(r: list[LargeFieldElem]) -> list[LargeFieldElem]:
    result = [LargeFieldElem.one()] + [LargeFieldElem.zero()] * ((1 << len(r)) - 1)
    for i in range(len(r)):
        for j in range(1 << i):
            result[(1 << i) + j] = result[j] * r[i]
            result[j] -= result[(1 << i) + j]
    return result


def large_large_dot_product(a: list[LargeFieldElem], b: list[LargeFieldElem]) -> LargeFieldElem:
    assert len(a) == len(b), "a and b have different lengths"
    return sum((a_i * b_i for (a_i, b_i) in zip(a, b)), LargeFieldElem.zero())


def small_large_dot_product(a: list[SmallFieldElem], b: list[LargeFieldElem]) -> LargeFieldElem:
    return sum((a_i.upcast(LargeFieldElem) * b_i for (a_i, b_i) in zip(a, b)), LargeFieldElem.zero())


def evaluate_multilinear(multilinear: list[SmallFieldElem], r: list[LargeFieldElem]) -> LargeFieldElem:
    assert 1 << len(r) == len(multilinear), "something's wrong"
    tensor_expanded_input = tensor_expansion(r)
    claim = small_large_dot_product(multilinear, tensor_expanded_input)
    return claim
